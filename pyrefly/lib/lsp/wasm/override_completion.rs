/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::OnceCell;

use lsp_types::TextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::class::Class;
use pyrefly_types::env::TypeEnv;
use pyrefly_types::literal::Lit;
use pyrefly_types::type_output::AnnotationPart;
use pyrefly_types::types::Type;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::error::suppress::detect_line_ending;
use crate::state::annotation_renderer::AnnotationRenderer;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::MemberRenderer;
use crate::state::lsp::line_indent_and_start;
use crate::state::lsp::override_in_scope;
use crate::state::state::Transaction;

/// Shares annotation and import planning across candidates in one completion request.
pub(crate) struct OverrideCompletion<'a, 'state> {
    transaction: &'a Transaction<'state>,
    handle: &'a Handle,
    module: Module,
    method: &'a StmtFunctionDef,
    annotations: AnnotationRenderer<'a, 'state>,
    indent: String,
    override_in_scope: bool,
    override_part: OnceCell<Option<AnnotationPart>>,
}

impl<'a, 'state> OverrideCompletion<'a, 'state> {
    /// Only unfinished, undecorated instance-method declarations can receive a stub.
    pub(crate) fn new(
        transaction: &'a Transaction<'state>,
        handle: &'a Handle,
        method: &'a StmtFunctionDef,
        import_format: ImportFormat,
    ) -> Option<Self> {
        if method.is_async || !method.decorator_list.is_empty() {
            return None;
        }
        let module = transaction.get_module_info(handle)?;
        let source = module.contents();
        let start = method.start().to_usize();
        let name_end = method.name.end().to_usize();
        let line_end = source[name_end..]
            .find('\n')
            .map_or(source.len(), |end| name_end + end);
        if source[start..name_end] != format!("def {}", method.name)
            || !source[name_end..line_end].trim().is_empty()
        {
            return None;
        }
        let (indent, line_start) = line_indent_and_start(source, method.start())?;
        if source[line_start.to_usize()..start] != indent {
            return None;
        }
        let ast = transaction.get_ast(handle)?;
        // Decorators execute before any following module imports.
        if ast
            .body
            .iter()
            .filter(|stmt| stmt.start() > method.start())
            .any(|stmt| match stmt {
                Stmt::Import(import) => import
                    .names
                    .iter()
                    .any(|alias| matches!(alias.name.as_str(), "typing" | "typing_extensions")),
                Stmt::ImportFrom(import) => {
                    import.level == 0
                        && import.module.as_ref().is_some_and(|module| {
                            matches!(module.as_str(), "typing" | "typing_extensions")
                        })
                }
                _ => false,
            })
        {
            return None;
        }
        let override_in_scope = override_in_scope(&ast);
        let annotations = AnnotationRenderer::new(transaction, handle, ast, import_format);
        Some(Self {
            transaction,
            handle,
            module,
            method,
            annotations,
            indent,
            override_in_scope,
            override_part: OnceCell::new(),
        })
    }

    /// Unsupported signatures retain the existing name-only completion.
    pub(crate) fn edits(
        &self,
        class: &Class,
        attr: &AttrInfo,
        env: &dyn TypeEnv,
    ) -> Option<(TextEdit, Vec<TextEdit>)> {
        let AttrDefinition::FullyResolved { cls, .. } = &attr.definition else {
            return None;
        };
        if cls == class {
            return None;
        }
        // A Function excludes overloaded and generic methods. Descriptors require
        // different forwarding rules and are outside this completion's scope.
        let Type::Function(function) = attr.ty.as_ref()? else {
            return None;
        };
        let flags = &function.metadata.flags;
        if flags.is_overload
            || flags.is_async
            || flags.is_staticmethod
            || flags.is_classmethod
            || flags.is_cached_property
            || flags.is_abstract_method
            || flags.property_metadata.is_some()
            || flags.has_final_decoration
            || matches!(
                attr.name.as_str(),
                "__new__" | "__init_subclass__" | "__class_getitem__"
            )
        {
            return None;
        }
        let Params::List(params) = &function.signature.params else {
            return None;
        };
        let (receiver, rest) = params.items().split_first()?;
        if !matches!(receiver, Param::Pos(..) | Param::PosOnly(Some(_), ..)) {
            return None;
        }
        for (i, param) in params.items().iter().enumerate() {
            if param.name()?.as_str() == "super"
                || (matches!(param, Param::KwOnly(..)) && Ast::is_mangled_attr(param.name()?))
                || (i != 0 && !supported_annotation(param.as_type()))
            {
                return None;
            }
            if let Param::PosOnly(_, _, Required::Optional(default))
            | Param::Pos(_, _, Required::Optional(default))
            | Param::KwOnly(_, _, Required::Optional(default)) = param
            {
                // Forwarding must not replace a parent's default with a placeholder.
                match default.as_ref()?.ty.as_ref()? {
                    Type::None => {}
                    Type::Literal(lit) if !matches!(lit.value, Lit::Enum(_)) => {}
                    _ => return None,
                }
            }
        }
        if !flags.is_return_inferred && !supported_annotation(&function.signature.ret) {
            return None;
        }
        let mut renderer = MemberRenderer::new(env, self.handle.sys_info().version(), true);
        renderer.text(&format!("{}(", attr.name));
        renderer.params(params, true);
        renderer.text(")");
        if !flags.is_return_inferred {
            renderer.text(" -> ");
            renderer.annotation(&function.signature.ret);
        }
        let parts = renderer.into_parts();
        // Resolve the decorator only after a signature qualifies, once per request.
        let override_part = self
            .override_part
            .get_or_init(|| {
                if self.override_in_scope {
                    return Some(AnnotationPart::Text("override".to_owned()));
                }
                let module = ModuleName::from_str("typing");
                let imported = self
                    .transaction
                    .import_handle(self.handle, module, None)
                    .finding()?;
                self.transaction
                    .get_exports(&imported)
                    .contains_key(&Name::new("override"))
                    .then(|| AnnotationPart::Reference {
                        module,
                        name: "override".to_owned(),
                    })
            })
            .as_ref()?;
        let parts = [
            AnnotationPart::Text("@".to_owned()),
            override_part.clone(),
            AnnotationPart::Text("\n".to_owned()),
        ]
        .into_iter()
        .chain(parts)
        .collect::<Vec<_>>();
        let rendered = self.annotations.render_parts(&parts)?;
        let answers = self.transaction.get_answers(self.handle)?;
        let fields = answers.bindings().get_class_fields(class.index())?;
        if fields.fields().any(|field| {
            rendered.reference_roots.contains(field.as_str())
                || (self.override_in_scope && field.as_str() == "override")
        }) {
            return None;
        }
        let (decorator, signature) = rendered
            .text
            .split_once('\n')
            .expect("the decorator is followed by a newline");
        let arguments = rest
            .iter()
            .map(|param| {
                let name = param.name().expect("parameter names were checked above");
                match param {
                    Param::PosOnly(..) | Param::Pos(..) => name.to_string(),
                    Param::Varargs(..) => format!("*{name}"),
                    Param::KwOnly(..) => format!("{name}={name}"),
                    Param::Kwargs(..) => format!("**{name}"),
                }
            })
            .collect::<Vec<_>>();
        let call = format!("super().{}({})", attr.name, arguments.join(", "));
        let body = if attr.name.as_str() == "__init__" {
            call
        } else {
            format!("return {call}")
        };
        let newline = detect_line_ending(self.module.contents());
        let indent = &self.indent;
        let body_indent = if indent.contains('\t') { "\t" } else { "    " };
        let edit = TextEdit {
            range: self.module.to_lsp_range(self.method.name.range()),
            new_text: format!("{signature}:{newline}{indent}{body_indent}{body}"),
        };
        let mut additional = self
            .annotations
            .import_edits(&rendered)
            .into_iter()
            .map(|(range, text)| TextEdit {
                range: self.module.to_lsp_range(range),
                new_text: text.replace('\n', newline),
            })
            .collect::<Vec<_>>();
        additional.push(TextEdit {
            range: self
                .module
                .to_lsp_range(TextRange::empty(self.method.start())),
            new_text: format!("{decorator}{newline}{indent}"),
        });
        Some((edit, additional))
    }
}

/// Limits annotations to types with source syntax in the annotation renderer.
fn supported_annotation(ty: &Type) -> bool {
    !ty.any(|ty| {
        !matches!(
            ty,
            Type::ClassType(_)
                | Type::None
                | Type::Any(_)
                | Type::Union(_)
                | Type::Tuple(_)
                | Type::Type(_)
                | Type::Literal(_)
                | Type::LiteralString(_)
                | Type::Never(_)
                | Type::SelfType(_)
        )
    })
}
