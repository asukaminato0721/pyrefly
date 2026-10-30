/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice::from_ref;

use lsp_types::TextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::class::Class;
use pyrefly_types::literal::Lit;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::type_output::AnnotationPart;
use pyrefly_types::types::Type;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::error::suppress::detect_line_ending;
use crate::state::import_tracker::ImportTracker;
use crate::state::state::Transaction;

impl Transaction<'_> {
    /// Generates a forwarding stub only for an unfinished inherited instance method.
    /// Unsupported signatures retain the existing name-only completion.
    pub(crate) fn method_override_edits(
        &self,
        handle: &Handle,
        class: &Class,
        method: &StmtFunctionDef,
        attr: &AttrInfo,
        stdlib: &Stdlib,
    ) -> Option<(TextEdit, Vec<TextEdit>)> {
        let AttrDefinition::FullyResolved { cls, .. } = &attr.definition else {
            return None;
        };
        if cls == class || method.is_async || !method.decorator_list.is_empty() {
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
        let module = self.get_module_info(handle)?;
        let source = module.contents();
        let start = method.range().start().to_usize();
        let name_end = method.name.range().end().to_usize();
        let line_end = source[name_end..]
            .find('\n')
            .map_or(source.len(), |end| name_end + end);
        if source[start..name_end] != format!("def {}", method.name)
            || !source[name_end..line_end].trim().is_empty()
        {
            return None;
        }
        let line_start = source[..start].rfind('\n').map_or(0, |start| start + 1);
        let indent = &source[line_start..start];
        if !indent.chars().all(|ch| ch == ' ' || ch == '\t') {
            return None;
        }
        let body_indent = if indent.contains('\t') { "\t" } else { "    " };
        let newline = detect_line_ending(source);
        let Params::List(params) = &function.signature.params else {
            return None;
        };
        let (receiver, rest) = params.items().split_first()?;
        if !matches!(receiver, Param::Pos(..) | Param::PosOnly(Some(_), ..)) {
            return None;
        }

        let mut parts = vec![AnnotationPart::Text(format!("{}(", attr.name))];
        let mut arguments = Vec::new();
        let mut positional_only = false;
        let mut keyword_only = false;
        for (index, param) in params.items().iter().enumerate() {
            let name = param.name()?;
            if name.as_str() == "super" {
                return None;
            }
            if index != 0 {
                parts.push(AnnotationPart::Text(", ".to_owned()));
            }
            if positional_only && !matches!(param, Param::PosOnly(..)) {
                parts.push(AnnotationPart::Text("/, ".to_owned()));
                positional_only = false;
            }
            if !keyword_only && matches!(param, Param::KwOnly(..)) {
                parts.push(AnnotationPart::Text("*, ".to_owned()));
                keyword_only = true;
            }
            let (prefix, required) = match param {
                Param::PosOnly(_, _, required) => {
                    positional_only = true;
                    ("", Some(required))
                }
                Param::Pos(_, _, required) | Param::KwOnly(_, _, required) => ("", Some(required)),
                Param::Varargs(..) => {
                    keyword_only = true;
                    ("*", None)
                }
                Param::Kwargs(..) => ("**", None),
            };
            parts.push(AnnotationPart::Text(format!("{prefix}{name}")));
            if index != 0 && !param.as_type().is_any() {
                parts.push(AnnotationPart::Text(": ".to_owned()));
                parts.extend(override_annotation(param.as_type(), stdlib)?);
            }
            if let Some(Required::Optional(default)) = required {
                // Only immutable scalar defaults can be reconstructed without
                // evaluating or copying an expression from another module.
                let default = match &default.as_ref()?.ty {
                    Type::None => "None".to_owned(),
                    Type::Literal(lit) if !matches!(lit.value, Lit::Enum(_)) => {
                        lit.value.to_string()
                    }
                    _ => return None,
                };
                parts.push(AnnotationPart::Text(format!(" = {default}")));
            }
        }
        if positional_only {
            parts.push(AnnotationPart::Text(", /".to_owned()));
        }
        parts.push(AnnotationPart::Text(")".to_owned()));
        if !flags.is_return_inferred {
            parts.push(AnnotationPart::Text(" -> ".to_owned()));
            parts.extend(override_annotation(&function.signature.ret, stdlib)?);
        }
        for param in rest {
            let name = param.name()?;
            arguments.push(match param {
                Param::PosOnly(..) | Param::Pos(..) => name.to_string(),
                Param::Varargs(..) => format!("*{name}"),
                Param::KwOnly(..) => format!("{name}={name}"),
                Param::Kwargs(..) => format!("**{name}"),
            });
        }

        let ast = self.get_ast(handle)?;
        let tracker = ImportTracker::from_ast(
            &ast,
            handle.module(),
            handle.path().is_init(),
            *handle.sys_info(),
        );
        let override_module = ModuleName::from_str("typing");
        // The import lookup occurs only after the signature and defaults qualify.
        let override_part = AnnotationPart::Reference {
            module: override_module,
            name: "override".to_owned(),
        };
        let mut all_parts = parts.clone();
        all_parts.push(override_part.clone());
        let candidates = tracker.direct_import_candidates(&all_parts, handle.module());
        let mut imports = String::new();
        for (module_name, name) in &candidates {
            let imported = self.import_handle(handle, *module_name, None).finding()?;
            if !self.get_exports(&imported).contains_key(&Name::new(name)) {
                return None;
            }
            imports.push_str(&format!("from {module_name} import {name}{newline}"));
        }
        let direct = candidates
            .iter()
            .map(|(module, head)| (*module, head.as_str()))
            .collect::<Vec<_>>();
        let (signature, missing) = tracker.resolve_annotation(&parts, handle.module(), &direct);
        let (decorator, missing_override) =
            tracker.resolve_annotation(&[override_part], handle.module(), &direct);
        if !missing.is_empty() || !missing_override.is_empty() {
            return None;
        }
        let answers = self.get_answers(handle)?;
        let bindings = answers.bindings();
        let fields = bindings.get_class_fields(class.index())?;
        for part in &all_parts {
            if let AnnotationPart::Reference { .. } = part {
                let (name, _) =
                    tracker.resolve_annotation(from_ref(part), handle.module(), &direct);
                let head = name.split('.').next()?;
                if fields.fields().any(|field| field.as_str() == head) {
                    return None;
                }
            }
        }
        let call = format!("super().{}({})", attr.name, arguments.join(", "));
        let body = if attr.name.as_str() == "__init__" {
            call
        } else {
            format!("return {call}")
        };
        let edit = TextEdit {
            range: module.to_lsp_range(method.name.range()),
            new_text: format!("{signature}:{newline}{indent}{body_indent}{body}"),
        };
        let mut additional = vec![TextEdit {
            range: module.to_lsp_range(TextRange::empty(method.range().start())),
            new_text: format!("@{decorator}{newline}{indent}"),
        }];
        if !imports.is_empty() {
            // Insert normal imports after the module docstring and all future imports.
            let position = ast.body.iter().find(|stmt| {
                !is_docstring_stmt(stmt) && !matches!(stmt,
                    Stmt::ImportFrom(import) if import.module.as_ref().is_some_and(|name| name.as_str() == "__future__")
                )
            })?.range().start();
            additional.push(TextEdit {
                range: module.to_lsp_range(TextRange::empty(position)),
                new_text: imports,
            });
        }
        Some((edit, additional))
    }
}

/// Limits annotations to types with source syntax in the annotation renderer.
fn override_annotation(ty: &Type, stdlib: &Stdlib) -> Option<Vec<AnnotationPart>> {
    if ty.any(|ty| {
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
    }) {
        return None;
    }
    Some(ty.get_annotation_parts(Some(stdlib)))
}
