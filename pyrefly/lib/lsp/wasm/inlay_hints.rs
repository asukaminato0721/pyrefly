/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter::once;
use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_types::literal::Lit;
use pyrefly_types::literal::LitEnum;
use pyrefly_types::literal::Literal;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ModModule;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::binding::UnpackedPosition;
use crate::binding::bindings::Bindings;
use crate::state::lsp::AllOffPartial;
use crate::state::lsp::AnnotationKind;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::InlayHintConfig;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;
use crate::types::callable::Param;
use crate::types::callable::Params;
use crate::types::tuple::Tuple;
use crate::types::typed_dict::TypedDict;
use crate::types::types::Forallable;
use crate::types::types::Type;
use crate::types::types::Union;

pub struct InlayHintData {
    pub position: TextSize,
    /// Label parts with optional location info for click-to-navigate
    pub label_parts: Vec<(String, Option<TextRangeWithModule>)>,
    /// Text inserted when the hint is accepted.
    pub insert_text: Option<String>,
    /// Extra imports needed to make the inserted text valid Python.
    pub insert_imports: Vec<String>,
}

#[derive(Debug)]
pub struct ParameterAnnotation {
    pub text_size: TextSize,
    pub has_annotation: bool,
    pub ty: Option<Type>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamNameMatch<'param> {
    pub name: &'param Name,
    pub is_vararg_repeat: bool,
}

impl<'param> ParamNameMatch<'param> {
    fn new(name: &'param Name, is_vararg_repeat: bool) -> Self {
        Self {
            name,
            is_vararg_repeat,
        }
    }
}

const CALLABLE_IMPORT: &str = "from collections.abc import Callable\n";

fn annotation_text(ty: &Type) -> (String, bool) {
    match ty {
        Type::Callable(callable) => callable_annotation_text(callable),
        Type::Function(function) => callable_annotation_text(&function.signature),
        Type::BoundMethod(bound_method) => match &bound_method.func {
            crate::types::types::BoundMethodType::Function(function) => {
                callable_annotation_text(&function.signature)
            }
            crate::types::types::BoundMethodType::Forall(forall) => {
                callable_annotation_text(&forall.body.signature)
            }
            crate::types::types::BoundMethodType::Overload(_) => (ty.to_string(), false),
        },
        Type::Forall(forall) => match &forall.body {
            Forallable::Callable(callable) => callable_annotation_text(callable),
            Forallable::Function(function) => callable_annotation_text(&function.signature),
            Forallable::TypeAlias(_) => (ty.to_string(), false),
        },
        Type::ClassType(class_type) => {
            let (targs, needs_callable_import) = targs_annotation_text(class_type.targs());
            let mut text = class_type.name().to_string();
            if !targs.is_empty() {
                text.push('[');
                text.push_str(&targs.join(", "));
                text.push(']');
            }
            (text, needs_callable_import)
        }
        Type::TypedDict(TypedDict::TypedDict(typed_dict))
        | Type::PartialTypedDict(TypedDict::TypedDict(typed_dict)) => {
            let (targs, needs_callable_import) = targs_annotation_text(typed_dict.targs());
            let mut text = typed_dict.name().to_string();
            if !targs.is_empty() {
                text.push('[');
                text.push_str(&targs.join(", "));
                text.push(']');
            }
            (text, needs_callable_import)
        }
        Type::Union(box Union { members, .. }) => {
            let mut needs_callable_import = false;
            let members = members
                .iter()
                .map(|member| {
                    let (text, member_needs_callable_import) = annotation_text(member);
                    needs_callable_import |= member_needs_callable_import;
                    text
                })
                .collect::<Vec<_>>();
            (members.join(" | "), needs_callable_import)
        }
        Type::Tuple(tuple) => tuple_annotation_text(tuple),
        Type::Type(inner) => {
            let (text, needs_callable_import) = annotation_text(inner);
            (format!("type[{text}]"), needs_callable_import)
        }
        Type::TypeForm(inner) => {
            let (text, needs_callable_import) = annotation_text(inner);
            (format!("TypeForm[{text}]"), needs_callable_import)
        }
        Type::TypeGuard(inner) => {
            let (text, needs_callable_import) = annotation_text(inner);
            (format!("TypeGuard[{text}]"), needs_callable_import)
        }
        Type::TypeIs(inner) => {
            let (text, needs_callable_import) = annotation_text(inner);
            (format!("TypeIs[{text}]"), needs_callable_import)
        }
        Type::Annotated(inner, metadata) => {
            let (inner_text, mut needs_callable_import) = annotation_text(inner);
            let mut parts = vec![inner_text];
            for item in metadata.iter() {
                let (text, item_needs_callable_import) = annotation_text(item);
                needs_callable_import |= item_needs_callable_import;
                parts.push(text);
            }
            (
                format!("Annotated[{}]", parts.join(", ")),
                needs_callable_import,
            )
        }
        Type::Unpack(inner) => {
            let (text, needs_callable_import) = annotation_text(inner);
            (format!("Unpack[{text}]"), needs_callable_import)
        }
        Type::Concatenate(args, tail) => {
            let mut needs_callable_import = false;
            let mut parts = args
                .iter()
                .map(|(arg, _)| {
                    let (text, arg_needs_callable_import) = annotation_text(arg);
                    needs_callable_import |= arg_needs_callable_import;
                    text
                })
                .collect::<Vec<_>>();
            let (tail_text, tail_needs_callable_import) = annotation_text(tail);
            needs_callable_import |= tail_needs_callable_import;
            parts.push(tail_text);
            (
                format!("Concatenate[{}]", parts.join(", ")),
                needs_callable_import,
            )
        }
        Type::Intersect(box (_, fallback)) => annotation_text(fallback),
        _ => (ty.to_string(), false),
    }
}

fn callable_annotation_text(callable: &crate::types::callable::Callable) -> (String, bool) {
    let (params_text, _) = callable_params_annotation_text(&callable.params);
    let (return_text, _) = annotation_text(&callable.ret);
    (format!("Callable[{params_text}, {return_text}]"), true)
}

fn callable_params_annotation_text(params: &Params) -> (String, bool) {
    match params {
        Params::List(param_list) => {
            if param_list.items().iter().any(|param| {
                matches!(
                    param,
                    Param::VarArg(..) | Param::KwOnly(..) | Param::Kwargs(..)
                )
            }) {
                return ("...".to_owned(), false);
            }
            let mut needs_callable_import = false;
            let params = param_list
                .items()
                .iter()
                .map(|param| {
                    let (text, param_needs_callable_import) = annotation_text(param.as_type());
                    needs_callable_import |= param_needs_callable_import;
                    text
                })
                .collect::<Vec<_>>();
            (format!("[{}]", params.join(", ")), needs_callable_import)
        }
        Params::Ellipsis | Params::Materialization => ("...".to_owned(), false),
        Params::ParamSpec(args, param_spec) => {
            let mut needs_callable_import = false;
            let mut prefix = args
                .iter()
                .map(|(arg, _)| {
                    let (text, arg_needs_callable_import) = annotation_text(arg);
                    needs_callable_import |= arg_needs_callable_import;
                    text
                })
                .collect::<Vec<_>>();
            let text = match param_spec {
                Type::Ellipsis if prefix.is_empty() => "...".to_owned(),
                Type::Ellipsis => {
                    prefix.push("...".to_owned());
                    format!("Concatenate[{}]", prefix.join(", "))
                }
                _ if prefix.is_empty() => {
                    let (text, param_spec_needs_callable_import) = annotation_text(param_spec);
                    needs_callable_import |= param_spec_needs_callable_import;
                    text
                }
                _ => {
                    let (text, param_spec_needs_callable_import) = annotation_text(param_spec);
                    needs_callable_import |= param_spec_needs_callable_import;
                    prefix.push(text);
                    format!("Concatenate[{}]", prefix.join(", "))
                }
            };
            (text, needs_callable_import)
        }
    }
}

fn tuple_annotation_text(tuple: &Tuple) -> (String, bool) {
    match tuple {
        Tuple::Concrete(elements) => {
            let mut needs_callable_import = false;
            let elements = elements
                .iter()
                .map(|element| {
                    let (text, element_needs_callable_import) = annotation_text(element);
                    needs_callable_import |= element_needs_callable_import;
                    text
                })
                .collect::<Vec<_>>();
            let body = if elements.is_empty() {
                "()".to_owned()
            } else {
                elements.join(", ")
            };
            (format!("tuple[{body}]"), needs_callable_import)
        }
        Tuple::Unbounded(element) => {
            let (text, needs_callable_import) = annotation_text(element);
            (format!("tuple[{text}, ...]"), needs_callable_import)
        }
        Tuple::Unpacked(box (prefix, unpacked, suffix)) => {
            let mut needs_callable_import = false;
            let mut parts = prefix
                .iter()
                .map(|element| {
                    let (text, element_needs_callable_import) = annotation_text(element);
                    needs_callable_import |= element_needs_callable_import;
                    text
                })
                .collect::<Vec<_>>();
            let (unpacked_text, unpacked_needs_callable_import) = annotation_text(unpacked);
            needs_callable_import |= unpacked_needs_callable_import;
            parts.push(format!("*{unpacked_text}"));
            for element in suffix {
                let (text, element_needs_callable_import) = annotation_text(element);
                needs_callable_import |= element_needs_callable_import;
                parts.push(text);
            }
            (
                format!("tuple[{}]", parts.join(", ")),
                needs_callable_import,
            )
        }
    }
}

fn targs_annotation_text(targs: &crate::types::types::TArgs) -> (Vec<String>, bool) {
    let mut needs_callable_import = false;
    let args = targs
        .as_slice()
        .iter()
        .take(targs.display_count())
        .map(|arg| {
            let (text, arg_needs_callable_import) = annotation_text(arg);
            needs_callable_import |= arg_needs_callable_import;
            text
        })
        .collect();
    (args, needs_callable_import)
}

fn insert_text_and_imports(prefix: &str, ty: &Type) -> (String, Vec<String>) {
    let (text, needs_callable_import) = annotation_text(ty);
    let insert_imports = if needs_callable_import {
        vec![CALLABLE_IMPORT.to_owned()]
    } else {
        Vec::new()
    };
    (format!("{prefix}{text}"), insert_imports)
}

// Re-export normalize_singleton_function_type_into_params which is shared with signature help
pub fn normalize_singleton_function_type_into_params(type_: Type) -> Option<Vec<Param>> {
    let callable = type_.to_callable()?;
    // We will drop the self parameter for signature help
    if let Params::List(params_list) = callable.params {
        if let Some(Param::PosOnly(Some(name), _, _) | Param::Pos(name, _, _)) =
            params_list.items().first()
            && (name.as_str() == "self" || name.as_str() == "cls" || name.as_str() == "_cls")
        {
            let mut params = params_list.into_items();
            params.remove(0);
            return Some(params);
        }
        return Some(params_list.into_items());
    }
    None
}

impl<'a> Transaction<'a> {
    pub fn inlay_hints(
        &self,
        handle: &Handle,
        inlay_hint_config: InlayHintConfig,
    ) -> Option<Vec<InlayHintData>> {
        let is_interesting = |e: &Expr, ty: &Type, class_name: Option<&Name>| {
            !ty.is_any()
                && match e {
                    Expr::Tuple(tuple) => {
                        !tuple.elts.is_empty() && tuple.elts.iter().all(|x| !Ast::is_literal(x))
                    }
                    Expr::Call(ExprCall { func, .. }) => {
                        // Exclude constructor calls
                        if let Expr::Name(name) = &**func
                            && let Some(class_name) = class_name
                        {
                            *name.id() != *class_name
                        } else if let Expr::Attribute(attr) = &**func
                            && let Some(class_name) = class_name
                        {
                            *attr.attr.id() != *class_name
                        } else {
                            true
                        }
                    }
                    Expr::Attribute(ExprAttribute {
                        box value, attr, ..
                    }) if let Type::Literal(box Literal {
                        value: Lit::Enum(box LitEnum { class, member, .. }),
                        ..
                    }) = ty =>
                    {
                        // Exclude enum literals
                        match value {
                            Expr::Name(object) => {
                                *object.id() != *class.name() || *attr.id() != *member
                            }
                            Expr::Attribute(ExprAttribute { attr: object, .. }) => {
                                *object.id() != *class.name() || *attr.id() != *member
                            }
                            _ => true,
                        }
                    }
                    _ => !Ast::is_literal(e),
                }
        };
        let bindings = self.get_bindings(handle)?;
        let stdlib = self.get_stdlib(handle);
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                key @ Key::ReturnType(id) => {
                    if inlay_hint_config.function_return_types {
                        match bindings.get(bindings.key_to_idx(&Key::Definition(*id))) {
                            Binding::Function(x, _pred, _class_meta) => {
                                if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                    && let Some(mut ty) = self.get_type(handle, key)
                                    && !ty.is_any()
                                {
                                    let fun = bindings.get(bindings.get(*x).undecorated_idx);
                                    if fun.def.is_async
                                        && let Some(Some((_, _, return_ty))) = self.ad_hoc_solve(
                                            handle,
                                            "inlay_hint_coroutine",
                                            |solver| solver.unwrap_coroutine(&ty),
                                        )
                                    {
                                        ty = return_ty;
                                    }
                                    // Use get_types_with_locations to get type parts with location info
                                    let type_parts = ty.get_types_with_locations(Some(&stdlib));
                                    let label_parts = once((" -> ".to_owned(), None))
                                        .chain(
                                            type_parts
                                                .iter()
                                                .map(|(text, loc)| (text.clone(), loc.clone())),
                                        )
                                        .collect();
                                    let (insert_text, insert_imports) =
                                        insert_text_and_imports(" -> ", &ty);
                                    res.push(InlayHintData {
                                        position: fun.def.parameters.range.end(),
                                        label_parts,
                                        insert_text: Some(insert_text),
                                        insert_imports,
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
                key @ Key::Definition(_)
                    if inlay_hint_config.variable_types
                        && let Some(ty) = self.get_type(handle, key) =>
                {
                    // For unpacked values, extract the element expression if available
                    let (e, is_unpacked) = match bindings.get(idx) {
                        Binding::NameAssign(x) if x.annotation.is_none() => (Some(&*x.expr), false),
                        Binding::Expr(None, e) => (Some(&**e), false),
                        Binding::UnpackedValue(None, unpack_idx, _, pos) => {
                            // Try to get the element expression from the unpacked source
                            let element_expr =
                                Self::get_unpacked_element_expr(&bindings, *unpack_idx, *pos);
                            (element_expr, true)
                        }
                        _ => (None, false),
                    };
                    // If the inferred type is a class type w/ no type arguments and the
                    // RHS is a call to a function that's the same name as the inferred class,
                    // we assume it's a constructor and do not display an inlay hint
                    let class_name = if let Type::ClassType(cls) = &ty
                        && cls.targs().is_empty()
                    {
                        Some(cls.name())
                    } else {
                        None
                    };
                    // For unpacked values without a known element expression (e.g., from
                    // function calls or nested unpacking), show the hint if the type is not Any.
                    // For regular assignments, require the expression to be interesting.
                    let should_show = if let Some(e) = e {
                        is_interesting(e, &ty, class_name)
                    } else {
                        // For unpacked values where we couldn't extract the element,
                        // show hint if type is not Any
                        is_unpacked && !ty.is_any()
                    };
                    if should_show {
                        // Use get_types_with_locations to get type parts with location info
                        let type_parts = ty.get_types_with_locations(Some(&stdlib));
                        let label_parts = once((": ".to_owned(), None))
                            .chain(
                                type_parts
                                    .iter()
                                    .map(|(text, loc)| (text.clone(), loc.clone())),
                            )
                            .collect();
                        let (insert_text, insert_imports) = if is_unpacked {
                            (String::new(), Vec::new())
                        } else {
                            insert_text_and_imports(": ", &ty)
                        };
                        res.push(InlayHintData {
                            position: key.range().end(),
                            label_parts,
                            insert_text: (!is_unpacked).then_some(insert_text),
                            insert_imports,
                        });
                    }
                }
                _ => {}
            }
        }

        if inlay_hint_config.call_argument_names != AllOffPartial::Off {
            res.extend(
                self.add_inlay_hints_for_positional_function_args(handle)
                    .into_iter()
                    .map(|(pos, text)| InlayHintData {
                        position: pos,
                        label_parts: vec![(text.clone(), None)],
                        insert_text: Some(text),
                        insert_imports: Vec::new(),
                    }),
            );
        }

        Some(res)
    }

    /// Helper to extract the element expression from an unpacked source.
    /// Returns the expression at the given position if the source is a tuple or list literal.
    /// For nested unpacking or function calls, returns None (caller should fall back to
    /// showing hints based on type information alone).
    fn get_unpacked_element_expr<'b>(
        bindings: &'b Bindings,
        unpack_idx: Idx<Key>,
        pos: UnpackedPosition,
    ) -> Option<&'b Expr> {
        // Get the binding for the unpacked source
        let source_binding = bindings.get(unpack_idx);
        // For top-level unpacking, the source is Binding::Expr containing the RHS.
        // For nested unpacking, it's Binding::UnpackedValue - we return None in that case.
        let source_expr = match source_binding {
            Binding::Expr(_, e) => Some(e),
            _ => None,
        }?;

        // Try to extract elements from tuple or list literals
        let elts = match &**source_expr {
            Expr::Tuple(tup) => Some(&tup.elts),
            Expr::List(lst) => Some(&lst.elts),
            _ => None,
        }?;

        // Extract the element at the given position
        // This mirrors the logic in solve.rs for Binding::UnpackedValue
        match pos {
            UnpackedPosition::Index(i) => elts.get(i),
            UnpackedPosition::ReverseIndex(i) => {
                elts.len().checked_sub(i).and_then(|idx| elts.get(idx))
            }
            // For slices (starred unpacking), we can't return a single element
            UnpackedPosition::Slice(_, _) => None,
        }
    }

    fn collect_function_calls_from_ast(module: Arc<ModModule>) -> Vec<ExprCall> {
        fn collect_function_calls(x: &Expr, calls: &mut Vec<ExprCall>) {
            if let Expr::Call(call) = x {
                calls.push(call.clone());
            }
            x.recurse(&mut |x| collect_function_calls(x, calls));
        }

        let mut function_calls = Vec::new();
        module.visit(&mut |x| collect_function_calls(x, &mut function_calls));
        function_calls
    }

    fn add_inlay_hints_for_positional_function_args(
        &self,
        handle: &Handle,
    ) -> Vec<(TextSize, String)> {
        let mut param_hints: Vec<(TextSize, String)> = Vec::new();

        if let Some(mod_module) = self.get_ast(handle) {
            let function_calls = Self::collect_function_calls_from_ast(mod_module);

            for call in function_calls {
                if let Some(answers) = self.get_answers(handle) {
                    let callee_type = if let Some((overloads, chosen_idx)) =
                        answers.get_all_overload_trace(call.arguments.range)
                    {
                        // If we have overload information, use the chosen overload
                        overloads
                            .get(chosen_idx.unwrap_or_default())
                            .map(|c| Type::Callable(Box::new(c.clone())))
                    } else {
                        // Otherwise, try to get the type of the callee directly
                        answers.get_type_trace(call.func.range())
                    };

                    if let Some(params) =
                        callee_type.and_then(normalize_singleton_function_type_into_params)
                    {
                        for (arg_idx, arg) in call.arguments.args.iter().enumerate() {
                            // Skip keyword arguments - they already show their parameter name
                            let is_keyword_arg = call
                                .arguments
                                .keywords
                                .iter()
                                .any(|kw| kw.value.range() == arg.range());

                            if !is_keyword_arg
                                && let Some(param_match) =
                                    Self::param_name_for_positional_argument(&params, arg_idx)
                                && !param_match.is_vararg_repeat
                                && param_match.name.as_str() != "self"
                                && param_match.name.as_str() != "cls"
                                && param_match.name.as_str() != "_cls"
                            {
                                param_hints.push((
                                    arg.range().start(),
                                    format!("{}= ", param_match.name.as_str()),
                                ));
                            }
                        }
                    }
                }
            }
        }

        param_hints.sort_by_key(|(pos, _)| *pos);
        param_hints
    }

    pub(crate) fn param_name_for_positional_argument<'param>(
        params: &'param [Param],
        positional_arg_index: usize,
    ) -> Option<ParamNameMatch<'param>> {
        let mut positional_params_seen = 0;
        for param in params {
            match param {
                Param::PosOnly(name, ..) => {
                    if positional_params_seen == positional_arg_index {
                        return name.as_ref().map(|name| {
                            ParamNameMatch::new(name, /* is_vararg_repeat */ false)
                        });
                    }
                    positional_params_seen += 1;
                }
                Param::Pos(name, ..) => {
                    if positional_params_seen == positional_arg_index {
                        return Some(ParamNameMatch::new(name, false));
                    }
                    positional_params_seen += 1;
                }
                Param::VarArg(name, ..) => {
                    if positional_arg_index >= positional_params_seen {
                        return name.as_ref().map(|name| {
                            ParamNameMatch::new(name, positional_arg_index > positional_params_seen)
                        });
                    }
                    break;
                }
                Param::KwOnly(..) | Param::Kwargs(..) => {}
            }
        }
        None
    }

    fn filter_parameters(
        &self,
        param_with_default: ParameterWithDefault,
        handle: &Handle,
    ) -> Option<ParameterAnnotation> {
        if param_with_default.name() == "self"
            || param_with_default.name() == "cls"
            || param_with_default.name() == "_cls"
        {
            return None;
        }
        let ty = match param_with_default.default() {
            Some(expr) => self.get_type_trace(handle, expr.range()),
            None => None,
        };
        Some(ParameterAnnotation {
            text_size: param_with_default.parameter.range().end(),
            ty,
            has_annotation: param_with_default.annotation().is_some(),
        })
    }

    fn collect_types_from_callees(&self, range: TextRange, handle: &Handle) -> Vec<Type> {
        fn callee_at(mod_module: Arc<ModModule>, position: TextSize) -> Option<ExprCall> {
            fn f(x: &Expr, find: TextSize, res: &mut Option<ExprCall>) {
                if let Expr::Call(call) = x
                    && call.func.range().contains_inclusive(find)
                {
                    f(call.func.as_ref(), find, res);
                    if res.is_some() {
                        return;
                    }
                    *res = Some(call.clone());
                } else {
                    x.recurse(&mut |x| f(x, find, res));
                }
            }
            let mut res = None;
            mod_module.visit(&mut |x| f(x, position, &mut res));
            res
        }
        match self.get_ast(handle) {
            Some(mod_module) => {
                let callee = callee_at(mod_module, range.start());
                match callee {
                    Some(ExprCall {
                        arguments: args, ..
                    }) => args
                        .args
                        .iter()
                        .filter_map(|arg| self.get_type_trace(handle, arg.range()))
                        .collect(),
                    None => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    fn collect_references(
        &self,
        handle: &Handle,
        idx: pyrefly_graph::index::Idx<Key>,
        bindings: Bindings,
        transaction: &mut CancellableTransaction,
    ) -> Vec<(pyrefly_python::module::Module, Vec<TextRange>)> {
        if let Key::Definition(id) = bindings.idx_to_key(idx)
            && let Some(module_info) = self.get_module_info(handle)
        {
            let definition_kind = DefinitionMetadata::VariableOrAttribute(None);
            if let Ok(references) = transaction.find_global_references_from_definition(
                *handle.sys_info(),
                definition_kind,
                TextRangeWithModule::new(module_info, id.range()),
                true,
            ) {
                return references;
            }
        }
        Vec::new()
    }

    pub fn infer_parameter_annotations(
        &self,
        handle: &Handle,
        cancellable_transaction: &mut CancellableTransaction,
    ) -> Vec<ParameterAnnotation> {
        if let Some(bindings) = self.get_bindings(handle) {
            let transaction = cancellable_transaction;
            fn transpose<T: Clone>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
                if v.is_empty() {
                    return Vec::new();
                }
                let max_len = v.iter().map(|row| row.len()).max().unwrap();
                let mut result = vec![Vec::new(); max_len];
                for row in v {
                    for (i, elem) in row.into_iter().enumerate() {
                        result[i].push(elem);
                    }
                }
                result
            }
            fn zip_types(
                inferred_types: Vec<Vec<Type>>,
                function_arguments: Vec<ParameterAnnotation>,
            ) -> Vec<ParameterAnnotation> {
                let zipped_inferred_types: Vec<Vec<Type>> = transpose(inferred_types);
                let types: Vec<(ParameterAnnotation, Vec<Type>)> =
                    match zipped_inferred_types.is_empty() {
                        true => function_arguments
                            .into_iter()
                            .map(
                                |arg: ParameterAnnotation| -> (ParameterAnnotation, Vec<Type>) {
                                    (arg, vec![])
                                },
                            )
                            .collect(),
                        false => function_arguments
                            .into_iter()
                            .zip(zipped_inferred_types)
                            .collect(),
                    };

                types
                    .into_iter()
                    .map(|(arg, mut ty)| {
                        let mut arg = arg;
                        if let Some(default_type) = arg.ty {
                            ty.push(default_type)
                        }
                        if ty.len() == 1 {
                            arg.ty = Some(ty[0].clone());
                        } else {
                            let ty = ty.into_iter().filter(|x| !x.is_any()).collect();
                            arg.ty = Some(Type::union(ty));
                        }
                        arg
                    })
                    .collect()
            }

            bindings
                .keys::<Key>()
                .flat_map(|idx| {
                    let binding = bindings.get(idx);
                    // Check if this binding is a function
                    if let Binding::Function(key_function, _, _) = binding {
                        let binding_func =
                            bindings.get(bindings.get(*key_function).undecorated_idx);
                        let args = binding_func.def.parameters.args.clone();
                        let func_args: Vec<ParameterAnnotation> = args
                            .into_iter()
                            .filter_map(|param_with_default| {
                                self.filter_parameters(param_with_default, handle)
                            })
                            .collect();
                        // Skip expensive reference collection and type inference
                        // for functions where every parameter is already annotated.
                        if func_args.iter().all(|arg| arg.has_annotation) {
                            return vec![];
                        }
                        let references =
                            self.collect_references(handle, idx, bindings.clone(), transaction);
                        let ranges: Vec<&TextRange> =
                            references.iter().flat_map(|(_, range)| range).collect();
                        let inferred_types = ranges
                            .into_iter()
                            .map(|range| self.collect_types_from_callees(*range, handle));
                        zip_types(inferred_types.collect(), func_args)
                    } else {
                        vec![]
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    pub fn inferred_types(
        &self,
        handle: &Handle,
        return_types: bool,
        containers: bool,
    ) -> Option<Vec<(TextSize, Type, AnnotationKind)>> {
        let is_interesting_type = |x: &Type| !x.is_any();
        let is_interesting_expr = |x: &Expr| !Ast::is_literal(x);
        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                // Return Annotation
                key @ Key::ReturnType(id) if return_types => {
                    match bindings.get(bindings.key_to_idx(&Key::Definition(*id))) {
                        Binding::Function(x, _pred, _class_meta) => {
                            if matches!(&bindings.get(idx), Binding::ReturnType(ret) if !ret.kind.has_return_annotation())
                                && let Some(ty) = self.get_type(handle, key)
                                && is_interesting_type(&ty)
                            {
                                let fun = bindings.get(bindings.get(*x).undecorated_idx);
                                res.push((
                                    fun.def.parameters.range.end(),
                                    ty,
                                    AnnotationKind::Return,
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                // Only annotate empty containers for now
                key @ Key::Definition(_) if containers => {
                    if let Some(ty) = self.get_type(handle, key) {
                        let e = match bindings.get(idx) {
                            Binding::NameAssign(x) if x.annotation.is_none() => match &*x.expr {
                                Expr::List(ExprList { elts, .. }) => {
                                    if elts.is_empty() {
                                        Some(&*x.expr)
                                    } else {
                                        None
                                    }
                                }
                                Expr::Dict(ExprDict { items, .. }) => {
                                    if items.is_empty() {
                                        Some(&*x.expr)
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some(e) = e
                            && is_interesting_expr(e)
                            && is_interesting_type(&ty)
                        {
                            res.push((key.range().end(), ty, AnnotationKind::Variable));
                        }
                    }
                }
                _ => {}
            }
        }
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_types::heap::TypeHeap;
    use ruff_python_ast::name::Name;

    use super::Transaction;
    use crate::types::callable::Param;
    use crate::types::callable::Required;
    use crate::types::types::Type;

    fn any_type() -> Type {
        TypeHeap::new().mk_any_explicit()
    }

    #[test]
    fn param_name_for_positional_argument_marks_vararg_repeats() {
        let params = vec![
            Param::Pos(Name::new_static("x"), any_type(), Required::Required),
            Param::VarArg(Some(Name::new_static("columns")), any_type()),
            Param::KwOnly(Name::new_static("kw"), any_type(), Required::Required),
        ];

        assert_eq!(match_summary(&params, 0), Some(("x", false)));
        assert_eq!(match_summary(&params, 1), Some(("columns", false)));
        assert_eq!(match_summary(&params, 3), Some(("columns", true)));
    }

    #[test]
    fn param_name_for_positional_argument_handles_missing_names() {
        let params = vec![
            Param::PosOnly(None, any_type(), Required::Required),
            Param::VarArg(None, any_type()),
        ];

        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 0).is_none());
        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 1).is_none());
        assert!(Transaction::<'static>::param_name_for_positional_argument(&params, 5).is_none());
    }

    #[test]
    fn duplicate_vararg_hints_are_not_emitted() {
        let params = vec![
            Param::Pos(Name::new_static("s"), any_type(), Required::Required),
            Param::VarArg(Some(Name::new_static("args")), any_type()),
            Param::KwOnly(Name::new_static("a"), any_type(), Required::Required),
        ];

        let labels: Vec<&str> = (0..4)
            .filter_map(|idx| {
                Transaction::<'static>::param_name_for_positional_argument(&params, idx)
            })
            .filter(|match_| !match_.is_vararg_repeat)
            .map(|match_| match_.name.as_str())
            .collect();

        assert_eq!(labels, vec!["s", "args"]);
    }

    fn match_summary(params: &[Param], idx: usize) -> Option<(&str, bool)> {
        Transaction::<'static>::param_name_for_positional_argument(params, idx)
            .map(|match_| (match_.name.as_str(), match_.is_vararg_repeat))
    }
}
