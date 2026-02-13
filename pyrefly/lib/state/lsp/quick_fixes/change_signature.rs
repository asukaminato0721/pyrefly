/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;
use crate::state::lsp::quick_fixes::extract_shared::NameRefCollector;
use crate::state::lsp::quick_fixes::extract_shared::decorator_matches_name;
use crate::state::lsp::quick_fixes::extract_shared::expand_range_to_remove_item;
use crate::state::lsp::quick_fixes::extract_shared::find_enclosing_function;
use crate::state::lsp::quick_fixes::extract_shared::function_has_decorator;

pub(crate) fn change_signature_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let (function_def, param_index) = find_parameter_context(
        transaction,
        handle,
        ast.as_ref(),
        module_info.path(),
        selection,
    )?;
    if !decorators_are_method_safe(function_def) {
        return None;
    }
    if !function_def.parameters.posonlyargs.is_empty()
        || !function_def.parameters.kwonlyargs.is_empty()
        || function_def.parameters.vararg.is_some()
        || function_def.parameters.kwarg.is_some()
    {
        return None;
    }
    let param_name = function_def.parameters.args[param_index]
        .name()
        .id
        .to_string();
    let method_ctx = method_context_from_ast(ast.as_ref(), function_def);
    let has_implicit_self = method_ctx.is_some() && !method_ctx.unwrap().is_staticmethod;
    if has_implicit_self && param_index == 0 {
        return None;
    }
    let mut collector = NameRefCollector::new(param_name.clone());
    collector.visit_stmts(&function_def.body);
    if collector.invalid || !collector.load_refs.is_empty() {
        return None;
    }
    let func_def = transaction
        .find_definition(
            handle,
            function_def.name.range().start(),
            FindPreference::default(),
        )
        .into_iter()
        .find(|def| {
            def.module.path() == module_info.path()
                && def.definition_range == function_def.name.range()
        })?;
    let signature_remove_range = expand_range_to_remove_item(
        module_info.contents(),
        function_def.parameters.args[param_index].range(),
    );
    let mut edits = vec![(module_info.dupe(), signature_remove_range, String::new())];
    let call_edits = collect_callsite_removals(
        transaction,
        handle,
        &func_def,
        &param_name,
        param_index,
        has_implicit_self,
    )?;
    edits.extend(call_edits);
    Some(vec![LocalRefactorCodeAction {
        title: format!("Change signature: remove parameter `{param_name}`"),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

#[derive(Clone, Copy)]
struct MethodContext {
    is_staticmethod: bool,
}

fn method_context_from_ast(
    ast: &ModModule,
    function_def: &StmtFunctionDef,
) -> Option<MethodContext> {
    let covering_nodes = Ast::locate_node(ast, function_def.range().start());
    for (idx, node) in covering_nodes.iter().enumerate() {
        if let AnyNodeRef::StmtFunctionDef(def) = node
            && def.range() == function_def.range()
        {
            if let Some(AnyNodeRef::StmtClassDef(_)) = covering_nodes.get(idx + 1) {
                return Some(MethodContext {
                    is_staticmethod: function_has_decorator(function_def, "staticmethod"),
                });
            }
        }
    }
    None
}

fn decorators_are_method_safe(function_def: &StmtFunctionDef) -> bool {
    function_def.decorator_list.iter().all(|decorator| {
        let expr = &decorator.expression;
        decorator_matches_name(expr, "staticmethod") || decorator_matches_name(expr, "classmethod")
    })
}

fn find_parameter_context<'a>(
    transaction: &Transaction<'_>,
    handle: &Handle,
    ast: &'a ModModule,
    module_path: &ModulePath,
    selection: TextRange,
) -> Option<(&'a StmtFunctionDef, usize)> {
    let position = selection.start();
    if let Some((function_def, param_index)) =
        find_parameter_from_definition(transaction, handle, ast, module_path, position)
    {
        return Some((function_def, param_index));
    }
    let function_def = find_enclosing_function(ast, selection)?;
    if !function_def.parameters.range().contains_range(selection) {
        return None;
    }
    let param_index = select_parameter_index(&function_def.parameters.args, position)?;
    Some((function_def, param_index))
}

fn find_parameter_from_definition<'a>(
    transaction: &Transaction<'_>,
    handle: &Handle,
    ast: &'a ModModule,
    module_path: &ModulePath,
    position: TextSize,
) -> Option<(&'a StmtFunctionDef, usize)> {
    let param_def = transaction
        .find_definition(handle, position, FindPreference::default())
        .into_iter()
        .find(|def| {
            def.module.path() == module_path
                && matches!(def.metadata.symbol_kind(), Some(SymbolKind::Parameter))
        })?;
    let function_def = find_enclosing_function(ast, param_def.definition_range)?;
    let param_index = function_def
        .parameters
        .args
        .iter()
        .position(|param| param.name().range() == param_def.definition_range)?;
    Some((function_def, param_index))
}

fn select_parameter_index(
    params: &[ruff_python_ast::ParameterWithDefault],
    position: TextSize,
) -> Option<usize> {
    if params.is_empty() {
        return None;
    }
    for (idx, param) in params.iter().enumerate() {
        if param.range().contains(position) {
            return Some(idx);
        }
    }
    for (idx, param) in params.iter().enumerate() {
        if param.range().start() >= position {
            return Some(idx);
        }
    }
    let last_idx = params.len().saturating_sub(1);
    Some(last_idx)
}

fn collect_callsite_removals(
    transaction: &Transaction<'_>,
    handle: &Handle,
    func_def: &crate::state::lsp::FindDefinitionItemWithDocstring,
    param_name: &str,
    param_index: usize,
    has_implicit_self: bool,
) -> Option<Vec<(Module, TextRange, String)>> {
    let definition = TextRangeWithModule::new(func_def.module.dupe(), func_def.definition_range);
    let candidate_handles = candidate_handles_for_definition(transaction, handle, &definition);
    let mut edits = Vec::new();
    for candidate in candidate_handles {
        let Some(module_info) = transaction.get_module_info(&candidate) else {
            continue;
        };
        let Some(ast) = transaction.get_ast(&candidate) else {
            continue;
        };
        let patched_definition = patch_definition_for_handle(transaction, &candidate, &definition);
        let mut call_edits = Vec::new();
        let mut aborted = false;
        ast.visit(&mut |expr| {
            if aborted {
                return;
            }
            let Expr::Call(call) = expr else {
                return;
            };
            let Some(call_ref_range) = call_reference_range(call) else {
                return;
            };
            if !call_matches_definition(
                transaction,
                &candidate,
                call_ref_range,
                &patched_definition,
            ) {
                return;
            }
            if call.arguments.args.iter().any(|arg| arg.is_starred_expr())
                || call
                    .arguments
                    .keywords
                    .iter()
                    .any(|keyword| keyword.arg.is_none())
            {
                aborted = true;
                return;
            }
            if let Some(remove_range) = argument_remove_range(
                module_info.contents(),
                call,
                param_name,
                param_index,
                has_implicit_self,
            ) {
                call_edits.push((remove_range, String::new()));
            }
        });
        if aborted {
            return None;
        }
        for (range, replacement) in call_edits {
            edits.push((module_info.dupe(), range, replacement));
        }
    }
    Some(edits)
}

fn candidate_handles_for_definition(
    transaction: &Transaction<'_>,
    handle: &Handle,
    definition: &TextRangeWithModule,
) -> Vec<Handle> {
    let sys_info = handle.sys_info().dupe();
    match definition.module.path().details() {
        ModulePathDetails::Memory(path_buf) => {
            let fs_handle = Handle::new(
                definition.module.name(),
                ModulePath::filesystem((**path_buf).clone()),
                sys_info.dupe(),
            );
            if transaction.get_module_info(&fs_handle).is_none() {
                return transaction.handles();
            }
            let mut rdeps = transaction.get_transitive_rdeps(fs_handle);
            rdeps.insert(Handle::new(
                definition.module.name(),
                definition.module.path().dupe(),
                sys_info,
            ));
            rdeps.into_iter().collect()
        }
        _ => {
            let def_handle = Handle::new(
                definition.module.name(),
                definition.module.path().dupe(),
                sys_info,
            );
            transaction
                .get_transitive_rdeps(def_handle)
                .into_iter()
                .collect()
        }
    }
}

fn patch_definition_for_handle(
    transaction: &Transaction<'_>,
    handle: &Handle,
    definition: &TextRangeWithModule,
) -> TextRangeWithModule {
    match definition.module.path().details() {
        ModulePathDetails::Memory(path_buf) if handle.path() != definition.module.path() => {
            let module = if let Some(info) = transaction.get_module_info(&Handle::new(
                definition.module.name(),
                ModulePath::filesystem((**path_buf).clone()),
                handle.sys_info().dupe(),
            )) {
                info
            } else {
                definition.module.dupe()
            };
            TextRangeWithModule {
                module,
                range: definition.range,
            }
        }
        _ => definition.clone(),
    }
}

fn call_reference_range(call: &ExprCall) -> Option<TextRange> {
    match call.func.as_ref() {
        Expr::Name(name) => Some(name.range()),
        Expr::Attribute(attribute) => Some(attribute.attr.range()),
        _ => None,
    }
}

fn call_matches_definition(
    transaction: &Transaction<'_>,
    handle: &Handle,
    call_ref_range: TextRange,
    definition: &TextRangeWithModule,
) -> bool {
    let defs =
        transaction.find_definition(handle, call_ref_range.start(), FindPreference::default());
    defs.into_iter().any(|def| {
        def.module.path() == definition.module.path() && def.definition_range == definition.range
    })
}

fn argument_remove_range(
    source: &str,
    call: &ExprCall,
    param_name: &str,
    param_index: usize,
    has_implicit_self: bool,
) -> Option<TextRange> {
    if let Some(keyword) = call.arguments.find_keyword(param_name) {
        return Some(expand_range_to_remove_item(source, keyword.range()));
    }
    let shift = if has_implicit_self { 1 } else { 0 };
    let arg_index = param_index.checked_sub(shift)?;
    let arg = call.arguments.find_positional(arg_index)?;
    Some(expand_range_to_remove_item(source, arg.range()))
}
