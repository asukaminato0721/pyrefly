/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;

use dupe::Dupe;
use lsp_types::Range;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::telemetry::SubTaskTelemetry;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextSize;

use crate::report::glean::convert::ScopeType;
use crate::report::glean::convert::compute_scope;
use crate::report::glean::convert::join_names;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;

pub trait ExternalReferences: Send + Sync {
    fn find_references(
        &self,
        qualified_name: &str,
        source_uri: &Url,
        timeout: Duration,
        telemetry: Option<SubTaskTelemetry>,
    ) -> Vec<(Url, Vec<Range>)>;
}

pub struct NoExternalReferences;

impl ExternalReferences for NoExternalReferences {
    fn find_references(
        &self,
        _qualified_name: &str,
        _source_uri: &Url,
        _timeout: Duration,
        _telemetry: Option<SubTaskTelemetry>,
    ) -> Vec<(Url, Vec<Range>)> {
        Vec::new()
    }
}

/// Compute the fully-qualified name for the symbol defined at `position`
/// in the AST, using the same `.<locals>` convention as the Glean indexer.
///
/// Walks enclosing scopes via `Ast::locate_node`, inserting `.<locals>`
/// for function scopes (e.g. `module.outer.<locals>.inner`).
fn qualified_name_at_position(
    ast: &ModModule,
    module_name: ModuleName,
    position: TextSize,
    name: &str,
) -> String {
    let covering_nodes = Ast::locate_node(ast, position);

    // Collect enclosing scopes, skipping the node whose name is being
    // defined (i.e. the node whose name range contains the position).
    let mut scopes: Vec<(&str, bool)> = Vec::new();
    for node in &covering_nodes {
        match node {
            AnyNodeRef::StmtFunctionDef(func) if !func.name.range().contains(position) => {
                scopes.push((func.name.as_str(), true));
            }
            AnyNodeRef::StmtClassDef(cls) if !cls.name.range().contains(position) => {
                scopes.push((cls.name.as_str(), false));
            }
            _ => {}
        }
    }

    // Reverse to outermost-first, then build the scope chain using
    // compute_scope at every level — keeping .<locals> logic centralized.
    scopes.reverse();
    let module_str = module_name.to_string();
    let mut container = module_str.clone();
    let mut is_function = false;
    // Build up the container chain from enclosing scopes.
    // e.g. for `def outer(): def inner(): pass`, after this loop
    // container = "mymod.outer" and is_function = true.
    for (scope_name, scope_is_function) in &scopes {
        let scope = compute_scope(&container, is_function, &ScopeType::Local, &module_str);
        container = join_names(&scope, scope_name);
        is_function = *scope_is_function;
    }

    // Now attach the target symbol to the container we built.
    // e.g. compute_scope inserts .<locals> because the container is a function,
    // giving us "mymod.outer.<locals>.inner".
    let scope = compute_scope(&container, is_function, &ScopeType::Local, &module_str);
    join_names(&scope, name)
}

/// Derive a fully-qualified name string from a resolved definition, suitable
/// for use with external reference providers (see [`ExternalReferences`]).
/// Returns `None` if the name cannot be determined.
#[allow(dead_code)]
pub(crate) fn compute_qualified_name(
    transaction: &Transaction,
    handle: &Handle,
    definition: &FindDefinitionItemWithDocstring,
) -> Option<String> {
    let module_name = definition.module.name();

    if let DefinitionMetadata::Module = &definition.metadata {
        return Some(module_name.to_string());
    }

    let display_name = definition.display_name.as_deref()?;
    let definition_handle = Handle::new(
        module_name,
        definition.module.path().dupe(),
        handle.sys_info().dupe(),
    );
    let ast = transaction.get_ast(&definition_handle)?;
    Some(qualified_name_at_position(
        &ast,
        module_name,
        definition.definition_range.start(),
        display_name,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_qname(code: &str, module: &str, position: TextSize, name: &str) -> String {
        use ruff_python_ast::PySourceType;
        let (ast, _, _) = Ast::parse(code, PySourceType::Python);
        let module_name = ModuleName::from_str(module);
        qualified_name_at_position(&ast, module_name, position, name)
    }

    #[test]
    fn test_qname_top_level_function() {
        let result = parse_and_qname("def foo(): pass", "mymod", TextSize::new(4), "foo");
        assert_eq!(result, "mymod.foo");
    }

    #[test]
    fn test_qname_method_in_class() {
        let code = "class Bar:\n    def baz(self): pass";
        let result = parse_and_qname(code, "mymod", TextSize::new(19), "baz");
        assert_eq!(result, "mymod.Bar.baz");
    }

    #[test]
    fn test_qname_nested_class() {
        let code = "class Outer:\n    class Inner: pass";
        let result = parse_and_qname(code, "mymod", TextSize::new(24), "Inner");
        assert_eq!(result, "mymod.Outer.Inner");
    }

    #[test]
    fn test_qname_nested_function() {
        let code = "def foo():\n    def bar(): pass";
        let result = parse_and_qname(code, "mymod", TextSize::new(19), "bar");
        assert_eq!(result, "mymod.foo.<locals>.bar");
    }
}
