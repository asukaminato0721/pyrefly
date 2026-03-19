/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::FuncDefIndex;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::PropertyRole;
use pyrefly_types::types::Type;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use ruff_python_ast::name::Name;

use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::binding::KeyUndecoratedFunctionRange;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::get_all_classes;
use crate::report::pysa::class::get_class_fields;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionNode;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::get_exported_decorated_function;
use crate::report::pysa::function::should_export_decorated_function;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::slow_fun_monitor::slow_fun_monitor_scope;
use crate::report::pysa::step_logger::StepLogger;
use crate::report::pysa::types::is_callable_like;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;

// Intentionally refer to decorators by names instead of uniquely identifying them. Some special handling
// (i.e., see the usage of `GRAPHQL_DECORATORS`) are triggered when decorators are matched by names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphQLDecoratorRef {
    pub(crate) module: &'static str,
    pub(crate) name: &'static str,
}

impl GraphQLDecoratorRef {
    pub(crate) fn matches_definition(&self, definition: &FindDefinitionItemWithDocstring) -> bool {
        if let Some(display_name) = &definition.display_name {
            self.module == definition.module.name().as_str() && self.name == *display_name
        } else {
            false
        }
    }

    pub(crate) fn matches_decorator_id(&self, module: ModuleName, name: &Name) -> bool {
        module.as_str() == self.module && name.as_str() == self.name
    }
}

// Tuples of decorators. For any tuple (x, y), it means if a callable is decorated by x, then find
// all callables that are decorated by y inside the return class type of the callable.
pub(crate) static GRAPHQL_DECORATORS: &[(&GraphQLDecoratorRef, &GraphQLDecoratorRef)] = &[
    (
        &GraphQLDecoratorRef {
            module: "graphqlserver.types",
            name: "graphql_root_field",
        },
        &GraphQLDecoratorRef {
            module: "graphqlserver.types",
            name: "graphql_field",
        },
    ),
    // For testing only
    (
        &GraphQLDecoratorRef {
            module: "test",
            name: "decorator_1",
        },
        &GraphQLDecoratorRef {
            module: "test",
            name: "decorator_2",
        },
    ),
    // For testing only
    (
        &GraphQLDecoratorRef {
            module: "graphql_callees",
            name: "entrypoint_decorator",
        },
        &GraphQLDecoratorRef {
            module: "graphql_callees",
            name: "method_decorator",
        },
    ),
];

/// Checks whether a decorator type matches a `GraphQLDecoratorRef` by extracting
/// the module and name from the function metadata.
fn decorator_matches_graphql_ref(ty: &Type, graphql_ref: &GraphQLDecoratorRef) -> bool {
    let func_metadata = match ty {
        Type::Function(box pyrefly_types::callable::Function { metadata, .. }) => Some(metadata),
        Type::Overload(pyrefly_types::types::Overload { box metadata, .. }) => Some(metadata),
        _ => None,
    };
    match func_metadata {
        Some(pyrefly_types::callable::FuncMetadata {
            kind: FunctionKind::Def(box func_id),
            ..
        }) => graphql_ref.matches_decorator_id(func_id.module.name(), &func_id.name),
        _ => false,
    }
}

/// Per-module information required for the pysa report step.
///
/// Built while AST + bindings + answers are still available, persists after
/// eviction.
pub struct PysaModuleIndex {
    /// FuncDefIndex → FunctionRef for functions defined with `def`.
    func_def_to_function_ref: HashMap<FuncDefIndex, FunctionRef>,

    /// ShortIdentifier → FunctionRef for decorated functions (skip_property_getter=false).
    short_identifier_to_function_ref: HashMap<ShortIdentifier, FunctionRef>,

    /// ShortIdentifier → FunctionRef for property setters (skip_property_getter=true).
    /// Only populated when the result differs from `short_identifier_to_function_ref`.
    short_identifier_to_setter_ref: HashMap<ShortIdentifier, FunctionRef>,

    /// (ClassId, Name) → FunctionRef for class callable fields (own fields only).
    class_field_to_function_ref: HashMap<(ClassId, Name), FunctionRef>,

    /// Per ClassId, per field_name: matched graphql method decorators.
    /// Only populated for class fields whose decorators match a method decorator
    /// in `GRAPHQL_DECORATORS`.
    class_field_graphql_decorator_ids:
        HashMap<ClassId, HashMap<Name, Vec<&'static GraphQLDecoratorRef>>>,
}

impl PysaModuleIndex {
    /// Build the index for a single module from its full context.
    ///
    /// Must be called while AST + bindings + answers are available (before eviction).
    pub fn build(context: &ModuleContext) -> PysaModuleIndex {
        // Step 1: Build short_identifier_to_function_ref and short_identifier_to_setter_ref
        // by iterating all KeyDecoratedFunction entries.
        let mut short_identifier_to_function_ref = HashMap::new();
        let mut short_identifier_to_setter_ref = HashMap::new();

        for idx in context.bindings.keys::<KeyDecoratedFunction>() {
            let key = context.bindings.idx_to_key(idx);
            let short_identifier = key.0;

            let exported = get_exported_decorated_function(
                idx, /* skip_property_getter */ false, context,
            );
            if !should_export_decorated_function(&exported, context) {
                continue;
            }
            let function_ref = FunctionRef::from_decorated_function(&exported, context);
            short_identifier_to_function_ref.insert(short_identifier, function_ref.clone());

            // Also compute with skip_property_getter=true for property setters.
            let is_property_getter = exported
                .metadata()
                .flags
                .property_metadata
                .as_ref()
                .is_some_and(|m| m.role == PropertyRole::Getter);
            if is_property_getter {
                let exported_setter = get_exported_decorated_function(
                    idx, /* skip_property_getter */ true, context,
                );
                if should_export_decorated_function(&exported_setter, context) {
                    let setter_ref =
                        FunctionRef::from_decorated_function(&exported_setter, context);
                    if setter_ref != function_ref {
                        short_identifier_to_setter_ref.insert(short_identifier, setter_ref);
                    }
                }
            }
        }

        // Step 2: Build func_def_to_function_ref by iterating all
        // KeyUndecoratedFunctionRange entries and reusing short_identifier_to_function_ref.
        let mut func_def_to_function_ref = HashMap::new();

        for idx in context.bindings.keys::<KeyUndecoratedFunctionRange>() {
            let key = context.bindings.idx_to_key(idx);
            let func_def_index = key.0;
            if let Some(answer) = context.answers.get_idx(idx) {
                let short_identifier = answer.0;
                if let Some(function_ref) = short_identifier_to_function_ref.get(&short_identifier)
                {
                    func_def_to_function_ref.insert(func_def_index, function_ref.clone());
                }
            }
        }

        // Step 3: Build class field mappings.
        let mut class_field_to_function_ref = HashMap::new();
        let mut class_field_graphql_decorator_ids: HashMap<
            ClassId,
            HashMap<Name, Vec<&'static GraphQLDecoratorRef>>,
        > = HashMap::new();

        for class in get_all_classes(context) {
            let class_id = ClassId::from_class(&class);

            for (name, field) in get_class_fields(&class, context) {
                if !is_callable_like(&field.ty()) {
                    continue;
                }
                let field_name = name.into_owned();

                if let Some(function_node) = FunctionNode::exported_function_from_class_field(
                    &class,
                    &field_name,
                    field,
                    context,
                ) {
                    let function_ref = function_node.as_function_ref(context);
                    class_field_to_function_ref
                        .insert((class_id, field_name.clone()), function_ref);

                    // Find which graphql method decorators from GRAPHQL_DECORATORS
                    // match any of this field's decorators.
                    if let FunctionNode::DecoratedFunction(decorated) = &function_node {
                        let matching_refs: Vec<&'static GraphQLDecoratorRef> = GRAPHQL_DECORATORS
                            .iter()
                            .filter_map(|(_, method_decorator)| {
                                if decorated.undecorated.decorators.iter().any(|(ty, _)| {
                                    decorator_matches_graphql_ref(ty, method_decorator)
                                }) {
                                    Some(*method_decorator)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        if !matching_refs.is_empty() {
                            class_field_graphql_decorator_ids
                                .entry(class_id)
                                .or_default()
                                .insert(field_name, matching_refs);
                        }
                    }
                }
            }
        }

        PysaModuleIndex {
            func_def_to_function_ref,
            short_identifier_to_function_ref,
            short_identifier_to_setter_ref,
            class_field_to_function_ref,
            class_field_graphql_decorator_ids,
        }
    }
}

/// Whole-program index mapping ModuleId → PysaModuleIndex.
///
/// Thread-safe for concurrent reads after construction.
pub struct WholeProgramPysaModuleIndex(dashmap::ReadOnlyView<ModuleId, PysaModuleIndex>);

impl WholeProgramPysaModuleIndex {
    /// Returns the PysaModuleIndex for the given module. Panics if the module
    /// is not in the index, which would indicate a bug in index construction.
    fn get_module_index(&self, module_id: ModuleId) -> &PysaModuleIndex {
        self.0
            .get(&module_id)
            .expect("PysaModuleIndex missing for module")
    }

    /// Look up a FunctionRef by FuncDefIndex in a given module.
    /// Used to replace `call_targets_from_callable_metadata` cross-module lookup.
    ///
    /// Panics if the func_def_index is not in the module's index, which would
    /// indicate a bug in index construction (all functions should be indexed).
    pub fn get_function_ref_by_func_def_index(
        &self,
        module_id: ModuleId,
        func_def_index: FuncDefIndex,
    ) -> &FunctionRef {
        self.get_module_index(module_id)
            .func_def_to_function_ref
            .get(&func_def_index)
            .expect("FuncDefIndex missing from PysaModuleIndex")
    }

    /// Look up a FunctionRef by ShortIdentifier in a given module.
    /// Used to replace `exported_function_from_definition_item_with_docstring` cross-module lookup.
    ///
    /// When `skip_property_getter` is true, returns the property setter's FunctionRef
    /// if one exists; otherwise falls back to the normal (getter) FunctionRef.
    pub fn get_function_ref_by_short_identifier(
        &self,
        module_id: ModuleId,
        short_identifier: ShortIdentifier,
        skip_property_getter: bool,
    ) -> Option<&FunctionRef> {
        let index = self.get_module_index(module_id);
        if skip_property_getter {
            index
                .short_identifier_to_setter_ref
                .get(&short_identifier)
                .or_else(|| {
                    index
                        .short_identifier_to_function_ref
                        .get(&short_identifier)
                })
        } else {
            index
                .short_identifier_to_function_ref
                .get(&short_identifier)
        }
    }

    /// Look up a FunctionRef for a class's own callable field.
    /// Used to replace `get_context_from_class` + `exported_function_from_class_field`
    /// cross-module lookups.
    pub fn get_function_ref_for_class_field(
        &self,
        module_id: ModuleId,
        class_id: ClassId,
        field_name: &Name,
    ) -> Option<&FunctionRef> {
        self.get_module_index(module_id)
            .class_field_to_function_ref
            .get(&(class_id, field_name.clone()))
    }

    /// Return all FunctionRefs for class fields that have a matching graphql decorator.
    pub fn get_graphql_decorated_class_fields(
        &self,
        module_id: ModuleId,
        class_id: ClassId,
        predicate: impl Fn(&GraphQLDecoratorRef) -> bool,
    ) -> Vec<&FunctionRef> {
        let index = self.get_module_index(module_id);
        let Some(fields) = index.class_field_graphql_decorator_ids.get(&class_id) else {
            return Vec::new();
        };
        fields
            .iter()
            .filter(|(_, refs)| refs.iter().any(|r| predicate(r)))
            .filter_map(|(name, _)| {
                index
                    .class_field_to_function_ref
                    .get(&(class_id, name.clone()))
            })
            .collect()
    }
}

/// Build the whole-program PysaModuleIndex for all handles.
///
/// Must be called while AST + bindings + answers are available for all modules.
pub fn build_pysa_module_index(
    handles: &[Handle],
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> WholeProgramPysaModuleIndex {
    let step = StepLogger::start("Building pysa module index", "Built pysa module index");

    let index = dashmap::DashMap::new();

    ThreadPool::new().install(|| {
        slow_fun_monitor_scope(|slow_function_monitor| {
            handles.par_iter().for_each(|handle| {
                let module_id = module_ids.get_from_handle(handle);
                let context = ModuleContext::create(handle.clone(), transaction, module_ids);
                let module_index = slow_function_monitor.monitor_function(
                    || PysaModuleIndex::build(&context),
                    format!(
                        "Building pysa module index for {}",
                        handle.module().as_str(),
                    ),
                    /* max_time_in_seconds */ 4,
                );
                index.insert(module_id, module_index);
            });
        })
    });

    step.finish();
    WholeProgramPysaModuleIndex(index.into_read_only())
}
