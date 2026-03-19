/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use serde::Serialize;

use crate::report::pysa::step_logger::StepLogger;

/// Represents a unique identifier for a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ModuleId(u32);

impl ModuleId {
    pub fn to_int(self) -> u32 {
        self.0
    }
}

/// Represents what makes a module unique
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleKey {
    name: ModuleName,
    path: ModulePath,
}

impl ModuleKey {
    pub fn from_handle(handle: &Handle) -> ModuleKey {
        ModuleKey {
            name: handle.module(),
            path: handle.path().dupe(),
        }
    }

    pub fn from_module(module: &Module) -> ModuleKey {
        ModuleKey {
            name: module.name(),
            path: module.path().dupe(),
        }
    }
}

pub struct ModuleIds(HashMap<ModuleKey, ModuleId>);

impl ModuleIds {
    /// Multiple python files can map to the same module name (e.g, `foo.bar`).
    /// This creates a unique and deterministic identifier for each handle.
    pub fn new(handles: &[Handle]) -> ModuleIds {
        let step = StepLogger::start("Building unique module ids", "Built unique module ids");

        let mut modules = handles
            .iter()
            .map(ModuleKey::from_handle)
            .collect::<Vec<_>>();
        modules.sort();

        let mut result = HashMap::new();
        let mut current_id = 1;
        for module in modules {
            assert!(
                result.insert(module, ModuleId(current_id)).is_none(),
                "Found multiple handles with the same module name and path"
            );
            current_id += 1;
        }

        step.finish();
        ModuleIds(result)
    }

    pub fn get_opt(&self, key: &ModuleKey) -> Option<ModuleId> {
        self.0.get(key).copied()
    }

    pub fn get_from_handle(&self, handle: &Handle) -> ModuleId {
        let key = ModuleKey::from_handle(handle);
        self.get_opt(&key).unwrap_or_else(|| {
            panic!(
                "ModuleIds missing entry for handle module={}, path={:?} — was the module indexed?",
                handle.module(),
                handle.path().details(),
            )
        })
    }

    pub fn get_from_module(&self, module: &Module) -> ModuleId {
        let key = ModuleKey::from_module(module);
        self.get_opt(&key).unwrap_or_else(|| {
            panic!(
                "ModuleIds missing entry for module={}, path={:?} — was the module indexed?",
                module.name(),
                module.path().details(),
            )
        })
    }

    /// Returns the ModuleId for the given handle, or None if the handle is not indexed.
    /// Use this for handles from LSP functions that may lazy-load modules not present
    /// when ModuleIds was built.
    pub fn get_from_handle_opt(&self, handle: &Handle) -> Option<ModuleId> {
        self.get_opt(&ModuleKey::from_handle(handle))
    }

    /// Returns the ModuleId for the given module, or None if the module is not indexed.
    /// Use this for modules from LSP functions that may lazy-load modules not present
    /// when ModuleIds was built.
    pub fn get_from_module_opt(&self, module: &Module) -> Option<ModuleId> {
        self.get_opt(&ModuleKey::from_module(module))
    }
}
