/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SymbolKind;
use pyrefly_python::module::TextRangeWithModule;

use crate::state::lsp::MIN_CHARACTERS_TYPED_AUTOIMPORT;
use crate::state::state::Transaction;

impl Transaction<'_> {
    pub fn workspace_symbols(
        &self,
        query: &str,
    ) -> Option<Vec<(String, SymbolKind, TextRangeWithModule)>> {
        if query.len() < MIN_CHARACTERS_TYPED_AUTOIMPORT {
            return None;
        }
        let mut matches = self.search_exports_fuzzy_scored(query);
        matches.sort_by(|left, right| {
            let left_is_init_reexport = left.is_reexport && left.handle.path().is_init();
            let right_is_init_reexport = right.is_reexport && right.handle.path().is_init();
            right
                .score
                .cmp(&left.score)
                .then_with(|| left_is_init_reexport.cmp(&right_is_init_reexport))
                .then_with(|| left.name.cmp(&right.name))
                .then_with(|| {
                    left.handle
                        .module()
                        .as_str()
                        .cmp(right.handle.module().as_str())
                })
        });
        let mut result = Vec::new();
        for match_result in matches {
            let handle = match_result.handle;
            let name = match_result.name;
            let export = match_result.export;
            if let Some(module) = self.get_module_info(&handle) {
                let kind = export
                    .symbol_kind
                    .map_or(SymbolKind::VARIABLE, |k| k.to_lsp_symbol_kind());
                let location = TextRangeWithModule {
                    module,
                    range: export.location,
                };
                result.push((name, kind, location));
            }
        }
        Some(result)
    }
}
