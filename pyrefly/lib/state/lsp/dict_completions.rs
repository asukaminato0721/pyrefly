/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::types::Union;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Key;
use crate::binding::narrow::int_from_slice;
use crate::lsp::wasm::completion::RankedCompletion;
use crate::state::state::Transaction;
use crate::types::types::Type;

#[derive(Clone)]
enum DictKeyLiteralContext {
    /// A key literal used to access an existing dict/TypedDict.
    /// Examples: `cfg["na|"]`, `cfg.get("na|")`.
    KeyAccess {
        base_expr: Expr,
        literal: ExprStringLiteral,
    },
    /// A string literal in a call argument whose completions should come from
    /// surrounding container expressions.
    /// Examples: `lookup(data, "na|")`, `df.select(col("na|"))`.
    CallArgument {
        source_exprs: Vec<Expr>,
        literal: ExprStringLiteral,
    },
    /// A key literal inside a dict literal being constructed.
    /// Example: `{"na|": 1}`.
    DictLiteral {
        dict: ExprDict,
        literal: ExprStringLiteral,
    },
}

impl DictKeyLiteralContext {
    fn literal_range(&self) -> TextRange {
        match self {
            Self::KeyAccess { literal, .. }
            | Self::CallArgument { literal, .. }
            | Self::DictLiteral { literal, .. } => literal.range(),
        }
    }
}

impl<'a> Transaction<'a> {
    fn type_contains_typed_dict(ty: &Type) -> bool {
        match ty {
            Type::TypedDict(_) | Type::PartialTypedDict(_) => true,
            Type::Union(box Union { members, .. }) => {
                members.iter().any(Self::type_contains_typed_dict)
            }
            _ => false,
        }
    }

    fn expr_has_typed_dict_type(&self, handle: &Handle, expr: &Expr) -> bool {
        self.get_type_trace(handle, expr.range())
            .map(|ty| Self::type_contains_typed_dict(&ty))
            .unwrap_or(false)
    }

    /// Extracts typed dict access from `.get()` method calls.
    /// This handles both `d.get("key")` and `d["key"]` patterns - the subscript
    /// case is handled in `dict_key_string_literal_at`.
    fn typed_dict_get_string_literal(
        &self,
        handle: &Handle,
        call: &ExprCall,
    ) -> Option<(Expr, ExprStringLiteral)> {
        let Expr::Attribute(attr) = call.func.as_ref() else {
            return None;
        };
        if attr.attr.id.as_str() != "get" {
            return None;
        }
        if !self.expr_has_typed_dict_type(handle, attr.value.as_ref()) {
            return None;
        }
        // If there's already a string literal, we want to provide completions
        // for the key name inside the quotes (e.g., `d.get("k|")` -> suggest "key")
        if let Some(Expr::StringLiteral(lit)) = call.arguments.args.first() {
            return Some((attr.value.as_ref().clone(), lit.clone()));
        }
        if let Some(lit) =
            call.arguments
                .keywords
                .iter()
                .find_map(|kw| match (&kw.arg, &kw.value) {
                    (Some(id), Expr::StringLiteral(lit)) if id.id.as_str() == "key" => Some(lit),
                    _ => None,
                })
        {
            return Some((attr.value.as_ref().clone(), lit.clone()));
        }
        None
    }

    fn dict_key_string_literal_at(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
    ) -> Option<(Expr, ExprStringLiteral)> {
        let nodes = Ast::locate_node(module, position);
        let mut best: Option<(u8, TextSize, Expr, ExprStringLiteral)> = None;
        for node in nodes {
            let candidate = match node {
                AnyNodeRef::ExprSubscript(sub) => {
                    if let Expr::StringLiteral(lit) = sub.slice.as_ref() {
                        Some((sub.value.as_ref().clone(), lit.clone()))
                    } else {
                        None
                    }
                }
                AnyNodeRef::ExprCall(call) => self.typed_dict_get_string_literal(handle, call),
                _ => None,
            };
            let Some((base_expr, literal)) = candidate else {
                continue;
            };
            let (priority, dist) = Self::string_literal_priority(position, literal.range());
            let should_update = match &best {
                Some((best_prio, best_dist, _, _)) => {
                    priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                }
                None => true,
            };
            if should_update {
                best = Some((priority, dist, base_expr, literal));
                if priority == 0 && dist == TextSize::from(0) {
                    break;
                }
            }
        }
        best.map(|(_, _, base_expr, literal)| (base_expr, literal))
    }

    fn string_literal_priority(position: TextSize, range: TextRange) -> (u8, TextSize) {
        if range.contains(position) {
            (0, TextSize::from(0))
        } else if position < range.start() {
            (1, range.start() - position)
        } else {
            (2, position - range.end())
        }
    }

    fn dict_key_literal_context(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
    ) -> Option<DictKeyLiteralContext> {
        // Prefer direct key access (`d["k"]` / `d.get("k")`) so we can reuse the base
        // expression for facet-based completions. Fall back to dict literal keys.
        if let Some((base_expr, literal)) =
            self.dict_key_string_literal_at(handle, module, position)
        {
            Some(DictKeyLiteralContext::KeyAccess { base_expr, literal })
        } else if let Some((source_exprs, literal)) =
            self.call_argument_string_literal_at(module, position)
        {
            Some(DictKeyLiteralContext::CallArgument {
                source_exprs,
                literal,
            })
        } else {
            Self::dict_literal_string_literal_at(module, position)
                .map(|(dict, literal)| DictKeyLiteralContext::DictLiteral { dict, literal })
        }
    }

    fn call_argument_string_literal_at(
        &self,
        module: &ModModule,
        position: TextSize,
    ) -> Option<(Vec<Expr>, ExprStringLiteral)> {
        let nodes = Ast::locate_node(module, position);
        let literal = nodes.iter().find_map(|node| match node {
            AnyNodeRef::ExprStringLiteral(literal) => Some((*literal).clone()),
            _ => None,
        })?;
        let literal_range = literal.range();
        let mut source_exprs = Vec::new();

        for node in nodes {
            let AnyNodeRef::ExprCall(call) = node else {
                continue;
            };
            let Some((arg_index, is_keyword)) =
                Self::call_argument_containing_range(call, literal_range)
            else {
                continue;
            };
            if let Expr::Attribute(attr) = call.func.as_ref() {
                Self::push_unique_expr(&mut source_exprs, attr.value.as_ref());
            }
            let positional_count = if is_keyword {
                call.arguments.args.len()
            } else {
                arg_index
            };
            for expr in call.arguments.args.iter().take(positional_count) {
                Self::push_unique_expr(&mut source_exprs, expr);
            }
        }

        (!source_exprs.is_empty()).then_some((source_exprs, literal.clone()))
    }

    fn call_argument_containing_range(call: &ExprCall, target: TextRange) -> Option<(usize, bool)> {
        for (idx, arg) in call.arguments.args.iter().enumerate() {
            if Self::range_contains_range(arg.range(), target) {
                return Some((idx, false));
            }
        }
        for kw in &call.arguments.keywords {
            if Self::range_contains_range(kw.value.range(), target) {
                return Some((call.arguments.args.len(), true));
            }
        }
        None
    }

    fn range_contains_range(outer: TextRange, inner: TextRange) -> bool {
        outer.start() <= inner.start() && inner.end() <= outer.end()
    }

    fn push_unique_expr(target: &mut Vec<Expr>, expr: &Expr) {
        let range = expr.range();
        if target.iter().any(|existing| existing.range() == range) {
            return;
        }
        target.push(expr.clone());
    }

    fn dict_literal_string_literal_at(
        module: &ModModule,
        position: TextSize,
    ) -> Option<(ExprDict, ExprStringLiteral)> {
        let nodes = Ast::locate_node(module, position);
        let mut best: Option<(u8, TextSize, ExprDict, ExprStringLiteral)> = None;
        for node in nodes {
            let AnyNodeRef::ExprDict(dict) = node else {
                continue;
            };
            let mut best_in_dict: Option<(u8, TextSize, ExprStringLiteral)> = None;
            for item in &dict.items {
                let Some(key_expr) = item.key.as_ref() else {
                    continue;
                };
                let Expr::StringLiteral(literal) = key_expr else {
                    continue;
                };
                let (priority, dist) = Self::string_literal_priority(position, literal.range());
                let should_update = match &best_in_dict {
                    Some((best_prio, best_dist, _)) => {
                        priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                    }
                    None => true,
                };
                if should_update {
                    best_in_dict = Some((priority, dist, literal.clone()));
                    if priority == 0 && dist == TextSize::from(0) {
                        break;
                    }
                }
            }
            let Some((priority, dist, literal)) = best_in_dict else {
                continue;
            };
            let should_update = match &best {
                Some((best_prio, best_dist, _, _)) => {
                    priority < *best_prio || (priority == *best_prio && dist < *best_dist)
                }
                None => true,
            };
            if should_update {
                best = Some((priority, dist, dict.clone(), literal));
                if priority == 0 && dist == TextSize::from(0) {
                    break;
                }
            }
        }
        best.map(|(_, _, dict, literal)| (dict, literal))
    }

    fn expression_facets(expr: &Expr) -> Option<(Identifier, Vec<FacetKind>)> {
        let mut facets = Vec::new();
        let mut current = expr;
        loop {
            match current {
                Expr::Subscript(sub) => {
                    if let Some(idx) = int_from_slice(sub.slice.as_ref()) {
                        facets.push(FacetKind::Index(idx));
                    } else if let Expr::StringLiteral(lit) = sub.slice.as_ref() {
                        facets.push(FacetKind::Key(lit.value.to_string()));
                    } else {
                        return None;
                    }
                    current = sub.value.as_ref();
                }
                Expr::Attribute(attr) => {
                    facets.push(FacetKind::Attribute(attr.attr.id.clone()));
                    current = attr.value.as_ref();
                }
                Expr::Name(name) => {
                    facets.reverse();
                    return Some((Ast::expr_name_identifier(name.clone()), facets));
                }
                _ => return None,
            }
        }
    }

    fn collect_typed_dict_keys(
        &self,
        handle: &Handle,
        base_type: Type,
    ) -> Option<BTreeMap<String, Type>> {
        self.ad_hoc_solve(handle, "typed_dict_keys", |solver| {
            let mut map = BTreeMap::new();
            let mut stack = vec![base_type];
            while let Some(ty) = stack.pop() {
                match ty {
                    Type::TypedDict(td) | Type::PartialTypedDict(td) => {
                        for (name, field) in solver.type_order().typed_dict_fields(&td) {
                            map.entry(name.to_string())
                                .or_insert_with(|| field.ty.clone());
                        }
                    }
                    Type::Union(box Union { members, .. }) => {
                        stack.extend(members.into_iter());
                    }
                    _ => {}
                }
            }
            map
        })
    }

    fn extend_key_suggestions_for_expr(
        &self,
        handle: &Handle,
        expr: &Expr,
        suggestions: &mut BTreeMap<String, Option<Type>>,
    ) {
        if let Some(bindings) = self.get_bindings(handle) {
            let base_info = if let Some((identifier, facets)) = Self::expression_facets(expr) {
                Some((identifier, facets))
            } else if let Expr::Name(name) = expr {
                Some((Ast::expr_name_identifier(name.clone()), Vec::new()))
            } else {
                None
            };

            if let Some((identifier, facets)) = base_info {
                let short_id = ShortIdentifier::new(&identifier);
                let idx_opt = {
                    let bound_key = Key::BoundName(short_id);
                    if bindings.is_valid_key(&bound_key) {
                        Some(bindings.key_to_idx(&bound_key))
                    } else {
                        let def_key = Key::Definition(short_id);
                        if bindings.is_valid_key(&def_key) {
                            Some(bindings.key_to_idx(&def_key))
                        } else {
                            None
                        }
                    }
                };

                if let Some(idx) = idx_opt {
                    let facets_clone = facets.clone();
                    if let Some(keys) = self.ad_hoc_solve(handle, "dict_key_facets", |solver| {
                        let info = solver.get_idx(idx);
                        info.key_facets_at(&facets_clone)
                    }) {
                        for (key, ty_opt) in keys {
                            suggestions.entry(key).or_insert(ty_opt);
                        }
                    }
                }
            }
        }

        if let Some(base_type) = self.get_type_trace(handle, expr.range())
            && let Some(typed_keys) = self.collect_typed_dict_keys(handle, base_type)
        {
            for (key, ty) in typed_keys {
                let entry = suggestions.entry(key).or_insert(None);
                if entry.is_none() {
                    *entry = Some(ty);
                }
            }
        }
    }

    /// Adds dict key completions for the given position. Returns `true` if this function
    /// claimed the position (i.e., we are inside a dict/TypedDict key string literal), in
    /// which case the caller should skip overload-based literal completions to avoid showing
    /// redundant entries.
    pub(crate) fn add_dict_key_completions(
        &self,
        handle: &Handle,
        module: &ModModule,
        position: TextSize,
        completions: &mut Vec<RankedCompletion>,
    ) -> bool {
        let Some(context) = self.dict_key_literal_context(handle, module, position) else {
            return false;
        };
        let literal_range = context.literal_range();
        // Allow the cursor to sit a few characters before the literal (e.g. between nested
        // subscripts) so completion requests fired just before the quotes still succeed.
        let allowance = TextSize::from(4);
        let lower_bound = literal_range
            .start()
            .checked_sub(allowance)
            .unwrap_or_else(|| TextSize::new(0));
        if position < lower_bound || position > literal_range.end() {
            return false;
        }
        let mut suggestions: BTreeMap<String, Option<Type>> = BTreeMap::new();

        match &context {
            DictKeyLiteralContext::KeyAccess { base_expr, .. } => {
                self.extend_key_suggestions_for_expr(handle, base_expr, &mut suggestions);
            }
            DictKeyLiteralContext::CallArgument { source_exprs, .. } => {
                for expr in source_exprs {
                    self.extend_key_suggestions_for_expr(handle, expr, &mut suggestions);
                }
            }
            DictKeyLiteralContext::DictLiteral { dict, .. } => {
                // Dict literals need contextual typing from the whole expression instead of a
                // source expression so we can pick up `cfg: Config = {"na|": 1}`.
                if let Some(base_type) = self.get_type_trace(handle, dict.range())
                    && let Some(typed_keys) = self.collect_typed_dict_keys(handle, base_type)
                {
                    for (key, ty) in typed_keys {
                        let entry = suggestions.entry(key).or_insert(None);
                        if entry.is_none() {
                            *entry = Some(ty);
                        }
                    }
                }
            }
        }

        if suggestions.is_empty() {
            return false;
        }

        for (label, ty_opt) in suggestions {
            let detail = ty_opt.as_ref().map(|ty| ty.to_string());
            completions.push(RankedCompletion::new(CompletionItem {
                label,
                detail,
                kind: Some(CompletionItemKind::FIELD),
                ..Default::default()
            }));
        }
        true
    }
}
