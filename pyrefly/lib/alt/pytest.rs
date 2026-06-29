/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::KeyDecorator;
use crate::binding::pytest::is_pytest_parametrize_decorator;
use crate::error::collector::ErrorCollector;
use crate::types::types::Type;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Infer parameter types from values in pytest parametrization decorators.
    pub(super) fn pytest_parametrize_parameter_types(
        &self,
        decorators: &[Idx<KeyDecorator>],
        errors: &ErrorCollector,
    ) -> Option<SmallMap<Name, Type>> {
        let aliases = self.bindings().pytest_info()?.aliases();
        let mut by_name: SmallMap<Name, Type> = SmallMap::new();
        for key in decorators {
            let binding = self.bindings().get::<KeyDecorator>(*key);
            if !is_pytest_parametrize_decorator(&binding.expr, aliases) {
                continue;
            }
            let Expr::Call(call) = &binding.expr else {
                unreachable!("pytest parametrize decorator is a call")
            };
            let [argnames, argvalues, ..] = &*call.arguments.args else {
                continue;
            };
            let Some(names) = Self::pytest_parametrize_names(argnames) else {
                continue;
            };
            let Some(types) = self.pytest_parametrize_value_types(argvalues, names.len(), errors)
            else {
                continue;
            };
            for (name, ty) in names.into_iter().zip(types) {
                match by_name.entry(name) {
                    Entry::Occupied(mut entry) => {
                        let old = entry.get().clone();
                        *entry.get_mut() = self.union(old, ty);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(ty);
                    }
                }
            }
        }
        Some(by_name)
    }

    fn pytest_parametrize_names(expr: &Expr) -> Option<Vec<Name>> {
        match expr {
            Expr::StringLiteral(lit) => {
                let names = lit
                    .value
                    .to_str()
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(Name::new)
                    .collect::<Vec<_>>();
                (!names.is_empty()).then_some(names)
            }
            Expr::Tuple(tuple) => Self::pytest_parametrize_names_from_elts(&tuple.elts),
            Expr::List(list) => Self::pytest_parametrize_names_from_elts(&list.elts),
            _ => None,
        }
    }

    fn pytest_parametrize_names_from_elts(elts: &[Expr]) -> Option<Vec<Name>> {
        let mut names = Vec::with_capacity(elts.len());
        for elt in elts {
            let Expr::StringLiteral(lit) = elt else {
                return None;
            };
            let name = lit.value.to_str();
            if name.is_empty() {
                return None;
            }
            names.push(Name::new(name));
        }
        Some(names)
    }

    fn pytest_parametrize_value_types(
        &self,
        expr: &Expr,
        arity: usize,
        errors: &ErrorCollector,
    ) -> Option<Vec<Type>> {
        let rows = match expr {
            Expr::List(list) => list.elts.as_slice(),
            Expr::Tuple(tuple) => tuple.elts.as_slice(),
            _ => return None,
        };
        let hint_errors = ErrorCollector::new(errors.module().dupe(), errors.style());
        let mut columns = vec![Vec::new(); arity];
        for row in rows {
            if arity == 1 {
                columns[0].push(self.expr_infer(row, &hint_errors).promote(self.stdlib));
                continue;
            }
            let elts = match row {
                Expr::Tuple(tuple) => tuple.elts.as_slice(),
                Expr::List(list) => list.elts.as_slice(),
                _ => return None,
            };
            if elts.len() != arity {
                return None;
            }
            for (column, elt) in columns.iter_mut().zip(elts) {
                column.push(self.expr_infer(elt, &hint_errors).promote(self.stdlib));
            }
        }
        columns
            .into_iter()
            .map(|column| {
                if column.is_empty() {
                    None
                } else {
                    Some(self.unions(column))
                }
            })
            .collect()
    }
}
