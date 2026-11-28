/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::KeyDecorator;
use crate::binding::pytest::PytestParametrization;
use crate::binding::pytest::PytestParametrizationRow;
use crate::error::collector::ErrorCollector;
use crate::types::types::Type;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Infer parameter types only from rows that supply values directly.
    pub(super) fn pytest_parametrize_parameter_types(
        &self,
        decorators: &[Idx<KeyDecorator>],
        errors: &ErrorCollector,
    ) -> SmallMap<Name, Type> {
        let mut columns: SmallMap<Name, Vec<Type>> = SmallMap::new();
        if let Some(pytest_info) = self.bindings().pytest_info() {
            let aliases = pytest_info.aliases();
            let hint_errors = ErrorCollector::new(errors.module().dupe(), errors.style());
            for key in decorators {
                let binding = self.bindings().get::<KeyDecorator>(*key);
                let Some(parametrization) = PytestParametrization::parse(&binding.expr, aliases)
                else {
                    continue;
                };
                if parametrization.direct_parameters().next().is_none() {
                    continue;
                }
                let Some(rows) = parametrization.rows(aliases) else {
                    continue;
                };
                if rows.is_empty() {
                    continue;
                }
                for (index, name) in parametrization.direct_parameters() {
                    let types = rows
                        .iter()
                        .map(|row| {
                            let ty = self.expr_infer(&row.values()[index], &hint_errors);
                            let ty = self.solver().expand(ty, self);
                            if matches!(row, PytestParametrizationRow::Single(_))
                                && ty.as_unions().iter().any(|member| {
                                    matches!(member, Type::ClassType(cls) if {
                                        let qname = cls.qname(self);
                                        qname.id() == "ParameterSet"
                                            && matches!(
                                                qname.module_name().as_str(),
                                                "_pytest.mark.structures" | "pytest"
                                            )
                                    })
                                })
                            {
                                return None;
                            }
                            // Speculative variables must not reach answer sanitization.
                            Some(ty.clean_var().promote(self.stdlib))
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(types) = types {
                        columns.entry(name.clone()).or_default().extend(types);
                    }
                }
            }
        }
        columns
            .into_iter()
            .map(|(name, types)| (name, self.unions(types)))
            .collect()
    }
}
