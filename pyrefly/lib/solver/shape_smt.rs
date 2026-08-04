/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SMT fallback for shape relations that canonicalization cannot decide.
//!
//! The neutral query representation keeps Pyrefly's shape semantics separate
//! from Z3. Unsupported expressions produce `Unknown`; callers must preserve
//! the ordinary type error in that case.

use pyrefly_types::dimension::Int;
use pyrefly_types::literal::Lit;
use pyrefly_types::quantified::QuantifiedKind;
use z3::Config;
use z3::SatResult;
use z3::Solver;
use z3::ast::Bool;
use z3::ast::Int as Z3Int;
use z3::with_z3_config;

use crate::types::types::Type;

const TIMEOUT_MS: u64 = 10;

#[derive(Debug)]
enum ShapeExpr {
    Literal(i64),
    Symbol(usize),
    Add(Box<ShapeExpr>, Box<ShapeExpr>),
    Sub(Box<ShapeExpr>, Box<ShapeExpr>),
    Mul(Box<ShapeExpr>, Box<ShapeExpr>),
    FloorDiv(Box<ShapeExpr>, Box<ShapeExpr>),
}

#[derive(Debug)]
enum ShapeConstraint {
    Equal(ShapeExpr, ShapeExpr),
    // Path assumptions will use these relations once they are collected from
    // control-flow narrowing. Equality is the only production caller today.
    #[allow(dead_code)]
    LessThan(ShapeExpr, ShapeExpr),
    #[allow(dead_code)]
    LessThanOrEqual(ShapeExpr, ShapeExpr),
}

#[derive(Debug)]
struct ShapeQuery {
    symbols: Vec<String>,
    assumptions: Vec<ShapeConstraint>,
    obligation: ShapeConstraint,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ShapeProof {
    Proven,
    Counterexample(Vec<(String, i64)>),
    Unknown,
}

struct QueryBuilder {
    symbols: Vec<(Type, String)>,
}

impl QueryBuilder {
    fn new() -> Self {
        Self {
            symbols: Vec::new(),
        }
    }

    fn expr(&mut self, ty: &Type) -> Option<ShapeExpr> {
        match ty {
            Type::Literal(lit) => match &lit.value {
                Lit::Int(value) => value.as_i64().map(ShapeExpr::Literal),
                _ => None,
            },
            Type::Int(dim) => self.int_expr(dim),
            Type::Quantified(q) if q.kind() == QuantifiedKind::IntVar => Some(self.symbol(ty)),
            Type::TypeVar(tv) if tv.kind() == QuantifiedKind::IntVar => Some(self.symbol(ty)),
            _ => None,
        }
    }

    fn int_expr(&mut self, dim: &Int) -> Option<ShapeExpr> {
        match dim {
            Int::Literal(value) => Some(ShapeExpr::Literal(*value)),
            Int::Symbolic(ty) => self.expr(ty),
            Int::Add(left, right) => Some(ShapeExpr::Add(
                Box::new(self.int_expr(left)?),
                Box::new(self.int_expr(right)?),
            )),
            Int::Sub(left, right) => Some(ShapeExpr::Sub(
                Box::new(self.int_expr(left)?),
                Box::new(self.int_expr(right)?),
            )),
            Int::Mul(left, right) => Some(ShapeExpr::Mul(
                Box::new(self.int_expr(left)?),
                Box::new(self.int_expr(right)?),
            )),
            Int::FloorDiv(left, right) => Some(ShapeExpr::FloorDiv(
                Box::new(self.int_expr(left)?),
                Box::new(self.int_expr(right)?),
            )),
            Int::Int | Int::Pow(_, _) => None,
        }
    }

    fn symbol(&mut self, ty: &Type) -> ShapeExpr {
        let index = self
            .symbols
            .iter()
            .position(|(candidate, _)| candidate == ty)
            .unwrap_or_else(|| {
                let index = self.symbols.len();
                self.symbols.push((ty.clone(), ty.to_string()));
                index
            });
        ShapeExpr::Symbol(index)
    }

    fn equality(mut self, left: &Type, right: &Type) -> Option<ShapeQuery> {
        let left = self.expr(left)?;
        let right = self.expr(right)?;
        Some(ShapeQuery {
            symbols: self.symbols.into_iter().map(|(_, name)| name).collect(),
            assumptions: Vec::new(),
            obligation: ShapeConstraint::Equal(left, right),
        })
    }
}

fn encode_expr(expr: &ShapeExpr, symbols: &[Z3Int]) -> Option<Z3Int> {
    match expr {
        ShapeExpr::Literal(value) => Some(Z3Int::from_i64(*value)),
        ShapeExpr::Symbol(index) => symbols.get(*index).cloned(),
        ShapeExpr::Add(left, right) => {
            Some(encode_expr(left, symbols)? + encode_expr(right, symbols)?)
        }
        ShapeExpr::Sub(left, right) => {
            Some(encode_expr(left, symbols)? - encode_expr(right, symbols)?)
        }
        ShapeExpr::Mul(left, right) => {
            Some(encode_expr(left, symbols)? * encode_expr(right, symbols)?)
        }
        // SMT integer division agrees with Python floor division for a positive
        // divisor. Other divisors stay unsupported until their sign can be
        // established from query assumptions.
        ShapeExpr::FloorDiv(left, right) if matches!(right.as_ref(), ShapeExpr::Literal(1..)) => {
            Some(encode_expr(left, symbols)? / encode_expr(right, symbols)?)
        }
        ShapeExpr::FloorDiv(_, _) => None,
    }
}

fn encode_constraint(constraint: &ShapeConstraint, symbols: &[Z3Int]) -> Option<Bool> {
    match constraint {
        ShapeConstraint::Equal(left, right) => {
            Some(encode_expr(left, symbols)?.eq(encode_expr(right, symbols)?))
        }
        ShapeConstraint::LessThan(left, right) => {
            Some(encode_expr(left, symbols)?.lt(encode_expr(right, symbols)?))
        }
        ShapeConstraint::LessThanOrEqual(left, right) => {
            Some(encode_expr(left, symbols)?.le(encode_expr(right, symbols)?))
        }
    }
}

fn prove(query: &ShapeQuery) -> ShapeProof {
    let mut config = Config::new();
    config.set_timeout_msec(TIMEOUT_MS);
    with_z3_config(&config, || {
        let symbols = (0..query.symbols.len())
            .map(|index| Z3Int::new_const(format!("shape_{index}")))
            .collect::<Vec<_>>();
        let solver = Solver::new();
        for assumption in &query.assumptions {
            let Some(assumption) = encode_constraint(assumption, &symbols) else {
                return ShapeProof::Unknown;
            };
            solver.assert(assumption);
        }
        let Some(obligation) = encode_constraint(&query.obligation, &symbols) else {
            return ShapeProof::Unknown;
        };
        solver.assert(obligation.not());
        match solver.check() {
            SatResult::Unsat => ShapeProof::Proven,
            SatResult::Unknown => ShapeProof::Unknown,
            SatResult::Sat => {
                let Some(model) = solver.get_model() else {
                    return ShapeProof::Unknown;
                };
                let values = symbols
                    .iter()
                    .enumerate()
                    .filter_map(|(index, symbol)| {
                        Some((
                            query.symbols[index].clone(),
                            model.eval(symbol, true)?.as_i64()?,
                        ))
                    })
                    .collect();
                ShapeProof::Counterexample(values)
            }
        }
    })
}

pub(crate) fn prove_equivalent(left: &Type, right: &Type) -> ShapeProof {
    let Some(query) = QueryBuilder::new().equality(left, right) else {
        return ShapeProof::Unknown;
    };
    prove(&query)
}

#[cfg(test)]
mod tests {
    use super::ShapeConstraint;
    use super::ShapeExpr;
    use super::ShapeProof;
    use super::ShapeQuery;
    use super::prove;

    #[test]
    fn proves_universal_inequality() {
        let query = ShapeQuery {
            symbols: vec!["N".to_owned()],
            assumptions: Vec::new(),
            obligation: ShapeConstraint::LessThan(
                ShapeExpr::Symbol(0),
                ShapeExpr::Add(
                    Box::new(ShapeExpr::Symbol(0)),
                    Box::new(ShapeExpr::Literal(1)),
                ),
            ),
        };
        assert_eq!(prove(&query), ShapeProof::Proven);
    }

    #[test]
    fn disproves_nonuniversal_inequality() {
        let query = ShapeQuery {
            symbols: vec!["N".to_owned()],
            assumptions: Vec::new(),
            obligation: ShapeConstraint::LessThanOrEqual(
                ShapeExpr::Add(
                    Box::new(ShapeExpr::Symbol(0)),
                    Box::new(ShapeExpr::Literal(1)),
                ),
                ShapeExpr::Symbol(0),
            ),
        };
        assert!(matches!(prove(&query), ShapeProof::Counterexample(_)));
    }
}
