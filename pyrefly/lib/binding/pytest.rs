/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice;

use pyrefly_graph::index::Idx;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::binding::binding::KeyClass;

// Tracks local names that refer to pytest so we can resolve @pytest.fixture and fixture aliases.
#[derive(Clone, Debug, Default)]
pub struct PytestAliases {
    pytest_module_aliases: Vec<String>,
    fixture_aliases: Vec<String>,
    mark_aliases: Vec<String>,
    param_aliases: Vec<String>,
}

impl PytestAliases {
    pub fn from_module(module: &ModModule) -> Self {
        Self::from_stmts(&module.body)
    }

    pub(crate) fn from_stmts(stmts: &[Stmt]) -> Self {
        let mut aliases = Self::default();
        for stmt in stmts {
            match stmt {
                Stmt::Import(import_stmt) => {
                    for alias in &import_stmt.names {
                        if alias.name.id == "pytest" {
                            let local_name = alias
                                .asname
                                .as_ref()
                                .map(|asname| asname.id.as_str())
                                .unwrap_or(alias.name.id.as_str());
                            aliases.pytest_module_aliases.push(local_name.to_owned());
                        }
                    }
                }
                Stmt::ImportFrom(StmtImportFrom {
                    module: Some(module),
                    names,
                    ..
                }) if module.id == "pytest" || module.id.starts_with("pytest.") => {
                    for alias in names {
                        if alias.name.id == "fixture" {
                            let local_name = alias
                                .asname
                                .as_ref()
                                .map(|asname| asname.id.as_str())
                                .unwrap_or(alias.name.id.as_str());
                            aliases.fixture_aliases.push(local_name.to_owned());
                        } else if module.id == "pytest" && alias.name.id == "mark" {
                            let local_name = alias.asname.as_ref().unwrap_or(&alias.name);
                            aliases.mark_aliases.push(local_name.id.to_string());
                        } else if module.id == "pytest" && alias.name.id == "param" {
                            let local_name = alias.asname.as_ref().unwrap_or(&alias.name);
                            aliases.param_aliases.push(local_name.id.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        aliases
    }

    pub fn is_empty(&self) -> bool {
        self.pytest_module_aliases.is_empty()
            && self.fixture_aliases.is_empty()
            && self.mark_aliases.is_empty()
            && self.param_aliases.is_empty()
    }

    fn is_pytest_module_alias(&self, name: &Name) -> bool {
        self.pytest_module_aliases
            .iter()
            .any(|alias| alias == name.as_str())
    }

    fn is_fixture_alias(&self, name: &Name) -> bool {
        self.fixture_aliases
            .iter()
            .any(|alias| alias == name.as_str())
    }

    /// Recognize parametrization through pytest module and mark aliases.
    fn is_parametrize(&self, expr: &Expr) -> bool {
        let Expr::Attribute(attr) = expr else {
            return false;
        };
        attr.attr.id == "parametrize"
            && match attr.value.as_ref() {
                Expr::Name(name) => self
                    .mark_aliases
                    .iter()
                    .any(|alias| alias == name.id.as_str()),
                Expr::Attribute(mark) if mark.attr.id == "mark" => {
                    matches!(mark.value.as_ref(), Expr::Name(name) if self.is_pytest_module_alias(name.id()))
                }
                _ => false,
            }
    }

    fn is_param(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Name(name) => self
                .param_aliases
                .iter()
                .any(|alias| alias == name.id.as_str()),
            Expr::Attribute(attr) if attr.attr.id == "param" => {
                matches!(attr.value.as_ref(), Expr::Name(name) if self.is_pytest_module_alias(name.id()))
            }
            _ => false,
        }
    }
}

/// A parsed parametrization separates direct values from fixture inputs.
pub(crate) struct PytestParametrization<'a> {
    names: Vec<Name>,
    indirect: Vec<Name>,
    values: &'a Expr,
    /// None means that the row layout depends on the pytest version.
    single_value: Option<bool>,
}

pub(crate) enum PytestParametrizationRow<'a> {
    Single(&'a Expr),
    Unpacked(&'a [Expr]),
}

impl<'a> PytestParametrizationRow<'a> {
    pub(crate) fn values(&self) -> &'a [Expr] {
        match self {
            Self::Single(value) => slice::from_ref(*value),
            Self::Unpacked(values) => values,
        }
    }
}

impl<'a> PytestParametrization<'a> {
    pub(crate) fn parse(expr: &'a Expr, aliases: &PytestAliases) -> Option<Self> {
        let Expr::Call(call) = expr else {
            return None;
        };
        if !aliases.is_parametrize(&call.func)
            || call
                .arguments
                .args
                .iter()
                .any(|arg| matches!(arg, Expr::Starred(_)))
            || call.arguments.keywords.iter().any(|kw| kw.arg.is_none())
        {
            return None;
        }
        let argnames = call.arguments.find_argument_value("argnames", 0)?;
        let names = parametrize_names(argnames)?;
        let values = call.arguments.find_argument_value("argvalues", 1)?;
        let indirect = match call.arguments.find_argument_value("indirect", 2) {
            None => Vec::new(),
            Some(Expr::BooleanLiteral(value)) if !value.value => Vec::new(),
            Some(Expr::BooleanLiteral(_)) => names.clone(),
            Some(expr @ (Expr::List(_) | Expr::Tuple(_))) => parametrize_names(expr)?,
            _ => return None,
        };
        let single_value = match argnames {
            Expr::StringLiteral(lit) if names.len() == 1 => {
                (!lit.value.to_str().trim_end().ends_with(',')).then_some(true)
            }
            _ => Some(false),
        };
        Some(Self {
            names,
            indirect,
            values,
            single_value,
        })
    }

    pub(crate) fn direct_parameters(&self) -> impl Iterator<Item = (usize, &Name)> {
        self.names
            .iter()
            .enumerate()
            .filter(|(_, name)| !self.indirect.contains(name))
    }

    /// Return complete rows, or skip inference when a row cannot be interpreted safely.
    pub(crate) fn rows(
        &self,
        aliases: &PytestAliases,
    ) -> Option<Vec<PytestParametrizationRow<'a>>> {
        let single_value = self.single_value?;
        let rows = match self.values {
            Expr::List(list) => &list.elts,
            Expr::Tuple(tuple) => &tuple.elts,
            _ => return None,
        };
        rows.iter()
            .map(|row| {
                let parsed = match row {
                    Expr::Call(call) if aliases.is_param(&call.func) => {
                        PytestParametrizationRow::Unpacked(call.arguments.args.as_ref())
                    }
                    Expr::Starred(_) => return None,
                    _ if single_value => PytestParametrizationRow::Single(row),
                    Expr::Tuple(tuple) => PytestParametrizationRow::Unpacked(&tuple.elts),
                    Expr::List(list) => PytestParametrizationRow::Unpacked(&list.elts),
                    _ => return None,
                };
                let values = parsed.values();
                (values.len() == self.names.len()
                    && !values.iter().any(|value| matches!(value, Expr::Starred(_))))
                .then_some(parsed)
            })
            .collect()
    }
}

fn parametrize_names(expr: &Expr) -> Option<Vec<Name>> {
    let elts = match expr {
        Expr::StringLiteral(lit) => {
            let names = lit
                .value
                .to_str()
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(Name::new)
                .collect::<Vec<_>>();
            return (!names.is_empty()).then_some(names);
        }
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::List(list) => &list.elts,
        _ => return None,
    };
    elts.iter()
        .map(|elt| match elt {
            Expr::StringLiteral(lit) if !lit.value.to_str().is_empty() => {
                Some(Name::new(lit.value.to_str()))
            }
            _ => None,
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct PytestFixtureDefinitions {
    module: Option<ShortIdentifier>,
    classes: SmallMap<Idx<KeyClass>, ShortIdentifier>,
}

#[derive(Clone, Debug)]
pub struct PytestBindingInfo {
    aliases: PytestAliases,
    fixtures: SmallMap<Name, PytestFixtureDefinitions>,
}

impl PytestBindingInfo {
    pub fn from_module(module: &ModModule) -> Option<Self> {
        let aliases = PytestAliases::from_module(module);
        if aliases.is_empty() {
            return None;
        }
        Some(Self {
            aliases,
            fixtures: SmallMap::new(),
        })
    }

    pub fn aliases(&self) -> &PytestAliases {
        &self.aliases
    }

    pub fn add_fixture_definition(
        &mut self,
        name: Name,
        return_type_key: ShortIdentifier,
        class_key: Option<Idx<KeyClass>>,
    ) {
        let definitions = self
            .fixtures
            .entry(name)
            .or_insert_with(|| PytestFixtureDefinitions {
                module: None,
                classes: SmallMap::new(),
            });
        match class_key {
            Some(class_key) => {
                definitions.classes.insert(class_key, return_type_key);
            }
            None => {
                definitions.module = Some(return_type_key);
            }
        }
    }

    pub fn visible_fixture_class_key(
        &self,
        name: &Name,
        class_key: Option<&Idx<KeyClass>>,
    ) -> Option<Option<Idx<KeyClass>>> {
        let defs = self.fixtures.get(name)?;
        if let Some(class_key) = class_key
            && defs.classes.contains_key(class_key)
        {
            return Some(Some(*class_key));
        }
        defs.module.as_ref()?;
        Some(None)
    }

    pub fn is_fixture_definition(
        &self,
        func_name: &ruff_python_ast::Identifier,
        class_key: Option<&Idx<KeyClass>>,
    ) -> bool {
        let Some(defs) = self.fixtures.get(&func_name.id) else {
            return false;
        };
        let func_id = ShortIdentifier::new(func_name);
        match class_key {
            Some(class_key) => defs.classes.get(class_key) == Some(&func_id),
            None => defs.module.as_ref() == Some(&func_id),
        }
    }
}

pub fn is_pytest_fixture_function(function_def: &StmtFunctionDef, aliases: &PytestAliases) -> bool {
    function_def
        .decorator_list
        .iter()
        .any(|decorator| is_pytest_fixture_decorator(&decorator.expression, aliases))
}

pub(crate) fn is_pytest_fixture_decorator(expr: &Expr, aliases: &PytestAliases) -> bool {
    match expr {
        Expr::Name(name) => aliases.is_fixture_alias(name.id()),
        Expr::Attribute(attr) => {
            attr.attr.id() == "fixture"
                && matches!(
                    attr.value.as_ref(),
                    Expr::Name(base) if aliases.is_pytest_module_alias(base.id())
                )
        }
        Expr::Call(call) => is_pytest_fixture_decorator(call.func.as_ref(), aliases),
        _ => false,
    }
}
