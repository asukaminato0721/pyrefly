/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;

use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::is_python_identifier;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Required;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::class::DjangoSpecialization;
use pyrefly_types::function::DescriptorSetterMetadata;
use pyrefly_types::function::FuncMetadata;
use pyrefly_types::function::Function;
use pyrefly_types::function::PropertyMetadata;
use pyrefly_types::function::PropertyRole;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::keywords::KwCall;
use pyrefly_types::keywords::TypeMap;
use pyrefly_types::known_class::KnownClass;
use pyrefly_types::literal::Lit;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::AnyStyle;
use pyrefly_types::types::Overload;
use pyrefly_types::types::OverloadType;
use pyrefly_types::types::Type;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassAttribute;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::enums::VALUE_PROP;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::class_metadata::DjangoReverseRelationIndex;
use crate::alt::types::instance::Instance;
use crate::binding::binding::BindingDjangoRelations;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

/// Django stubs use this attribute to specify the Python type that a field should infer to
const DJANGO_PRIVATE_GET_TYPE: Name = Name::new_static("_pyi_private_get_type");
const DJANGO_PRIVATE_SET_TYPE: Name = Name::new_static("_pyi_private_set_type");
const COMBINABLE: Name = Name::new_static("Combinable");

pub fn is_django_choices_subclass(bases_with_metadata: &[(Class, &ClassMetadata)]) -> bool {
    bases_with_metadata.iter().any(|(base, base_meta)| {
        base.is(KnownClass::DjangoChoices)
            || base_meta
                .enum_metadata()
                .as_ref()
                .is_some_and(|meta| meta.is_django)
    })
}

/// Strip the label element from a Django enum tuple value.
/// Django `Choices` enums use `(value, label)` tuples; this strips the last
/// element (the label) and returns the remaining value portion.
pub fn transform_django_enum_value(ty: Type, heap: &TypeHeap) -> Type {
    match ty {
        Type::Tuple(Tuple::Concrete(elements)) if elements.len() >= 2 => {
            let value_len = elements.len() - 1;
            heap.mk_concrete_tuple(elements.into_iter().take(value_len).collect())
        }
        ty => ty,
    }
}
const CHOICES: Name = Name::new_static("choices");
const LABEL: Name = Name::new_static("label");
const LABELS: Name = Name::new_static("labels");
const VALUES: Name = Name::new_static("values");
const ID: Name = Name::new_static("id");
const PK: Name = Name::new_static("pk");
const AUTO_FIELD: Name = Name::new_static("AutoField");
const NULL: Name = Name::new_static("null");
const BLANK: Name = Name::new_static("blank");
const MODEL: Name = Name::new_static("Model");
const ANNOTATE: Name = Name::new_static("annotate");
const MANYRELATEDMANAGER: Name = Name::new_static("ManyRelatedManager");
const SYMMETRICAL: Name = Name::new_static("symmetrical");
const AS_MANAGER: Name = Name::new_static("as_manager");

pub(crate) struct DjangoAnnotateCall {
    metadata: FuncMetadata,
    attributes: SmallMap<Name, (Type, TextRange)>,
}

/// Find a keyword argument by name and return its value expression.
fn find_keyword<'a>(call_expr: &'a ExprCall, name: &Name) -> Option<&'a Expr> {
    call_expr
        .arguments
        .keywords
        .iter()
        .find(|kw| kw.arg.as_ref().is_some_and(|n| n.as_str() == name.as_str()))
        .map(|kw| &kw.value)
}

/// Check if a keyword argument with the given name exists and has value `True`.
fn has_keyword_true(call_expr: &ExprCall, name: &Name) -> bool {
    find_keyword(call_expr, name)
        .is_some_and(|v| matches!(v, Expr::BooleanLiteral(lit) if lit.value))
}

const RELATED_NAME: Name = Name::new_static("related_name");

const RELATED_MANAGER: Name = Name::new_static("RelatedManager");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DjangoRelationKind {
    ForeignKey,
    OneToOne,
    ManyToMany,
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    fn is_django_annotate(metadata: &FuncMetadata) -> bool {
        let Some(symbol) = metadata.kind.to_func_symbol() else {
            return false;
        };
        let Some(cls) = symbol.cls.as_ref() else {
            return false;
        };
        symbol.name == ANNOTATE
            && (cls.is(KnownClass::DjangoQuerySet) || cls.is(KnownClass::DjangoManager))
    }

    pub(crate) fn infer_django_annotate_call(
        &self,
        callee: &Type,
        arguments: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<DjangoAnnotateCall> {
        let metadata = callee.toplevel_func_metadata()?.clone();
        if !Self::is_django_annotate(&metadata) {
            return None;
        }

        let mut attributes = SmallMap::new();
        // Positional annotations require Django's generated expression aliases.
        for keyword in &arguments.keywords {
            if let Some(name) = &keyword.arg {
                attributes.insert(
                    name.id.clone(),
                    (
                        self.expr_infer(&keyword.value, errors),
                        keyword.value.range(),
                    ),
                );
            }
        }
        if attributes.is_empty() {
            None
        } else {
            Some(DjangoAnnotateCall {
                metadata,
                attributes,
            })
        }
    }

    pub(crate) fn apply_django_annotate_call(
        &self,
        result: Type,
        call: Option<DjangoAnnotateCall>,
        errors: &ErrorCollector,
    ) -> Type {
        let Some(call) = call else {
            return result;
        };
        let mut queryset = match result {
            Type::ClassType(queryset) => queryset,
            result => return result,
        };
        let Some(mut base) = self.django_queryset_base(&queryset) else {
            return self.heap.mk_class_type(queryset);
        };
        if base.targs().len() != 2 {
            return self.heap.mk_class_type(queryset);
        }

        let row = base.targs().as_slice()[1].clone();
        if !self.is_django_model_row(&row) {
            return self.heap.mk_class_type(queryset);
        }
        // Django resolves keyword annotations in order, so later ones can reference earlier ones.
        let mut attributes = TypeMap::new();
        for (name, (expression, range)) in call.attributes {
            let context = self.heap.mk_kw_call(KwCall {
                func_metadata: call.metadata.clone(),
                keywords: attributes.clone(),
                return_ty: row.clone(),
            });
            attributes.0.insert(
                name,
                self.django_expression_type(&expression, &context, range, errors),
            );
        }
        let annotation = self.heap.mk_kw_call(KwCall {
            func_metadata: call.metadata,
            keywords: attributes,
            return_ty: row.clone(),
        });
        // The intersection preserves call-site metadata while generic QuerySet operations use the
        // original row as their fallback type.
        base.targs_mut().as_mut()[1] = self.heap.mk_intersect(vec![annotation], row);
        if queryset.is(KnownClass::DjangoQuerySet) {
            queryset = base;
        } else {
            queryset = queryset.with_django_base(base);
        }
        self.heap.mk_class_type(queryset)
    }

    /// Preserve constructor arguments whose meaning depends on the receiving query.
    pub(crate) fn apply_django_expression_call(
        &self,
        result: Type,
        callee: &Type,
        arguments: &Arguments,
        errors: &ErrorCollector,
    ) -> Type {
        let Type::ClassType(expression) = &result else {
            return result;
        };
        if self.unwrap_class_object_silently(callee).is_none()
            || !(expression.is(KnownClass::DjangoF)
                || self
                    .get_mro_for_class(expression.class_object())
                    .ancestors(self.stdlib)
                    .any(|base| {
                        base.is(KnownClass::DjangoBaseExpression) || base.is(KnownClass::DjangoF)
                    }))
        {
            return result;
        }
        let source = if expression.is(KnownClass::DjangoF) || expression.is(KnownClass::DjangoValue)
        {
            arguments
                .find_argument_value(
                    if expression.is(KnownClass::DjangoF) {
                        "name"
                    } else {
                        "value"
                    },
                    0,
                )
                .map(|expr| self.expr_infer(expr, errors))
                .unwrap_or_else(|| self.heap.mk_any_implicit())
        } else if expression.is(KnownClass::DjangoWhen) {
            arguments
                .find_argument_value("then", 1)
                .map(|expr| self.expr_infer(expr, errors))
                .unwrap_or_else(|| self.heap.mk_none())
        } else {
            let mut sources = if expression.is(KnownClass::DjangoCast)
                || expression.is(KnownClass::DjangoExpressionWrapper)
            {
                arguments
                    .find_argument_value("expression", 0)
                    .into_iter()
                    .map(|expr| self.expr_infer(expr, errors))
                    .collect()
            } else {
                arguments
                    .args
                    .iter()
                    .map(|expr| self.expr_infer(expr, errors))
                    .collect()
            };
            // A star argument can supply an unknown number of sources.
            if arguments
                .args
                .iter()
                .any(|arg| matches!(arg, Expr::Starred(_)))
            {
                sources = vec![self.heap.mk_any_implicit()];
            }
            self.heap.mk_concrete_tuple(sources)
        };
        let output_field = if expression.is(KnownClass::DjangoValue)
            || expression.is(KnownClass::DjangoExpressionWrapper)
            || expression.is(KnownClass::DjangoCast)
        {
            arguments.find_argument_value("output_field", 1)
        } else {
            arguments.find_keyword("output_field").map(|kw| &kw.value)
        };
        let output_type = output_field
            .map(|expr| self.django_expression_field_type(&self.expr_infer(expr, errors)))
            .unwrap_or_else(|| self.heap.mk_none());
        let default = arguments
            .find_keyword("default")
            .map(|kw| self.expr_infer(&kw.value, errors))
            .unwrap_or_else(|| self.heap.mk_none());
        self.heap
            .mk_class_type(
                expression
                    .clone()
                    .with_django_expression(source, output_type, default),
            )
    }

    /// Read an expression's output field without requiring a model declaration.
    fn django_expression_field_type(&self, field: &Type) -> Type {
        let field = self.solver().force(field.clone(), self);
        if field.is_none() {
            return self.heap.mk_none();
        }
        if let Type::ClassType(field) = &field
            && self.inherits_from_django_field(field.class_object())
        {
            for (param, arg) in field.targs().iter_paired() {
                let arg = self.solver().force(arg.clone(), self);
                if param.name().as_str() == "_GT" && !matches!(arg, Type::Any(AnyStyle::Implicit)) {
                    return arg;
                }
            }
            if let Some(member) =
                self.get_class_member(field.class_object(), &DJANGO_PRIVATE_GET_TYPE)
            {
                return member.ty();
            }
        }
        self.heap.mk_any_implicit()
    }

    /// Resolve supported expressions to database values in the receiving row.
    fn django_expression_type(
        &self,
        expression: &Type,
        row: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        self.distribute_over_union(expression, |expression| {
            let Type::ClassType(cls) = expression else {
                return self.heap.mk_any_implicit();
            };
            let ancestors = self.get_mro_for_class(cls.class_object());
            let is = |known| {
                cls.is(known) || ancestors.ancestors(self.stdlib).any(|base| base.is(known))
            };
            let (source, output_type, default) = match cls.django_specialization() {
                Some(DjangoSpecialization::Expression {
                    source,
                    output_type,
                    default,
                }) => (source.clone(), output_type.clone(), default.clone()),
                _ => (
                    self.heap.mk_any_implicit(),
                    self.heap.mk_none(),
                    self.heap.mk_none(),
                ),
            };
            if is(KnownClass::DjangoF) {
                return self.django_expression_reference(&source, row, range, errors);
            }
            if is(KnownClass::DjangoWhen) {
                return self.django_expression_source_type(&source, row, range, errors, true);
            }
            let sources = if is(KnownClass::DjangoValue) {
                vec![source.promote(self.stdlib)]
            } else if let Type::Tuple(Tuple::Concrete(sources)) = source {
                sources
                    .iter()
                    .map(|source| {
                        self.django_expression_source_type(source, row, range, errors, true)
                    })
                    .collect()
            } else {
                vec![self.heap.mk_any_implicit()]
            };
            let default = self.django_expression_source_type(
                &default,
                row,
                range,
                errors,
                is(KnownClass::DjangoCase),
            );
            let mut source_type = self.unions(sources.clone());
            if is(KnownClass::DjangoCase) {
                source_type = self.union(source_type, default.clone());
            }
            let explicit_output = !output_type.is_none();
            let mut ty = if explicit_output {
                output_type
            } else if is(KnownClass::DjangoValue) {
                match &source_type {
                    Type::None => source_type.clone(),
                    Type::ClassType(value)
                        if matches!(
                            value.known(),
                            Some(
                                KnownClass::Bool
                                    | KnownClass::Bytes
                                    | KnownClass::Float
                                    | KnownClass::Int
                                    | KnownClass::Str
                                    | KnownClass::Decimal
                                    | KnownClass::Datetime
                                    | KnownClass::DatetimeDate
                                    | KnownClass::DatetimeTime
                                    | KnownClass::DatetimeTimedelta
                            )
                        ) => source_type.clone(),
                    _ => self.heap.mk_any_implicit(),
                }
            } else {
                let field = self.attr_infer_for_type(
                    expression,
                    &Name::new_static("output_field"),
                    range,
                    &self.error_swallower(),
                    None,
                );
                let output = self.django_expression_field_type(&field);
                if !output.is_any() {
                    output
                } else if is(KnownClass::DjangoSum)
                    || is(KnownClass::DjangoMin)
                    || is(KnownClass::DjangoMax)
                    || is(KnownClass::DjangoCase)
                    || is(KnownClass::DjangoCoalesce)
                {
                    source_type.clone()
                } else if is(KnownClass::DjangoAvg)
                    || is(KnownClass::DjangoStdDev)
                    || is(KnownClass::DjangoVariance)
                {
                    self.distribute_over_union(&source_type, |ty| {
                        if matches!(ty, Type::ClassType(cls) if cls.is(KnownClass::Int) || cls.is(KnownClass::Bool)) {
                            self.heap.mk_class_type(self.stdlib.float().clone())
                        } else if ty.is_none()
                            || matches!(ty, Type::ClassType(cls) if matches!(
                                cls.known(),
                                Some(KnownClass::Float | KnownClass::Decimal | KnownClass::DatetimeTimedelta)
                            ))
                        {
                            ty.clone()
                        } else {
                            self.heap.mk_any_implicit()
                        }
                    })
                } else {
                    self.heap.mk_any_implicit()
                }
            };
            if is(KnownClass::DjangoSum) && !explicit_output {
                // SQL adds boolean values as integers and can produce values greater than one.
                ty = self.distribute_over_union(&ty, |ty| {
                    if matches!(ty, Type::ClassType(cls) if cls.is(KnownClass::Bool)) {
                        self.heap.mk_class_type(self.stdlib.int().clone())
                    } else {
                        ty.clone()
                    }
                });
            }
            let nullable = if is(KnownClass::DjangoAggregate) {
                !is(KnownClass::DjangoCount) && self.django_expression_may_be_none(&default)
            } else if is(KnownClass::DjangoCase) {
                self.django_expression_may_be_none(&source_type)
            } else if is(KnownClass::DjangoCoalesce) {
                sources.iter().all(|ty| self.django_expression_may_be_none(ty))
            } else if is(KnownClass::DjangoConcat) {
                false
            } else if is(KnownClass::DjangoFunc)
                || is(KnownClass::DjangoExpressionWrapper)
                || is(KnownClass::DjangoValue)
            {
                sources.iter().any(|ty| self.django_expression_may_be_none(ty))
            } else {
                false
            };
            if nullable {
                self.union(ty, self.heap.mk_none())
            } else {
                self.distribute_over_union(&ty, |ty| {
                    if ty.is_none() {
                        self.heap.mk_never()
                    } else {
                        ty.clone()
                    }
                })
            }
        })
    }

    /// Interpret strings as field paths where Django uses F(). Other defaults remain literal values.
    fn django_expression_source_type(
        &self,
        source: &Type,
        row: &Type,
        range: TextRange,
        errors: &ErrorCollector,
        strings_are_fields: bool,
    ) -> Type {
        self.distribute_over_union(source, |source| {
            if strings_are_fields
                && let Type::Literal(lit) = source
                && let Lit::Str(path) = &lit.value
            {
                // Count("*") counts rows rather than resolving a model field.
                if path.as_str() == "*" {
                    return self.heap.mk_any_implicit();
                }
                return self.django_expression_reference(source, row, range, errors);
            }
            if let Type::ClassType(cls) = source
                && (cls.is(KnownClass::DjangoF)
                    || self
                        .get_mro_for_class(cls.class_object())
                        .ancestors(self.stdlib)
                        .any(|base| {
                            base.is(KnownClass::DjangoBaseExpression)
                                || base.is(KnownClass::DjangoF)
                        }))
            {
                self.django_expression_type(source, row, range, errors)
            } else {
                source.clone().promote(self.stdlib)
            }
        })
    }

    /// Unknown values can be SQL NULL as well as values of an unresolved type.
    fn django_expression_may_be_none(&self, ty: &Type) -> bool {
        let mut nullable = false;
        self.map_over_union(ty, |ty| nullable |= ty.is_none() || ty.is_any());
        nullable
    }

    /// Resolve a literal Django field path, including traversal through related managers.
    fn django_expression_reference(
        &self,
        source: &Type,
        row: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let Type::Literal(lit) = source else {
            return self.heap.mk_any_implicit();
        };
        let Lit::Str(path) = &lit.value else {
            return self.heap.mk_any_implicit();
        };
        let mut ty = row.clone();
        for part in path.split("__") {
            ty = self.distribute_over_union(&ty, |base| {
                if base.is_none() {
                    return base.clone();
                }
                let related_manager = self.is_django_manager_type(base);
                let base = if related_manager && let Type::ClassType(cls) = base {
                    self.get_mro_for_class(cls.class_object())
                        .ancestors(self.stdlib)
                        .find(|base| base.is(KnownClass::DjangoBaseManager))
                        .map(|base| base.substitute_with(&cls.substitution(), self))
                        .and_then(|base| base.targs().as_slice().first().cloned())
                        .unwrap_or_else(|| self.heap.mk_any_implicit())
                } else {
                    base.clone()
                };
                let mut name = Name::new(part);
                if let Some(ty) = self.django_annotated_fields(&base).get(&name) {
                    return ty.clone();
                }
                let Some(model) = self.attribute_lookup_classes(&base).and_then(|classes| {
                    classes
                        .into_iter()
                        .find(|cls| self.get_metadata_for_class(cls).is_django_model())
                }) else {
                    // SQL transforms can follow a scalar field and need not be Python attributes.
                    return self.heap.mk_any_implicit();
                };
                if !self.has_attr(&base, &name) {
                    // Default reverse query names omit the accessor's `_set` suffix.
                    let reverse = Name::new(format!("{part}_set"));
                    let reverse_type = self.attr_infer_for_type(
                        &base,
                        &reverse,
                        range,
                        &self.error_swallower(),
                        None,
                    );
                    if self.is_django_manager_type(&reverse_type) {
                        name = reverse;
                    } else {
                        self.error(
                            errors,
                            range,
                            ErrorKind::MissingAttributeDjangoField,
                            format!(
                                "Django field path `{path}` has no field `{part}` on model `{}`",
                                model.name(self)
                            ),
                        );
                        return self.heap.mk_any_implicit();
                    }
                }
                let ty =
                    self.attr_infer_for_type(&base, &name, range, &self.error_swallower(), None);
                if related_manager {
                    // A reverse relation can have no matching rows in a left join.
                    self.union(ty, self.heap.mk_none())
                } else {
                    ty
                }
            });
        }
        self.distribute_over_union(&ty, |ty| {
            if let Type::ClassType(cls) = ty
                && self
                    .get_metadata_for_class(cls.class_object())
                    .is_django_model()
            {
                // A query returns a relation's stored key, rather than a model instance.
                self.get_pk_field_type(cls.class_object())
                    .map(|(ty, _)| ty)
                    .unwrap_or_else(|| self.heap.mk_any_implicit())
            } else if self.is_django_manager_type(ty) {
                self.heap.mk_any_implicit()
            } else {
                ty.clone()
            }
        })
    }

    fn is_django_model_row(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) => self
                .get_metadata_for_class(cls.class_object())
                .is_django_model(),
            Type::KwCall(call) => self.is_django_model_row(&call.return_ty),
            Type::Intersect(intersection) => self.is_django_model_row(&intersection.1),
            _ => false,
        }
    }

    fn collect_django_annotated_fields(&self, ty: &Type, fields: &mut SmallMap<Name, Type>) {
        match ty {
            Type::KwCall(call) => {
                self.collect_django_annotated_fields(&call.return_ty, fields);
                if Self::is_django_annotate(&call.func_metadata) {
                    for (name, ty) in &call.keywords.0 {
                        fields.insert(name.clone(), ty.clone());
                    }
                }
            }
            Type::Intersect(intersection) => {
                for member in &intersection.0 {
                    self.collect_django_annotated_fields(member, fields);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn django_annotated_fields(&self, ty: &Type) -> SmallMap<Name, Type> {
        let mut fields = SmallMap::new();
        self.collect_django_annotated_fields(ty, &mut fields);
        fields
    }

    fn is_one_to_one_field(&self, field: &Class) -> bool {
        field.is(KnownClass::DjangoOneToOneField)
    }

    /// Fill inferred descriptor arguments while retaining declared arguments.
    fn specialize_django_field_descriptor(
        &self,
        field: &ClassType,
        get_type: Type,
        set_type: Type,
        infer_arguments: bool,
    ) -> ClassType {
        let mut field = field.clone();
        for (param, arg) in field.targs_mut().iter_paired_mut() {
            match param.name().as_str() {
                "_ST" if infer_arguments || matches!(arg, Type::Any(AnyStyle::Implicit)) => {
                    *arg = set_type.clone();
                }
                "_GT" if infer_arguments || matches!(arg, Type::Any(AnyStyle::Implicit)) => {
                    *arg = get_type.clone();
                }
                "_To" | "_Through" if matches!(arg, Type::Any(AnyStyle::Implicit)) => {
                    if let Type::ClassType(manager) = &get_type
                        && let [target, through] = manager.targs().as_slice()
                    {
                        *arg = if param.name().as_str() == "_To" {
                            target.clone()
                        } else {
                            through.clone()
                        };
                    }
                }
                _ => {}
            }
        }
        field
    }

    /// Specialize model fields while retaining their normal descriptor overloads.
    pub fn get_django_class_field_type(
        &self,
        ty: &Type,
        class: &Class,
        name: &Name,
        definition: &ClassFieldDefinition,
    ) -> Option<Type> {
        let initial_value = match definition {
            ClassFieldDefinition::AssignedInBody { value, .. } => match value.as_ref() {
                ExprOrBinding::Expr(expr) => Some(expr),
                ExprOrBinding::Binding(_) => None,
            },
            _ => None,
        };
        if !self.get_metadata_for_class(class).is_django_model() {
            return self
                .get_django_field_types(ty, class, Some(name), initial_value)
                .map(|(get_type, _)| get_type);
        }
        let has_annotation_type = match definition {
            ClassFieldDefinition::AssignedInBody {
                annotation: Some(annotation),
                ..
            }
            | ClassFieldDefinition::DeclaredByAnnotation { annotation, .. } => {
                self.get_idx(*annotation).annotation.ty.is_some()
            }
            _ => false,
        };
        // Constructor type parameters describe inference holes before arguments such as
        // `db_default` solve them. A specialized constructor has no such parameters.
        let infer_arguments = !has_annotation_type
            && initial_value
                .and_then(|expr| expr.as_call_expr())
                .and_then(|call| {
                    self.unwrap_class_object_silently(
                        &self.expr_infer(&call.func, &self.error_swallower()),
                    )
                })
                .is_some_and(|(parameters, _)| !parameters.is_empty());
        let specialized = self.distribute_over_union(ty, |member| {
            let types = if infer_arguments && let Type::ClassType(field) = member {
                self.get_django_field_types_from_class(
                    field.class_object(),
                    class,
                    Some(name),
                    initial_value,
                )
            } else {
                self.get_django_field_types(member, class, Some(name), initial_value)
            };
            let Some((get_type, set_type)) = types else {
                return member.clone();
            };
            if let Type::ClassType(field) = member
                && self.inherits_from_django_field(field.class_object())
            {
                let Type::ClassType(field) = self.solver().force(member.clone(), self) else {
                    unreachable!("forcing a field instance preserves its class type")
                };
                let override_type = if !infer_arguments
                    && field.targs().iter_paired().any(|(param, arg)| {
                        param.name().as_str() == "_ST"
                            && !matches!(arg, Type::Any(AnyStyle::Implicit))
                    }) {
                    set_type.clone()
                } else {
                    get_type.clone()
                };
                let field = self.specialize_django_field_descriptor(
                    &field,
                    get_type.clone(),
                    set_type.clone(),
                    infer_arguments,
                );
                self.heap
                    .mk_class_type(field.with_django_field(get_type, set_type, override_type))
            } else {
                get_type
            }
        });
        (specialized != *ty).then_some(specialized)
    }

    /// Resolve constructed fields to their instance read and assignment contracts.
    pub(crate) fn django_field_instance_types(&self, ty: &Type) -> Option<(Type, Type)> {
        match ty {
            Type::ClassType(cls) => match cls.django_specialization()? {
                DjangoSpecialization::Field {
                    get_type, set_type, ..
                } => Some((get_type.clone(), set_type.clone())),
                _ => None,
            },
            Type::Union(union) => {
                let (index, (get_type, set_type)) =
                    union
                        .members
                        .iter()
                        .enumerate()
                        .find_map(|(index, member)| {
                            self.django_field_instance_types(member)
                                .map(|types| (index, types))
                        })?;
                let mut get_types = union.members[..index].to_vec();
                let mut set_types = get_types.clone();
                get_types.push(get_type);
                set_types.push(set_type);
                for member in &union.members[index + 1..] {
                    let (get_type, set_type) = self
                        .django_field_instance_types(member)
                        .unwrap_or_else(|| (member.clone(), member.clone()));
                    get_types.push(get_type);
                    set_types.push(set_type);
                }
                // Reading may produce any branch, but a write must satisfy every branch.
                Some((
                    self.unions(get_types),
                    self.intersect_guarantees(&set_types),
                ))
            }
            _ => None,
        }
    }

    /// Return the read and assignment types of a Django field.
    pub fn get_django_field_types(
        &self,
        ty: &Type,
        class: &Class,
        field_name: Option<&Name>,
        initial_value_expr: Option<&Expr>,
    ) -> Option<(Type, Type)> {
        match ty {
            Type::ClassType(cls) if cls.is(KnownClass::DjangoGetter) => cls
                .targs()
                .as_slice()
                .first()
                .map(|ty| (ty.clone(), ty.clone())),
            Type::ClassType(cls) => {
                let (mut get_type, mut set_type) = self.get_django_field_types_from_class(
                    cls.class_object(),
                    class,
                    field_name,
                    initial_value_expr,
                )?;
                // Explicit descriptor arguments take precedence over inferred defaults.
                for (param, arg) in cls.targs().iter_paired() {
                    let arg = self.solver().force(arg.clone(), self);
                    if !matches!(arg, Type::Any(AnyStyle::Implicit)) {
                        match param.name().as_str() {
                            "_GT" => get_type = arg,
                            "_ST" => set_type = arg,
                            _ => {}
                        }
                    }
                }
                Some((get_type, set_type))
            }
            Type::ClassDef(cls) => {
                self.get_django_field_types_from_class(cls, class, field_name, initial_value_expr)
            }
            Type::Union(f) => {
                let (first_index, (get_type, set_type)) =
                    f.members.iter().enumerate().find_map(|(index, variant)| {
                        self.get_django_field_types(variant, class, field_name, initial_value_expr)
                            .map(|types| (index, types))
                    })?;
                // Ordinary unions need no collections when none of their members transform.
                let mut get_types = Vec::with_capacity(f.members.len());
                let mut set_types = Vec::with_capacity(f.members.len());
                get_types.extend_from_slice(&f.members[..first_index]);
                set_types.extend_from_slice(&f.members[..first_index]);
                get_types.push(get_type);
                set_types.push(set_type);
                for variant in &f.members[first_index + 1..] {
                    let (get_type, set_type) = self
                        .get_django_field_types(variant, class, field_name, initial_value_expr)
                        .unwrap_or_else(|| (variant.clone(), variant.clone()));
                    get_types.push(get_type);
                    set_types.push(set_type);
                }
                Some((self.unions(get_types), self.unions(set_types)))
            }
            _ => None,
        }
    }

    fn get_django_field_types_from_class(
        &self,
        field: &Class,
        class: &Class,
        field_name: Option<&Name>,
        initial_value_expr: Option<&Expr>,
    ) -> Option<(Type, Type)> {
        if !(self.get_metadata_for_class(class).is_django_model()
            && self.inherits_from_django_field(field))
        {
            return None;
        }

        let base_type = if field_name.is_some()
            && let Some(e) = initial_value_expr
            && let Some(call_expr) = e.as_call_expr()
            && let Some(to_expr) = call_expr.arguments.find_argument_value("to", 0)
            && let Some(model_type) = self.resolve_target(to_expr, class)
        {
            if self.is_foreign_key_like_field(field) {
                Some((model_type.clone(), model_type))
            } else if self.is_many_to_many_field(field) {
                return self.get_manager_type(model_type).map(|ty| (ty.clone(), ty));
            } else {
                None
            }
        } else {
            None
        };

        let base_type = base_type.or_else(|| {
            let get_type = self.get_class_member(field, &DJANGO_PRIVATE_GET_TYPE)?.ty();
            let set_type = self
                .get_class_member(field, &DJANGO_PRIVATE_SET_TYPE)
                .map(|field| field.ty())
                .unwrap_or_else(|| get_type.clone());
            Some((get_type, set_type))
        })?;

        let (base_get_type, set_type) = base_type;
        let get_type =
            self.narrow_charfield_choices(field, initial_value_expr, base_get_type.clone());
        let set_type = if get_type != base_get_type {
            // Ordinary writes must preserve the finite read type. Expressions retain their
            // declared assignment type because their database result is not known here.
            if let Some(combinable) =
                self.try_get_from_export(ModuleName::django_models_expressions(), COMBINABLE)
            {
                let combinable = self.class_def_to_instance_type(combinable);
                self.union(get_type.clone(), self.intersects(&[set_type, combinable]))
            } else {
                get_type.clone()
            }
        } else {
            set_type
        };

        if let Some(e) = initial_value_expr
            && let Some(call_expr) = e.as_call_expr()
            && self.is_django_field_nullable(call_expr)
        {
            Some((
                self.union(get_type, self.heap.mk_none()),
                self.union(set_type, self.heap.mk_none()),
            ))
        } else {
            Some((get_type, set_type))
        }
    }

    /// Narrow CharField with inline choices to a Literal type.
    /// Only blank=False is supported for now.
    fn narrow_charfield_choices(
        &self,
        field: &Class,
        initial_value_expr: Option<&Expr>,
        base_type: Type,
    ) -> Type {
        if let Some(e) = initial_value_expr
            && let Some(call_expr) = e.as_call_expr()
            && self.is_char_field(field)
            && !self.is_django_field_blank(call_expr)
            && let Some(literal_type) = self.extract_charfield_choices_literal_type(call_expr)
        {
            literal_type
        } else {
            base_type
        }
    }

    /// Check if a class inherits from Django's Field class
    fn inherits_from_django_field(&self, cls: &Class) -> bool {
        self.get_mro_for_class(cls)
            .ancestors(self.stdlib)
            .any(|ancestor| ancestor.is(KnownClass::DjangoField))
    }

    /// Bind a generated manager and its initial queryset for a class-body assignment.
    /// The caller checks that the receiving class is a Django model.
    fn get_django_manager_from_queryset_type(
        &self,
        model: &Class,
        initial_value_expr: &Expr,
    ) -> Option<Type> {
        let call = initial_value_expr.as_call_expr()?;
        let Expr::Attribute(attr) = call.func.as_ref() else {
            return None;
        };
        if attr.attr.id != AS_MANAGER {
            return None;
        }
        let callee = self.expr_infer(&call.func, &self.error_swallower());
        if !callee
            .toplevel_func_metadata()
            .is_some_and(Self::is_django_as_manager)
        {
            return None;
        }
        let receiver = self.expr_infer(&attr.value, &self.error_swallower());
        let (_, mut queryset_type) = self.unwrap_class_object_silently(&receiver)?;
        let Type::ClassType(queryset) = &queryset_type else {
            return None;
        };
        let base = self.django_queryset_base(queryset)?;
        let [original_model, original_row] = base.targs().as_slice() else {
            return None;
        };
        let original_model = original_model.clone();
        let original_row = original_row.clone();
        let model = self.heap.mk_self_type(self.as_class_type_unchecked(model));

        // Only construction fills missing model/row arguments. Explicit method returns
        // never pass through this binding step.
        queryset_type.subst_mut_fn(&mut |q| {
            if [&original_model, &original_row]
                .iter()
                .any(|arg| matches!(arg, Type::Quantified(arg) if arg.as_ref() == q))
            {
                Some(model.clone())
            } else {
                Some(q.as_gradual_type())
            }
        });
        let Type::ClassType(queryset) = queryset_type else {
            unreachable!("substitution preserves the queryset class instance");
        };
        // Derive the base from the substituted arguments, including nested row types.
        let mut base = self
            .django_queryset_base(&queryset)
            .expect("substitution preserves the QuerySet base");
        base.targs_mut().as_mut()[0] = model.clone();
        if original_row == original_model
            || matches!(original_row, Type::Any(_) | Type::Quantified(_))
        {
            base.targs_mut().as_mut()[1] = model.clone();
        }
        let queryset = if queryset.is(KnownClass::DjangoQuerySet) {
            base
        } else {
            queryset.with_django_base(base)
        };
        let Type::ClassDef(manager) = self.try_get_from_export(
            ModuleName::from_str("django.db.models.manager"),
            Name::new_static("Manager"),
        )?
        else {
            return None;
        };
        let manager = self.specialize_nontypeddict_to_classtype(
            manager,
            vec![model],
            TextRange::default(),
            &self.error_swallower(),
        );
        Some(
            self.heap
                .mk_class_type(manager.with_django_queryset(queryset)),
        )
    }

    /// Bind a constructor or `as_manager()` assignment to its receiving model.
    pub(crate) fn get_django_manager_assignment_type(
        &self,
        model: &Class,
        initial_value_expr: &Expr,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if let Some(manager) = self.get_django_manager_from_queryset_type(model, initial_value_expr)
        {
            self.expr_infer(initial_value_expr, errors);
            return Some(manager);
        }
        let call = initial_value_expr.as_call_expr()?;
        let callee = self.expr_infer(&call.func, &self.error_swallower());
        let (_, constructor) = self.unwrap_class_object_silently(&callee)?;
        let Type::ClassType(constructor) = constructor else {
            return None;
        };
        let mut base = iter::once(&constructor)
            .chain(
                self.get_mro_for_class(constructor.class_object())
                    .ancestors(self.stdlib),
            )
            .find(|base| {
                base.is(KnownClass::DjangoManager) || base.is(KnownClass::DjangoBaseManager)
            })?
            .clone();
        let [original_model] = base.targs().as_slice() else {
            return None;
        };
        // Tentative diagnostics stay local until this path accepts the constructor result.
        let inference_errors = self.error_collector();
        let Type::ClassType(mut manager) = self.expr_infer(initial_value_expr, &inference_errors)
        else {
            return None;
        };
        if manager.class_object() != constructor.class_object() {
            return None;
        }
        errors.extend(inference_errors);
        let model = self.heap.mk_self_type(self.as_class_type_unchecked(model));
        // The model parameter can occur at any position in a custom manager's parameters.
        // Preserve other arguments inferred from the constructor.
        let model_parameter = manager.tparams().iter().position(
            |parameter| matches!(original_model, Type::Quantified(q) if q.as_ref() == parameter),
        );
        if let Some(index) = model_parameter {
            manager.targs_mut().as_mut()[index] = model.clone();
        }
        base.targs_mut().as_mut()[0] = model;
        let manager = if manager.class_object() == base.class_object() {
            base
        } else if model_parameter.is_some() {
            manager
        } else {
            // Bare and concretely bound subclasses cannot express rebinding in their arguments.
            manager.with_django_base(base)
        };
        Some(self.heap.mk_class_type(manager))
    }

    /// Return the effective QuerySet base, including a construction-time specialization.
    fn django_queryset_base(&self, cls: &ClassType) -> Option<ClassType> {
        if let Some(DjangoSpecialization::EffectiveBase(base)) = cls.django_specialization() {
            Some(base.clone())
        } else if cls.is(KnownClass::DjangoQuerySet) {
            Some(cls.clone())
        } else {
            self.get_mro_for_class(cls.class_object())
                .ancestors(self.stdlib)
                .find(|base| base.is(KnownClass::DjangoQuerySet))
                .map(|base| base.substitute_with(&cls.substitution(), self))
        }
    }

    /// Identify Django's implementation, including access through a subclass.
    fn is_django_as_manager(metadata: &FuncMetadata) -> bool {
        metadata.kind.to_func_symbol().is_some_and(|symbol| {
            symbol.name == AS_MANAGER
                && symbol
                    .cls
                    .as_ref()
                    .is_some_and(|cls| cls.is(KnownClass::DjangoQuerySet))
        })
    }

    /// Check manager replacements by nominal type, independently of their attribute names.
    pub(crate) fn is_django_manager_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::ClassType(cls) if
            cls.is(KnownClass::DjangoBaseManager)
            || self.get_mro_for_class(cls.class_object()).ancestors(self.stdlib)
                .any(|base| base.is(KnownClass::DjangoBaseManager)))
    }

    /// Resolve generated members through the same path as ordinary instance members.
    pub(crate) fn get_django_instance_attribute(
        &self,
        cls: &ClassType,
        name: &Name,
    ) -> Option<ClassAttribute> {
        match cls.django_specialization()? {
            DjangoSpecialization::Expression { .. } => None,
            DjangoSpecialization::Field {
                get_type,
                set_type,
                override_type,
            } => {
                if name == &dunder::GET {
                    if let Some(field) = self.get_class_member(cls.class_object(), name) {
                        let mut attr =
                            self.as_instance_attribute(name, &field, &Instance::of_class(cls));
                        if let ClassAttribute::ReadWrite { read_type, .. } = &mut attr {
                            read_type.transform_toplevel_callable_signatures(|signature, _| {
                            if matches!(signature.get_positional_param(1), Some(Type::ClassType(instance))
                                if instance.is(KnownClass::DjangoModel)) {
                                signature.ret = get_type.clone();
                            }
                        });
                        }
                        return Some(attr);
                    }
                    let metadata = FuncMetadata::method(self, cls.class_object(), dunder::GET);
                    let signatures = vec1![
                        (self.heap.mk_none(), self.heap.mk_class_type(cls.clone())),
                        (
                            self.heap.mk_class_type(self.stdlib.object().clone()),
                            get_type.clone(),
                        ),
                    ]
                    .mapped(|(instance_type, ret)| {
                        OverloadType::Function(Function {
                            signature: Callable::list(
                                ParamList::new(vec![
                                    self.class_self_param(cls.class_object(), false),
                                    Param::PosOnly(None, instance_type, Required::Required),
                                    Param::PosOnly(
                                        None,
                                        self.heap.mk_any_explicit(),
                                        Required::Required,
                                    ),
                                ]),
                                ret,
                            ),
                            metadata: metadata.clone(),
                        })
                    });
                    return Some(self.as_instance_attribute(
                        name,
                        &ClassField::new_synthesized(self.heap.mk_overload(Overload {
                            signatures,
                            metadata: Box::new(metadata),
                        })),
                        &Instance::of_class(cls),
                    ));
                }
                if name != &dunder::SET {
                    return None;
                }
                let mut metadata = FuncMetadata::method(self, cls.class_object(), dunder::SET);
                metadata.flags.descriptor_setter = Some(Box::new(DescriptorSetterMetadata {
                    read_type: get_type.clone(),
                    override_type: override_type.clone(),
                }));
                let method = self.heap.mk_function(Function {
                    signature: Callable::list(
                        ParamList::new(vec![
                            self.class_self_param(cls.class_object(), false),
                            Param::PosOnly(None, self.heap.mk_any_explicit(), Required::Required),
                            Param::PosOnly(None, set_type.clone(), Required::Required),
                        ]),
                        self.heap.mk_none(),
                    ),
                    metadata,
                });
                Some(self.as_instance_attribute(
                    name,
                    &ClassField::new_synthesized(method),
                    &Instance::of_class(cls),
                ))
            }
            DjangoSpecialization::Queryset(queryset) => {
                let manager_member =
                    self.get_class_member_with_defining_class(cls.class_object(), name);
                if let Some(member) = &manager_member
                    && member.defining_class.is(KnownClass::DjangoBaseManager)
                {
                    if !matches!(name.as_str(), "all" | "get_queryset") {
                        return None;
                    }
                    let mut attribute =
                        self.as_instance_attribute(name, &member.value, &Instance::of_class(cls));
                    // Bind manager-owned methods before inserting the model's Self type.
                    // Only the return changes; parameters and function metadata remain intact.
                    match &mut attribute {
                        ClassAttribute::ReadWrite { read_type: ty, .. }
                        | ClassAttribute::ReadOnly(ty, _) => {
                            ty.transform_toplevel_callable_signatures(|signature, _| {
                                signature.ret = self.heap.mk_class_type(queryset.clone());
                            });
                        }
                        _ => return None,
                    }
                    return Some(attribute);
                }

                let member =
                    self.get_class_member_with_defining_class(queryset.class_object(), name)?;
                if !member.value.is_ordinary_instance_method() && !member.value.is_staticmethod() {
                    return None;
                }
                if manager_member.is_none() {
                    let policy = self
                        .get_class_fields(&member.defining_class)
                        .and_then(|fields| fields.django_queryset_only(name));
                    let queryset_only = member.defining_class.is(KnownClass::DjangoQuerySet)
                        && matches!(name.as_str(), "delete" | "adelete" | "as_manager");
                    if policy == Some(true)
                        || (policy.is_none() && (name.as_str().starts_with('_') || queryset_only))
                    {
                        return None;
                    }
                }
                self.get_instance_attribute(queryset, name)
            }
            DjangoSpecialization::EffectiveBase(base) => {
                let member = self.get_class_member_with_defining_class(cls.class_object(), name)?;
                if !self.has_superclass(base.class_object(), &member.defining_class) {
                    return None;
                }
                let field = self.get_class_member(base.class_object(), name)?;
                Some(self.as_instance_attribute(
                    name,
                    field.as_ref(),
                    &Instance::of_protocol(base, self.heap.mk_class_type(cls.clone())),
                ))
            }
            DjangoSpecialization::TargetPkType(pk_type) => {
                let [target, _] = cls.targs().as_slice() else {
                    return None;
                };
                let member = self.get_class_member(cls.class_object(), name)?;
                let mut attribute =
                    self.as_instance_attribute(name, &member, &Instance::of_class(cls));
                let (ClassAttribute::ReadWrite { read_type: ty, .. }
                | ClassAttribute::ReadOnly(ty, _)) = &mut attribute
                else {
                    return None;
                };
                // The stubs accept a related object by primary key as `_To | int`.
                let int = self.heap.mk_class_type(self.stdlib.int().clone());
                ty.transform_toplevel_callable_signatures(|signature, _| {
                    signature.params.visit_mut(&mut |param_ty| {
                        param_ty.transform_mut(&mut |t| {
                            if let Type::Union(union) = t
                                && union.members.contains(target)
                                && union.members.contains(&int)
                            {
                                let members = union
                                    .members
                                    .iter()
                                    .map(|m| if *m == int { pk_type } else { m }.clone())
                                    .collect();
                                *t = self.unions(members);
                            }
                        })
                    });
                });
                Some(attribute)
            }
        }
    }

    /// Get the ManyRelatedManager class from the django stubs, specialized to the target model.
    /// The manager records the target's primary key type so that methods accepting related
    /// objects by primary key use that type.
    fn get_manager_type(&self, target_model_type: Type) -> Option<Type> {
        let model_class = self.try_get_from_export(ModuleName::django_models(), MODEL)?;
        let model_instance_type = self.class_def_to_instance_type(model_class);
        let pk_type = match &target_model_type {
            Type::ClassType(target) => self
                .get_pk_field_type(target.class_object())
                .map(|(pk_type, _)| pk_type),
            _ => None,
        };
        let manager = self.specialize_manager_type(
            MANYRELATEDMANAGER,
            vec![target_model_type, model_instance_type],
        )?;
        match (manager, pk_type) {
            (Type::ClassType(manager), Some(pk_type)) => Some(
                self.heap
                    .mk_class_type(manager.with_django_target_pk_type(pk_type)),
            ),
            (manager, _) => Some(manager),
        }
    }

    fn get_related_manager_type(&self, target_model_type: Type) -> Option<Type> {
        self.specialize_manager_type(RELATED_MANAGER, vec![target_model_type])
    }

    fn specialize_manager_type(&self, name: Name, type_args: Vec<Type>) -> Option<Type> {
        let manager_class_type =
            self.try_get_from_export(ModuleName::django_models_fields_related_descriptors(), name)?;
        let Type::ClassDef(manager_class) = manager_class_type else {
            return None;
        };
        Some(self.specialize(
            manager_class,
            type_args,
            TextRange::default(),
            &self.error_swallower(),
        ))
    }

    fn resolve_target(&self, to_expr: &Expr, class: &Class) -> Option<Type> {
        match to_expr {
            // Use expr_infer to resolve the model in the current scope.
            Expr::Name(_) | Expr::Attribute(_) => {
                let model_type = self.expr_infer(to_expr, &self.error_swallower());
                Some(self.class_def_to_instance_type(&model_type))
            }
            Expr::StringLiteral(ExprStringLiteral { value, .. }) => {
                if value.to_str() == "self" {
                    Some(self.instantiate(class))
                } else {
                    // Django string references may include an app label, but the imported model is
                    // still looked up by its class name in the current module.
                    let target = value.to_str();
                    let class_name = Name::new(
                        target
                            .rsplit_once('.')
                            .map_or(target, |(_, model_name)| model_name),
                    );
                    let module_name = class.module_name();

                    self.try_get_from_export(module_name, class_name)
                        .map(|model_type| self.class_def_to_instance_type(model_type))
                }
            }
            // we may have to extend this function to handle different kinds of fields in the future
            _ => None,
        }
    }

    fn class_def_to_instance_type(&self, ty: &Type) -> Type {
        if let Type::ClassDef(class) = ty {
            self.instantiate(class)
        } else {
            ty.clone()
        }
    }

    pub fn is_foreign_key_like_field(&self, field: &Class) -> bool {
        field.is(KnownClass::DjangoForeignKey) || field.is(KnownClass::DjangoOneToOneField)
    }

    pub fn is_many_to_many_field(&self, field: &Class) -> bool {
        field.is(KnownClass::DjangoManyToManyField)
    }

    pub fn get_django_enum_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let enum_metadata = metadata.enum_metadata()?;
        if !enum_metadata.is_django {
            return None;
        }

        let enum_members = self.get_enum_members(cls);

        let mut label_types: Vec<Type> = enum_members
            .iter()
            .filter_map(|lit| {
                if let Lit::Enum(lit_enum) = lit
                    && let Type::Tuple(Tuple::Concrete(elements)) = &lit_enum.ty
                    && elements.len() >= 2
                {
                    Some(elements[elements.len() - 1].clone().promote(self.stdlib))
                } else {
                    None
                }
            })
            .collect();

        if label_types.is_empty() || label_types.len() < enum_members.len() {
            // Members without a custom label type have default label type str.
            label_types.push(self.heap.mk_class_type(self.stdlib.str().clone()));
        }

        // Also include the type of __empty__ field if it exists, since it contributes to label types
        let empty_name = Name::new_static("__empty__");
        let has_empty = if let Some(field) = self.get_class_member(cls, &empty_name) {
            label_types.push(field.ty());
            true
        } else {
            false
        };

        let label_type = self.unions(label_types);

        let base_value_attr = self.get_enum_or_instance_attribute(
            &self.as_class_type_unchecked(cls),
            metadata,
            &VALUE_PROP,
        );
        let base_value_type = base_value_attr
            .and_then(|attr| {
                self.resolve_get_class_attr(
                    &VALUE_PROP,
                    attr,
                    TextRange::default(),
                    &self.error_swallower(),
                    None,
                )
                .ok()
            })
            .unwrap_or_else(|| self.heap.mk_any_implicit());

        // if value is optional, make the type optional
        let values_type = if has_empty {
            self.union(base_value_type.clone(), self.heap.mk_none())
        } else {
            base_value_type
        };

        let mut fields = SmallMap::new();

        let field_specs = [
            (
                LABELS,
                self.heap
                    .mk_class_type(self.stdlib.list(label_type.clone())),
            ),
            (LABEL, self.property(cls, LABEL, label_type.clone())),
            (
                VALUES,
                self.heap
                    .mk_class_type(self.stdlib.list(values_type.clone())),
            ),
            (
                CHOICES,
                self.heap.mk_class_type(
                    self.stdlib
                        .list(self.heap.mk_concrete_tuple(vec![values_type, label_type])),
                ),
            ),
        ];

        for (name, ty) in field_specs {
            fields.insert(name, ClassSynthesizedField::new(ty));
        }

        Some(ClassSynthesizedFields::new(fields))
    }

    fn property(&self, cls: &Class, name: Name, ty: Type) -> Type {
        let signature = Callable::list(ParamList::new(vec![self.class_self_param(cls, false)]), ty);
        let mut metadata = FuncMetadata::method(self, cls, name);
        metadata.flags.property_metadata = Some(PropertyMetadata {
            role: PropertyRole::Getter,
            getter: self.heap.mk_any_error(),
            setter: None,
            has_deleter: false,
            descriptor_class: None,
        });
        self.heap.mk_function(Function {
            signature,
            metadata,
        })
    }

    /// Get the primary key field type for a Django model.
    /// Returns a tuple of (pk_type, has_custom_pk) where has_custom_pk indicates
    /// whether the model has a custom primary key field defined.
    fn get_pk_field_type(&self, model: &Class) -> Option<(Type, bool)> {
        let metadata = self.get_metadata_for_class(model);

        if let Some(pk_field_name) = metadata
            .django_model_metadata()
            .and_then(|dm| dm.custom_primary_key_field.as_ref())
        {
            let instance_type = self.heap.mk_class_type(self.as_class_type_unchecked(model));
            let pk_type = self.attr_infer_for_type(
                &instance_type,
                pk_field_name,
                TextRange::default(),
                &self.error_swallower(),
                None,
            );
            // A relational primary key stores the related model's raw key value.
            // Attribute lookup handles cycles between models' primary keys.
            let pk_type = if self
                .get_non_synthesized_class_member(model, pk_field_name)
                .is_some_and(|field| {
                    matches!(field.ty(), Type::ClassType(field)
                        if self.is_foreign_key_like_field(field.class_object()))
                }) {
                self.attr_infer_for_type(
                    &pk_type,
                    &PK,
                    TextRange::default(),
                    &self.error_swallower(),
                    None,
                )
            } else {
                pk_type
            };
            Some((pk_type, true))
        } else {
            // No custom pk, use default AutoField type
            let auto_field_type =
                self.try_get_from_export(ModuleName::django_models_fields(), AUTO_FIELD)?;
            self.get_django_field_types(auto_field_type, model, None, None)
                .map(|(get_type, _)| (get_type, false))
        }
    }

    fn is_django_field_nullable(&self, call_expr: &ExprCall) -> bool {
        has_keyword_true(call_expr, &NULL)
    }

    /// Check if a Django field has `blank=True`.
    fn is_django_field_blank(&self, call_expr: &ExprCall) -> bool {
        has_keyword_true(call_expr, &BLANK)
    }

    /// Check if a Django field is a CharField.
    fn is_char_field(&self, field: &Class) -> bool {
        field.is(KnownClass::DjangoCharField)
    }

    /// Extract a Literal type from CharField choices.
    ///
    /// Only supports inline tuple-of-tuples: choices=(("A", "Label A"), ("B", "Label B"), ...)
    /// Returns None if:
    /// - choices is not found
    /// - format is not the simple inline tuple-of-tuples
    /// - any value is not a string literal
    fn extract_charfield_choices_literal_type(&self, call_expr: &ExprCall) -> Option<Type> {
        let choices_value = find_keyword(call_expr, &CHOICES)?;

        let elements = &choices_value.as_tuple_expr()?.elts;

        let mut choice_literals = Vec::new();
        for element in elements {
            let inner_tuple = element.as_tuple_expr()?;
            let string_lit = inner_tuple.elts.first()?.as_string_literal_expr()?;
            choice_literals.push(Lit::from_string_literal(string_lit)?.to_implicit_type());
        }

        if choice_literals.is_empty() {
            None
        } else {
            Some(self.unions(choice_literals))
        }
    }

    /// Create a get_FOO_display method signature for a field with choices.
    /// The method takes self and returns str.
    fn get_display_method(&self, cls: &Class, method_name: &Name) -> ClassSynthesizedField {
        let params = vec![self.class_self_param(cls, false)];
        let ret = self.heap.mk_class_type(self.stdlib.str().clone());
        ClassSynthesizedField::new(self.heap.mk_function(Function {
            signature: Callable::list(ParamList::new(params), ret),
            metadata: FuncMetadata::method(self, cls, method_name.clone()),
        }))
    }

    /// Returns the primary key type of the related model.
    fn get_foreign_key_id_type(&self, class: &Class, class_field: &ClassField) -> Option<Type> {
        let field_type = class_field.ty();
        if !matches!(&field_type, Type::ClassType(field)
            if self.is_foreign_key_like_field(field.class_object()))
        {
            return None;
        }

        // Get the related model type from the field
        let (ty, _) = self.get_django_field_types(&field_type, class, None, None)?;
        let (related_cls, is_foreign_key_nullable) = match ty {
            Type::Union(f) => {
                // Nullable foreign key: extract the class type from the union
                let cls = f.members.iter().find_map(|variant| match variant {
                    Type::ClassType(cls) => Some(cls.clone()),
                    _ => None,
                })?;
                (cls, true)
            }
            Type::ClassType(cls) => (cls, false),
            _ => return None,
        };

        // Get the pk type from the related model and make it nullable if needed
        let (pk_type, _) = self.get_pk_field_type(related_cls.class_object())?;
        if is_foreign_key_nullable {
            Some(self.union(pk_type, self.heap.mk_none()))
        } else {
            Some(pk_type)
        }
    }

    pub fn get_django_model_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let django_metadata = metadata.django_model_metadata()?;

        let mut fields = SmallMap::new();

        if let Some((pk_type, has_custom_pk)) = self.get_pk_field_type(cls) {
            if !has_custom_pk {
                // No custom pk, so synthesize an id field
                fields.insert(ID, ClassSynthesizedField::new(pk_type.clone()));
            }
            fields.insert(PK, ClassSynthesizedField::new(pk_type));
        }

        // Synthesize `<field_name>_id` fields for ForeignKey and OneToOneField fields.
        // We use field names cached in metadata (detected during binding phase)
        // to avoid triggering type resolution during synthesis, which can cause cycles.
        for field_name in &django_metadata.foreign_key_like_fields {
            if let Some(class_field) = self.get_field_from_current_class_only(cls, field_name)
                && let Some(fk_id_type) = self.get_foreign_key_id_type(cls, class_field)
            {
                let id_field_name = Name::new(format!("{}_id", field_name));
                fields.insert(id_field_name, ClassSynthesizedField::new(fk_id_type));
            }
        }

        // Synthesize `get_<field_name>_display()` methods for fields with choices.
        // Same caching strategy as FK fields above.
        for field_name in &django_metadata.fields_with_choices {
            let method_name = Name::new(format!("get_{}_display", field_name));
            fields.insert(
                method_name.clone(),
                self.get_display_method(cls, &method_name),
            );
        }

        let reverse_relations = self.django_reverse_relations_index();
        if let Some(reverse_fields) = reverse_relations.get(cls) {
            for (name, field) in reverse_fields.fields() {
                fields.insert(name.clone(), field.clone());
            }
        }

        Some(ClassSynthesizedFields::new(fields))
    }

    pub fn solve_django_reverse_relations(
        &self,
        binding: &BindingDjangoRelations,
        _range: TextRange,
        _errors: &ErrorCollector,
    ) -> DjangoReverseRelationIndex {
        let mut per_class = SmallMap::new();

        for candidate_class in &binding.classes {
            let Some(source_class) = &self.get_idx(candidate_class.class_idx).0 else {
                continue;
            };
            if !self.get_metadata_for_class(source_class).is_django_model() {
                continue;
            }

            for field_idx in &candidate_class.fields {
                let binding = self.bindings().get(*field_idx);
                let ClassFieldDefinition::AssignedInBody { value, .. } = &binding.definition else {
                    continue;
                };
                let ExprOrBinding::Expr(expr) = value.as_ref() else {
                    continue;
                };
                let Some(call_expr) = expr.as_call_expr() else {
                    continue;
                };

                let Some(relation_kind) = self.django_relation_kind(expr) else {
                    continue;
                };

                let Some(to_expr) = call_expr.arguments.find_argument_value("to", 0) else {
                    continue;
                };
                let Some(target_type) = self.resolve_target(to_expr, source_class) else {
                    continue;
                };
                let target_class = match &target_type {
                    Type::ClassType(cls_type) => cls_type.class_object(),
                    Type::ClassDef(class_def) => class_def,
                    _ => continue,
                };
                if relation_kind == DjangoRelationKind::ManyToMany
                    && self.is_symmetrical_self_m2m(call_expr, source_class, target_class)
                {
                    continue;
                }
                let Some(related_name) =
                    self.django_related_name(call_expr, source_class, relation_kind)
                else {
                    continue;
                };
                let Some(related_type) =
                    self.django_reverse_field_type(relation_kind, source_class)
                else {
                    continue;
                };

                per_class
                    .entry(*target_class)
                    .or_insert_with(SmallMap::new)
                    .insert(related_name, ClassSynthesizedField::new(related_type));
            }
        }

        let mut reverse_relations = SmallMap::new();
        for (class, fields) in per_class.into_iter_hashed() {
            reverse_relations.insert_hashed(class, ClassSynthesizedFields::new(fields));
        }

        DjangoReverseRelationIndex::new(reverse_relations)
    }

    fn django_relation_kind(&self, expr: &Expr) -> Option<DjangoRelationKind> {
        let ty = self.expr_infer(expr, &self.error_swallower());
        let field_class = match &ty {
            Type::ClassType(cls) => cls.class_object(),
            Type::ClassDef(cls) => cls,
            _ => return None,
        };

        if self.is_one_to_one_field(field_class) {
            Some(DjangoRelationKind::OneToOne)
        } else if self.is_many_to_many_field(field_class) {
            Some(DjangoRelationKind::ManyToMany)
        } else if self.is_foreign_key_like_field(field_class) {
            Some(DjangoRelationKind::ForeignKey)
        } else {
            None
        }
    }

    fn django_reverse_field_type(
        &self,
        relation_kind: DjangoRelationKind,
        source_class: &Class,
    ) -> Option<Type> {
        let source_type = self.instantiate(source_class);
        match relation_kind {
            DjangoRelationKind::ForeignKey => self.get_related_manager_type(source_type),
            DjangoRelationKind::OneToOne => Some(source_type),
            DjangoRelationKind::ManyToMany => self.get_manager_type(source_type),
        }
    }

    fn django_related_name(
        &self,
        call_expr: &ExprCall,
        source_class: &Class,
        relation_kind: DjangoRelationKind,
    ) -> Option<Name> {
        match find_keyword(call_expr, &RELATED_NAME) {
            None | Some(Expr::NoneLiteral(_)) => {
                Some(self.django_default_related_name(source_class, relation_kind))
            }
            Some(Expr::StringLiteral(lit)) => {
                self.format_related_name(lit.value.to_str(), source_class)
            }
            _ => None,
        }
    }

    fn django_default_related_name(
        &self,
        source_class: &Class,
        relation_kind: DjangoRelationKind,
    ) -> Name {
        let mut name = source_class.name(self).as_str().to_lowercase();
        if matches!(
            relation_kind,
            DjangoRelationKind::ForeignKey | DjangoRelationKind::ManyToMany
        ) {
            name.push_str("_set");
        }
        Name::new(name)
    }

    fn format_related_name(&self, raw: &str, source_class: &Class) -> Option<Name> {
        if raw.ends_with('+') {
            return None;
        }

        let class_name = source_class.name(self).as_str().to_lowercase();
        let mut substituted = raw.replace("%(class)s", &class_name);
        if substituted.contains("%(app_label)s") {
            let module_name = source_class.module_name();
            let mut module_parts = module_name.as_str().rsplit('.');
            let mut child = module_parts
                .next()
                .expect("rsplit always yields at least one element");
            let mut app_label = None;
            for parent in module_parts {
                if child == "models" {
                    app_label = Some(parent);
                    break;
                }
                child = parent;
            }
            substituted = substituted.replace("%(app_label)s", &app_label?.to_lowercase());
        }

        // Django rejects a related name that is not a valid identifier. This also discards
        // names with unsubstituted placeholders, which are never valid identifiers.
        if !is_python_identifier(&substituted) {
            return None;
        }

        Some(Name::new(substituted))
    }

    fn is_symmetrical_self_m2m(
        &self,
        call_expr: &ExprCall,
        source_class: &Class,
        target_class: &Class,
    ) -> bool {
        if source_class != target_class {
            return false;
        }
        // Suppress the reverse accessor only when symmetry is statically known.
        match find_keyword(call_expr, &SYMMETRICAL) {
            None => true,
            Some(Expr::NoneLiteral(_)) => true,
            Some(Expr::BooleanLiteral(lit)) => lit.value,
            Some(_) => false,
        }
    }
}
