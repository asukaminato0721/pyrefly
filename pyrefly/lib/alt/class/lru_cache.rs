/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::dunder;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Required;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::function::FuncMetadata;
use pyrefly_types::function::Function;
use pyrefly_types::quantified::AnchorIndex;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedIdentity;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::quantified::QuantifiedOrigin;
use pyrefly_types::type_var::PreInferenceVariance;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::Forall;
use pyrefly_types::types::Overload;
use pyrefly_types::types::OverloadType;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::make_bound_method_helper;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::instance::Instance;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Store the complete callable so that overloads and function type parameters survive caching.
    pub(super) fn add_lru_cache_type_param(&self, name: &Identifier, tparams: &mut TParams) {
        if self.module().name().as_str() == "functools" && name.id == "_lru_cache_wrapper" {
            let callable = Quantified::new(
                QuantifiedIdentity::new(
                    self.module().name(),
                    AnchorIndex::first(name.range),
                    QuantifiedOrigin::synthetic(),
                ),
                Name::new_static("_Callable"),
                QuantifiedKind::TypeVar,
                Some(self.heap.mk_any_implicit()),
                Restriction::Unrestricted,
                PreInferenceVariance::Covariant,
            );
            tparams.extend(&TParams::new(vec![callable]));
        }
    }

    pub(crate) fn get_lru_cache_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        if !cls.has_toplevel_qname("functools", "_lru_cache_wrapper") {
            return None;
        }
        let wrapper = self.heap.mk_class_type(self.as_class_type_unchecked(cls));
        let getter = self.synthesized_method(
            cls,
            dunder::GET,
            vec![
                Param::PosOnly(None, wrapper.clone(), Required::Required),
                Param::PosOnly(None, self.heap.mk_any_implicit(), Required::Required),
                Param::PosOnly(None, self.heap.mk_any_implicit(), Required::Optional(None)),
            ],
            wrapper,
        );
        Some(ClassSynthesizedFields::new(SmallMap::from_iter([(
            dunder::GET,
            ClassSynthesizedField::new(getter),
        )])))
    }

    /// Specialize the wrapper's callable members through ordinary instance and descriptor lookup.
    pub(super) fn get_lru_cache_attribute(&self, instance: &Instance, name: &Name) -> Option<Type> {
        if !instance
            .class
            .has_toplevel_qname("functools", "_lru_cache_wrapper")
        {
            return None;
        }
        let [_, callable] = instance.targs.as_slice() else {
            return None;
        };
        if callable.is_any() {
            return None;
        }
        if name.as_str() == "__wrapped__" {
            return Some(match callable {
                Type::BoundMethod(method) => method.func.clone().as_type(),
                callable => callable.clone(),
            });
        }
        if name == &dunder::CALL {
            return Some(callable.clone());
        }
        if name != &dunder::GET {
            return None;
        }
        let cls = ClassType::new(instance.class.dupe(), instance.targs.clone());
        let wrapper = self.heap.mk_class_type(cls.clone());
        let receiver = Quantified::new(
            QuantifiedIdentity::new(
                instance.class.qname().module_name(),
                AnchorIndex::new(instance.class.range(), 1),
                QuantifiedOrigin::synthetic(),
            ),
            Name::new_static("_Receiver"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Undefined,
        );
        let receiver_ty = receiver.clone().to_type(self.heap);
        let owner = self.heap.mk_type_of(receiver_ty.clone());
        let is_classmethod = callable
            .toplevel_func_metadata()
            .is_some_and(|meta| meta.flags.is_classmethod);
        let bound = make_bound_method_helper(
            self.heap,
            if is_classmethod {
                owner.clone()
            } else {
                receiver_ty.clone()
            },
            callable.clone(),
            &|meta| !meta.flags.is_staticmethod,
        )
        .unwrap_or_else(|ty| ty);
        let mut bound_cls = cls;
        bound_cls.targs_mut().as_mut()[1] = bound;
        let bound_wrapper = self.heap.mk_class_type(bound_cls);
        let metadata = FuncMetadata::method(instance.class, dunder::GET);
        let tparams = Arc::new(TParams::new(vec![receiver]));
        let signature = |obj, ret| {
            OverloadType::Forall(Forall {
                tparams: tparams.dupe(),
                body: Function {
                    signature: Callable::list(
                        ParamList::new(vec![
                            Param::PosOnly(None, obj, Required::Required),
                            Param::PosOnly(None, owner.clone(), Required::Optional(None)),
                        ]),
                        ret,
                    ),
                    metadata: metadata.clone(),
                },
            })
        };
        let unbound = signature(
            self.heap.mk_none(),
            if is_classmethod {
                bound_wrapper.clone()
            } else {
                wrapper
            },
        );
        let bound = signature(receiver_ty, bound_wrapper);
        Some(self.heap.mk_overload(Overload {
            signatures: vec1![unbound, bound],
            metadata: Box::new(metadata),
        }))
    }
}
