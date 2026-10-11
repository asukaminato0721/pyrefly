/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solving support for `functools.singledispatch` and `functools.singledispatchmethod`.

use std::fmt;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::types::BoundMethodType;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::BindingSingleDispatchRegistry;
use crate::binding::binding::KeySingleDispatchRegistry;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::callable::Callable;
use crate::types::callable::Param;
use crate::types::callable::Params;
use crate::types::function::FuncFlags;
use crate::types::function::FuncMetadata;
use crate::types::function::Function;
use crate::types::function::FunctionKind;
use crate::types::keywords::KwCall;
use crate::types::keywords::TypeMap;
use crate::types::types::Forallable;
use crate::types::types::Type;

/// Definition-site facts about a decorated function, used to validate it when it is a
/// `@singledispatch` dispatcher.
pub(crate) struct DispatcherDef<'a> {
    pub params: &'a [Param],
    pub id_range: TextRange,
    pub defining_cls: Option<&'a Class>,
    pub is_staticmethod: bool,
}

/// Resolved registrations for method dispatchers referenced in a module.
#[derive(Clone, Debug, Default, PartialEq, Eq, TypeEq, Visit, VisitMut)]
pub struct SingleDispatchRegistry {
    /// Each entry contains the dispatcher identity, dispatch type, and implementation.
    registrations: Vec<(FunctionKind, Type, Type)>,
}

impl fmt::Display for SingleDispatchRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SingleDispatchRegistry(len={})",
            self.registrations.len()
        )
    }
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Resolve registration candidates once, with normal incremental dependencies.
    pub(crate) fn solve_single_dispatch_registry(
        &self,
        binding: &BindingSingleDispatchRegistry,
    ) -> SingleDispatchRegistry {
        let mut registrations = Vec::new();
        for (decorator_idx, function_idx) in &binding.registrations {
            let decorator = &self.get_idx(*decorator_idx).ty;
            let Some(FuncMetadata {
                kind: FunctionKind::SingleDispatchMethodRegister(target, explicit),
                ..
            }) = decorator.toplevel_func_metadata()
            else {
                continue;
            };
            let target = target
                .toplevel_func_metadata()
                .expect("Registration retains dispatcher metadata");
            let implementation = self.get_idx(*function_idx).clone();
            let is_static = implementation
                .toplevel_func_metadata()
                .is_some_and(|m| m.flags.is_staticmethod);
            let dispatch_type = explicit.as_deref().cloned().or_else(|| {
                let (signature, _) = implementation.toplevel_callable_signatures().next()?;
                if is_static {
                    Self::first_positional_param_type(signature)
                } else {
                    signature
                        .strip_first_param()
                        .as_ref()
                        .and_then(Self::first_positional_param_type)
                }
            });
            if let Some(dispatch_type) = dispatch_type {
                self.map_over_union(&dispatch_type, |dispatch_type| {
                    registrations.push((
                        target.kind.clone(),
                        dispatch_type.clone(),
                        implementation.clone(),
                    ));
                });
            }
        }
        SingleDispatchRegistry { registrations }
    }

    /// Preserve the dispatcher's identity through both registration decorator forms.
    pub(crate) fn singledispatchmethod_register(
        &self,
        dispatcher: Type,
        dispatch_type: Option<Type>,
    ) -> Type {
        self.heap.mk_function(Function {
            signature: Callable {
                params: Params::Ellipsis,
                ret: Type::any_implicit(),
            },
            metadata: FuncMetadata {
                kind: FunctionKind::SingleDispatchMethodRegister(
                    Box::new(dispatcher),
                    dispatch_type.map(Box::new),
                ),
                flags: FuncFlags::default(),
            },
        })
    }

    /// Select an implementation by the dispatch argument, then check its complete signature.
    pub(crate) fn call_singledispatchmethod(
        &self,
        dispatcher: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let metadata = dispatcher
            .toplevel_func_metadata()
            .expect("A dispatcher has function metadata");
        let local_registry = self.get(&KeySingleDispatchRegistry);
        let source_registry = metadata.kind.class().and_then(|cls| {
            (cls.module_name() != self.module().name() || cls.module_path() != self.module().path())
                .then(|| self.get_from_class(&cls, &KeySingleDispatchRegistry))
                .flatten()
        });
        let registrations = source_registry
            .into_iter()
            .chain([local_registry])
            .flat_map(|registry| &registry.registrations)
            .filter(|(target, _, _)| target == &metadata.kind);
        let Some(CallArg::Arg(first)) = args.first() else {
            return self
                .freeform_call_infer(
                    dispatcher.clone(),
                    args,
                    kws,
                    callee_range,
                    arg_range,
                    hint,
                    errors,
                )
                .ty;
        };
        let arg_ty = first.infer(self, errors);
        let registrations: Vec<_> = registrations.collect();
        self.distribute_over_union(&arg_ty, |arg_ty| {
            if arg_ty.is_any() {
                return arg_ty.clone();
            }
            let runtime_type = self.type_of(arg_ty.clone());
            let runtime_class = match &runtime_type {
                Type::ClassDef(cls) => Some(cls),
                Type::Type(ty) => match &**ty {
                    Type::ClassType(cls) | Type::SelfType(cls) => Some(cls.class_object()),
                    _ => None,
                },
                _ => None,
            };
            let mut selected: Option<(&Type, usize)> = None;
            for (_, dispatch_type, implementation) in &registrations {
                // Dispatch uses the runtime MRO. Assignment compatibility would incorrectly
                // dispatch an `int` to a `float` registration.
                let priority = match (runtime_class, dispatch_type) {
                    (Some(cls), Type::ClassType(dispatch_cls)) => {
                        if cls == dispatch_cls.class_object() {
                            Some(0)
                        } else {
                            self.get_mro_for_class(cls)
                                .ancestors(self.stdlib)
                                .position(|ancestor| {
                                    ancestor.class_object() == dispatch_cls.class_object()
                                })
                                .map(|index| index + 1)
                        }
                    }
                    _ => self
                        .is_subset_eq(arg_ty, dispatch_type)
                        .then_some(usize::MAX),
                };
                if let Some(priority) = priority
                    && selected.is_none_or(|(_, previous)| priority <= previous)
                {
                    selected = Some((implementation, priority));
                }
            }
            let mut callable = selected.map_or_else(
                || dispatcher.clone(),
                |(implementation, _)| implementation.clone(),
            );
            if selected.is_some()
                && let Type::BoundMethod(method) = dispatcher
            {
                let receiver = match &method.obj {
                    Type::Type(cls) => &**cls,
                    receiver => receiver,
                };
                if let Type::ClassType(cls) = receiver
                    && let Some(defining_cls) = callable
                        .toplevel_func_metadata()
                        .and_then(|m| m.kind.class())
                    && let Some(ancestor) = self.as_superclass(cls, &defining_cls)
                {
                    callable = ancestor.targs().substitution().substitute_into(callable);
                }
                callable.subst_self_type_mut(receiver);
            }
            if selected.is_some()
                && let Some(flags) = callable.toplevel_func_metadata().map(|m| &m.flags)
                && !flags.is_staticmethod
                && (matches!(dispatcher, Type::BoundMethod(_)) || flags.is_classmethod)
            {
                callable.transform_toplevel_callable_signatures(|signature, _| {
                    if let Some(bound) = signature.strip_first_param() {
                        *signature = bound;
                    }
                });
            }
            let mut narrowed_args = vec![CallArg::ty(arg_ty, first.range())];
            narrowed_args.extend_from_slice(&args[1..]);
            self.freeform_call_infer(
                callable,
                &narrowed_args,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            )
            .ty
        })
    }

    /// Whether `ct` is a `@singledispatch` dispatcher class (`_SingleDispatchCallable`)
    fn is_singledispatch_class(ct: &ClassType) -> bool {
        ct.has_qname("functools", "_SingleDispatchCallable")
            || ct.has_qname("singledispatch", "_SingleDispatchCallable")
    }

    /// Whether `ty` is a `@singledispatch` dispatcher, as either the raw `_SingleDispatchCallable`
    /// instance or the callback-protocol function it is rewritten into.
    pub(crate) fn is_singledispatch_dispatcher(ty: &Type) -> bool {
        match ty {
            Type::ClassType(ct) => Self::is_singledispatch_class(ct),
            Type::Function(f) => matches!(
                &f.metadata.kind,
                FunctionKind::CallbackProtocol(cls) if Self::is_singledispatch_class(cls)
            ),
            Type::Forall(forall) => matches!(
                &forall.body,
                Forallable::Function(f)
                    if matches!(&f.metadata.kind, FunctionKind::CallbackProtocol(cls) if Self::is_singledispatch_class(cls))
            ),
            _ => false,
        }
    }

    /// Model the dispatcher as a callback protocol over the fallback signature, so calls are checked
    /// against the fallback while `.register`/`.dispatch` and the revealed type stay `_SingleDispatchCallable`.
    pub(crate) fn singledispatch_dispatcher_as_callback(
        &self,
        ty: Type,
        original_decoratee: &Type,
    ) -> Type {
        let Type::ClassType(ct) = &ty else {
            return ty;
        };
        if !Self::is_singledispatch_class(ct) {
            return ty;
        }
        // A generic fallback keeps its type params so the call can bind them (`to_callable` drops
        // the `Forall`, which would otherwise collapse the dispatched return to `Unknown`).
        if let Type::Forall(forall) = original_decoratee {
            let signature = match &forall.body {
                Forallable::Function(f) => f.signature.clone(),
                Forallable::Callable(c) => c.clone(),
                Forallable::TypeAlias(_) => return ty,
            };
            let func = Function {
                signature,
                metadata: FuncMetadata {
                    kind: FunctionKind::CallbackProtocol(Box::new(ct.clone())),
                    flags: FuncFlags::default(),
                },
            };
            return Forallable::Function(func).forall(forall.tparams.clone());
        }
        if let Some(mut signature) = original_decoratee.clone().to_callable() {
            // Use the element type `_T` as the return (normally identical to the fallback return),
            // since `_T` reflects later normalization of the element such as an inferred `Never` -> `Any`.
            if let [ret] = ct.targs().as_slice() {
                signature.ret = ret.clone();
            }
            return self.heap.mk_function(Function {
                signature,
                metadata: FuncMetadata {
                    kind: FunctionKind::CallbackProtocol(Box::new(ct.clone())),
                    flags: FuncFlags::default(),
                },
            });
        }
        ty
    }

    /// For checking a dispatcher call only, widen the dispatch (first) parameter to `Any` so any
    /// dispatched argument is accepted; a parameter mentioning a type variable is left intact.
    pub(crate) fn widen_singledispatch_dispatch_param(&self, mut ty: Type) -> Type {
        let is_dispatcher = ty.toplevel_func_metadata().is_some_and(|metadata| {
            matches!(
                &metadata.kind,
                FunctionKind::CallbackProtocol(cls) if Self::is_singledispatch_class(cls)
            )
        });
        if !is_dispatcher {
            return ty;
        }
        ty.transform_toplevel_callable_signatures(|signature, _| {
            let Params::List(params) = &mut signature.params else {
                return;
            };
            let dispatch_ty = params.items_mut().iter_mut().find_map(|p| match p {
                Param::PosOnly(_, t, _) | Param::Pos(_, t, _) | Param::Varargs(_, t) => Some(t),
                _ => None,
            });
            let Some(dispatch_ty) = dispatch_ty else {
                return;
            };
            let mut mentions_tvar = false;
            dispatch_ty.for_each_quantified(&mut |_| mentions_tvar = true);
            if !mentions_tvar {
                *dispatch_ty = Type::any_implicit();
            }
        });
        ty
    }

    /// Replace a singledispatch dispatcher's `Never` element type with gradual `Any`.
    pub(crate) fn widen_singledispatch_never(&self, mut ty: Type) -> Type {
        if let Type::ClassType(ct) = &mut ty
            && Self::is_singledispatch_class(ct)
            && let [arg] = ct.targs().as_slice()
            && self.solver().force(arg.clone()).is_never()
        {
            ct.targs_mut().as_mut()[0] = Type::any_implicit();
        }
        ty
    }

    /// Dispatch happens on the first positional parameter, so a `singledispatch` fallback must
    /// have one.
    pub(crate) fn validate_singledispatch_dispatcher_signature(
        &self,
        ty: &Type,
        def: DispatcherDef,
        errors: &ErrorCollector,
    ) {
        if !Self::is_singledispatch_dispatcher(ty)
            && !ty
                .toplevel_func_metadata()
                .is_some_and(|m| m.flags.is_singledispatchmethod)
        {
            return;
        }
        let skip_self = def.defining_cls.is_some()
            && !def.is_staticmethod
            && matches!(
                def.params.first(),
                Some(Param::Pos(..) | Param::PosOnly(..))
            );
        let message = match def.params.get(skip_self as usize) {
            None => "Singledispatch function requires at least one parameter",
            Some(Param::KwOnly(..) | Param::Kwargs(..)) => {
                "First parameter of a singledispatch function must be positional"
            }
            Some(_) => return,
        };
        self.error(
            errors,
            def.id_range,
            ErrorKind::BadFunctionDefinition,
            message.to_owned(),
        );
    }

    /// The fallback first-parameter type carried by a tagged `singledispatch` `register` method,
    /// regardless of any bound-method wrapping.
    pub(crate) fn singledispatch_register_first(ty: &Type) -> Option<Type> {
        let kind = match ty {
            Type::Function(f) => &f.metadata.kind,
            Type::Overload(o) => &o.metadata.kind,
            Type::BoundMethod(bm) => match &bm.func {
                BoundMethodType::Function(f) => &f.metadata.kind,
                BoundMethodType::Forall(fa) => &fa.body.metadata.kind,
                BoundMethodType::Overload(o) => &o.metadata.kind,
            },
            _ => return None,
        };
        match kind {
            FunctionKind::SingleDispatchRegister(first) => Some((**first).clone()),
            _ => None,
        }
    }

    /// The dispatch type of a singledispatch signature: its first positional parameter, or the
    /// element type of a leading `*args` (dispatch happens on the first runtime argument either way).
    pub(crate) fn first_positional_param_type(sig: &Callable) -> Option<Type> {
        let Params::List(params) = &sig.params else {
            return None;
        };
        params.items().iter().find_map(|p| match p {
            Param::PosOnly(_, t, _) | Param::Pos(_, t, _) | Param::Varargs(_, t) => Some(t.clone()),
            _ => None,
        })
    }

    /// A registered impl can only be dispatched to if its dispatch type is a subtype of the fallback's
    /// first parameter.
    pub(crate) fn check_singledispatch_register(
        &self,
        dispatch_ty: &Type,
        fallback_first: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if !self.is_subset_eq(dispatch_ty, fallback_first) {
            self.error(
                errors,
                range,
                ErrorKind::BadSingledispatchRegister,
                format!(
                    "Dispatch type `{}` is not a subtype of fallback first argument type `{}`",
                    self.for_display(dispatch_ty.clone()),
                    self.for_display(fallback_first.clone()),
                ),
            );
        }
    }

    /// Accessing `.register` collapses the base to `_SingleDispatchCallable`, losing the fallback's
    /// first parameter; tag the returned method with that type so the dispatch type can be validated.
    pub(crate) fn tag_singledispatch_register(
        &self,
        base: &Type,
        attr_name: &Name,
        mut ty: Type,
    ) -> Type {
        if attr_name.as_str() == "register"
            && let Type::Function(f) = base
            && let FunctionKind::CallbackProtocol(cls) = &f.metadata.kind
            && matches!(
                cls.qname().module_name().as_str(),
                "functools" | "singledispatch"
            )
            && cls.name().as_str() == "_SingleDispatchCallable"
            && let Some(first) = Self::first_positional_param_type(&f.signature)
            && let Some(m) = ty.toplevel_func_metadata_mut()
        {
            m.kind = FunctionKind::SingleDispatchRegister(Box::new(first.clone()));
        }
        ty
    }

    /// Handle a `@fn.register(...)` call: validate the dispatch class against the fallback, then return
    /// the impl's own type (`register(impl)`) or tag the factory form `register(C)` for later application.
    pub(crate) fn call_singledispatch_register(
        &self,
        fallback_first: Type,
        register_ty: &Type,
        arguments: &Arguments,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let (cls_expr, has_func) = singledispatch_register_args(arguments);
        // Infer the first argument as a value, not a type form, so a name colliding with a special
        // form isn't misread: a class object is the dispatch type, a lone callable is the impl.
        let arg_ty = cls_expr.map(|e| self.expr_infer(e, errors));
        let dispatch_class = arg_ty
            .as_ref()
            .and_then(|t| self.unwrap_class_object_silently(t))
            .map(|(_, dispatch_ty)| dispatch_ty);
        if let Some(dispatch_ty) = &dispatch_class {
            self.check_singledispatch_register(dispatch_ty, &fallback_first, callee_range, errors);
        }
        // Bare functional `register(impl)`: the lone argument is the impl and its first parameter (if
        // any) is the dispatch type. Return the impl's own type so direct calls to it are checked.
        if dispatch_class.is_none()
            && arguments.args.len() == 1
            && arguments.keywords.is_empty()
            && let Some(impl_ty) = &arg_ty
            && let Some((sig, _)) = impl_ty.toplevel_callable_signatures().next()
        {
            if let Some(dispatch_ty) = Self::first_positional_param_type(sig) {
                self.check_singledispatch_register(
                    &dispatch_ty,
                    &fallback_first,
                    callee_range,
                    errors,
                );
            }
            return impl_ty.clone();
        }
        let return_ty = self
            .freeform_call_infer(
                register_ty.clone(),
                args,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            )
            .ty;
        if dispatch_class.is_some() && !has_func {
            self.heap.mk_kw_call(KwCall {
                func_metadata: FuncMetadata {
                    kind: FunctionKind::SingleDispatchRegister(Box::new(fallback_first)),
                    flags: FuncFlags::default(),
                },
                keywords: TypeMap::new(),
                return_ty,
            })
        } else {
            return_ty
        }
    }

    /// Applying the factory decorator `f.register(C)` to an impl returns the impl's own type, so
    /// direct calls to it are argument-checked; the dispatch class was validated at `.register(C)`.
    pub(crate) fn apply_singledispatch_register(
        &self,
        register_ty: &Type,
        impl_arg: &Expr,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let impl_ty = self.expr_infer(impl_arg, errors);
        // A non-callable argument is left to normal call checking.
        if impl_ty.toplevel_callable_signatures().next().is_none() {
            self.freeform_call_infer(
                register_ty.clone(),
                args,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            )
            .ty
        } else {
            impl_ty
        }
    }
}

/// A `.register(...)` call's dispatch-class arg (first positional or `cls=`) and whether a `func`
/// is supplied (second positional or `func=`), separating the factory form from the functional one.
fn singledispatch_register_args(arguments: &Arguments) -> (Option<&Expr>, bool) {
    let keyword = |name: &str| {
        arguments
            .keywords
            .iter()
            .find_map(|k| (k.arg.as_ref()?.id.as_str() == name).then_some(&k.value))
    };
    let cls = arguments.args.first().or_else(|| keyword("cls"));
    let has_func = arguments.args.len() >= 2 || keyword("func").is_some();
    (cls, has_func)
}
