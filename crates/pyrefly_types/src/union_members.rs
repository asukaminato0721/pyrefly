/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::ops::DerefMut;
use std::slice;
use std::sync::Arc;
use std::vec;

use pyrefly_derive::TypeEq;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;

use crate::identity::IdentityIgnored;
use crate::types::Type;
use crate::types::UnionDisplay;

/// Shares one authoritative member list and keeps its display coverage valid.
/// Raw mutable access invalidates coverage because it can reorder or remove members.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, TypeEq)]
pub struct UnionMembers {
    types: Arc<Vec<Type>>,
    display: IdentityIgnored<Option<UnionDisplay>>,
}

impl UnionMembers {
    pub(crate) fn display(&self) -> Option<&UnionDisplay> {
        self.display.as_ref()
    }

    pub(crate) fn take_display(&mut self) -> Option<UnionDisplay> {
        self.display.0.take()
    }

    pub(crate) fn set_display(&mut self, display: UnionDisplay) {
        assert!(
            display.matches_len(self.types.len()),
            "coverage must match the members"
        );
        self.display.0 = Some(display);
    }

    pub(crate) fn into_parts(self) -> (Vec<Type>, Option<UnionDisplay>) {
        (Arc::unwrap_or_clone(self.types), self.display.0)
    }

    /// Returns owned members, without copying when this is the only owner.
    pub fn into_vec(self) -> Vec<Type> {
        Arc::unwrap_or_clone(self.types)
    }
}

impl Default for UnionMembers {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl From<Vec<Type>> for UnionMembers {
    fn from(members: Vec<Type>) -> Self {
        Self {
            types: Arc::new(members),
            display: IdentityIgnored(None),
        }
    }
}

impl Deref for UnionMembers {
    type Target = Vec<Type>;

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl DerefMut for UnionMembers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.display.0 = None;
        Arc::make_mut(&mut self.types)
    }
}

impl IntoIterator for UnionMembers {
    type Item = Type;
    type IntoIter = vec::IntoIter<Type>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a> IntoIterator for &'a UnionMembers {
    type Item = &'a Type;
    type IntoIter = slice::Iter<'a, Type>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut UnionMembers {
    type Item = &'a mut Type;
    type IntoIter = slice::IterMut<'a, Type>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl Visit<Type> for UnionMembers {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.types.visit(f);
    }
}

impl VisitMut<Type> for UnionMembers {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        // Element-wise visits cannot reorder members. Partial coverage is valid only
        // while every covered member stays equal. The snapshot shares storage,
        // including nested unions, rather than owning another recursive display tree.
        let before = self
            .display()
            .filter(|display| !display.is_full())
            .map(|_| Arc::clone(&self.types));
        Arc::make_mut(&mut self.types).visit_mut(f);
        if let Some(before) = before
            && let Some(display) = self.display()
            && before
                .iter()
                .zip(self.types.iter())
                .enumerate()
                .any(|(index, (before, after))| display.covers(index) && before != after)
        {
            self.display.0 = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyrefly_python::module_name::ModuleName;
    use pyrefly_util::visit::VisitMut;
    use ruff_python_ast::name::Name;

    use crate::types::Type;
    use crate::types::UnionDisplay;
    use crate::union_members::UnionMembers;

    #[test]
    fn union_members_copy_on_write() {
        let original = UnionMembers::from(vec![Type::None]);
        let mut changed = original.clone();
        assert!(Arc::ptr_eq(&original.types, &changed.types));
        changed.push(Type::Ellipsis);
        assert!(!Arc::ptr_eq(&original.types, &changed.types));
        assert_eq!(original.as_slice(), &[Type::None]);
        assert_eq!(changed.into_vec(), vec![Type::None, Type::Ellipsis]);
    }

    #[test]
    fn union_members_raw_mutation_invalidates_coverage() {
        let mut members =
            UnionMembers::from(vec![Type::None, Type::Ellipsis, Type::any_explicit()]);
        members.set_display(UnionDisplay::new(
            (ModuleName::unknown(), Name::new_static("Alias")),
            vec![true, true, false].into_boxed_slice(),
        ));
        let mut changed = members.clone();
        changed.reverse();
        assert!(changed.display().is_none());
        assert!(members.display().is_some());
    }

    #[test]
    fn union_members_visits_validate_coverage() {
        let mut members =
            UnionMembers::from(vec![Type::None, Type::Ellipsis, Type::any_explicit()]);
        members.set_display(UnionDisplay::new(
            (ModuleName::unknown(), Name::new_static("Alias")),
            vec![true, true, false].into_boxed_slice(),
        ));
        members.visit_mut(&mut |ty| {
            if matches!(ty, Type::Any(_)) {
                *ty = Type::any_implicit();
            }
        });
        assert!(
            members.display().is_some(),
            "changing an extra preserves coverage"
        );
        members.visit_mut(&mut |ty| {
            if matches!(ty, Type::None) {
                *ty = Type::any_implicit();
            }
        });
        assert!(
            members.display().is_none(),
            "changing an alias member drops coverage"
        );
    }
}
