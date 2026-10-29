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

use crate::types::Type;

/// Shares union members until mutation requires an owned copy.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, TypeEq)]
pub struct UnionMembers(Arc<Vec<Type>>);

impl UnionMembers {
    /// Returns owned members, without copying when this is the only owner.
    pub fn into_vec(self) -> Vec<Type> {
        Arc::unwrap_or_clone(self.0)
    }
}

impl From<Vec<Type>> for UnionMembers {
    fn from(members: Vec<Type>) -> Self {
        Self(Arc::new(members))
    }
}

impl Deref for UnionMembers {
    type Target = Vec<Type>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UnionMembers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
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
        self.0.visit(f);
    }
}

impl VisitMut<Type> for UnionMembers {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        Arc::make_mut(&mut self.0).visit_mut(f);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::types::Type;
    use crate::union_members::UnionMembers;

    #[test]
    fn union_members_copy_on_write() {
        let original = UnionMembers::from(vec![Type::None]);
        let mut changed = original.clone();
        assert!(Arc::ptr_eq(&original.0, &changed.0));
        changed.push(Type::Ellipsis);
        assert!(!Arc::ptr_eq(&original.0, &changed.0));
        assert_eq!(original.as_slice(), &[Type::None]);
        assert_eq!(changed.into_vec(), vec![Type::None, Type::Ellipsis]);
    }
}
