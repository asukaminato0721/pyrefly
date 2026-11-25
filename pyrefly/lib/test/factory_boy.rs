/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(test)]
pub mod util;

use crate::factory_boy_testcase;

factory_boy_testcase!(
    test_post_generation_model,
    r#"
from dataclasses import dataclass, field
from typing import Self, assert_type
import factory

@dataclass
class Item:
    owner: str
    tags: list[str] = field(default_factory=list)

class ItemFactory(factory.Factory):
    owner = factory.LazyAttribute(lambda o: "owner")

    class Meta:
        model = Item

    @factory.post_generation
    def tags(self, create, extracted, **kwargs):
        assert_type(self, Item)
        self.tags.append("x")
        self.owner.upper()
        self.tags.append(1)  # E: Argument `Literal[1]` is not assignable to parameter `object` with type `str`

    @factory.post_generation
    def annotated(self: Item, create, extracted, **kwargs):
        assert_type(self, Item)

    def method(self):
        assert_type(self, Self)

assert_type(ItemFactory.tags, factory.PostGeneration)
"#,
);

factory_boy_testcase!(
    test_post_generation_inherited_model,
    r#"
from typing import assert_type
from factory import Factory, post_generation as hook

class Item:
    name: str

class Other:
    value: int

class BaseFactory(Factory):
    class Meta:
        model = Item

class ChildFactory(BaseFactory):
    @hook
    def update(instance, create, extracted, **kwargs):
        assert_type(instance, Item)
        instance.name.upper()

class OtherFactory(BaseFactory):
    class Meta:
        model = Other

    @hook
    def update(self, /, create, extracted, **kwargs):
        assert_type(self, Other)

class InheritedMetaFactory(BaseFactory):
    class Meta(BaseFactory.Meta):
        pass

    @hook
    def update(self, create, extracted, **kwargs):
        assert_type(self, Item)
"#,
);

factory_boy_testcase!(
    test_post_generation_unknown_model,
    r#"
from typing import Any, assert_type
from factory import Factory, post_generation

class MissingMetaFactory(Factory):
    @post_generation
    def update(self, create, extracted, **kwargs):
        assert_type(self, Any)

class MissingModelFactory(Factory):
    class Meta:
        pass

    @post_generation
    def update(self, create, extracted, **kwargs):
        assert_type(self, Any)

class StringModelFactory(Factory):
    class Meta:
        model = "app.Item"

    @post_generation
    def update(self, create, extracted, **kwargs):
        assert_type(self, Any)
"#,
);

factory_boy_testcase!(
    test_post_generation_unrelated_decorator_or_class,
    r#"
from typing import Self, assert_type
import factory

class Item:
    pass

def post_generation[F](fn: F) -> F:
    return fn

class ItemFactory(factory.Factory):
    class Meta:
        model = Item

    @post_generation
    def update(self):
        assert_type(self, Self)

    @post_generation
    def annotated(self: Item):  # E: method self type `Item` is not a superclass of class `ItemFactory`
        pass

class NotFactory:
    class Meta:
        model = Item

    @factory.post_generation
    def update(self, create, extracted, **kwargs):
        assert_type(self, Self)

    @factory.post_generation
    def annotated(self: Item, create, extracted, **kwargs):  # E: method self type `Item` is not a superclass of class `NotFactory`
        pass
"#,
);
