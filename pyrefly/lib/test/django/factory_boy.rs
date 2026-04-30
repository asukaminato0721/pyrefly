/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::factory_boy_testcase;

factory_boy_testcase!(
    bug = "https://github.com/facebook/pyrefly/issues/3214",
    test_create_returns_model,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:  # E: Class member `UserFactory.Meta` overrides parent class `DjangoModelFactory` in an inconsistent manner
        model = User

    username = "testuser"

user = UserFactory.create()
assert_type(user, User)  # E: assert_type(Unknown, User) failed
"#,
);

factory_boy_testcase!(
    bug = "https://github.com/facebook/pyrefly/issues/3214",
    test_build_returns_model,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:  # E: Class member `UserFactory.Meta` overrides parent class `DjangoModelFactory` in an inconsistent manner
        model = User

user = UserFactory.build()
assert_type(user, User)  # E: assert_type(Unknown, User) failed
"#,
);

factory_boy_testcase!(
    bug = "https://github.com/facebook/pyrefly/issues/3214",
    test_create_batch_returns_list,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:  # E: Class member `UserFactory.Meta` overrides parent class `DjangoModelFactory` in an inconsistent manner
        model = User

users = UserFactory.create_batch(3)
assert_type(users, list[User])  # E: assert_type(list[Unknown], list[User]) failed
"#,
);

factory_boy_testcase!(
    bug = "https://github.com/facebook/pyrefly/issues/3214",
    test_model_attribute_access,
    r#"
from django.db import models
from factory.django import DjangoModelFactory

class Document(models.Model):
    title = models.CharField(max_length=200)

class DocumentFactory(DjangoModelFactory):
    class Meta:  # E: Class member `DocumentFactory.Meta` overrides parent class `DjangoModelFactory` in an inconsistent manner
        model = Document

doc = DocumentFactory.create()
title = doc.title
"#,
);
