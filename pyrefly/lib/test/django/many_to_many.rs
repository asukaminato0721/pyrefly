/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

// https://github.com/meta/pyrefly/issues/5448
django_testcase!(
    test_many_to_many_uuid_primary_key,
    r#"
from uuid import UUID, uuid4
from django.db import models

class Tag(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4)

class Article(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4)
    tags = models.ManyToManyField(Tag, related_name="articles")

tag = Tag()
article = Article()
tag_id: UUID = tag.id
article.tags.add(tag_id, tag)
article.tags.remove(tag_id, tag)
article.tags.set([tag.id, tag])
article.tags.set(Tag.objects.all(), clear=True, through_defaults={})
tag.articles.add(article.id, article)
tag.articles.set([article.id, article])
add_tag = article.tags.add
add_tag(tag_id)

async def update() -> None:
    await article.tags.aadd(tag_id, tag)
    await article.tags.aremove(tag_id, tag)
    await article.tags.aset([tag.id, tag])

article.tags.add("invalid")  # E: is not assignable to parameter `*objs`
article.tags.add(article)  # E: is not assignable to parameter `*objs`
article.tags.add(1)  # E: is not assignable to parameter `*objs`
article.tags.set([1])  # E: is not assignable to parameter `objs`
article.tags.set([tag_id], clear="invalid")  # E: is not assignable to parameter `clear`
"#,
);

django_testcase!(
    test_many_to_many_primary_key_types,
    r#"
from typing import assert_type
from uuid import UUID
from django.db import models
from django.db.models.fields.related_descriptors import ManyRelatedManager

class BaseTag(models.Model):
    key = models.CharField(max_length=50, primary_key=True)

    class Meta:
        abstract = True

class Tag(BaseTag):
    pass

class Article(models.Model):
    tags = models.ManyToManyField("Tag", related_name="articles")

tag = Tag()
article = Article()
assert_type(article.tags.add(tag.key, tag), None)
article.tags.remove("news")
article.tags.set(["news", tag])
tag.articles.add(article.pk, article)
tag.articles.remove(1)
tag.articles.set([1, article])
tag.articles.set(Article.objects.all())

def update(manager: ManyRelatedManager[Tag, models.Model]) -> None:
    manager.add("news")
    manager.set(["news"])

article.tags.add(1)  # E: is not assignable to parameter `*objs`
article.tags.set([1])  # E: is not assignable to parameter `objs`
tag.articles.add("invalid")  # E: is not assignable to parameter `*objs`
tag.articles.add(UUID(int=0))  # E: is not assignable to parameter `*objs`

async def update_async() -> None:
    assert_type(await article.tags.aadd("news"), None)
    await article.tags.aremove("news")
    await article.tags.aset(["news", tag])
    await tag.articles.aadd(1)
    await tag.articles.aremove(1)
    await tag.articles.aset([1, article])
    await article.tags.aadd(1)  # E: is not assignable to parameter `*objs`
    await article.tags.aremove(1)  # E: is not assignable to parameter `*objs`
    await article.tags.aset([1])  # E: is not assignable to parameter `objs`
"#,
);

django_testcase!(
    test_many_to_many_keyword_target,
    r#"
from typing import assert_type
from django.db import models
from django.db.models.fields.related_descriptors import ManyRelatedManager

class Author(models.Model): ...

class Book(models.Model):
    authors = models.ManyToManyField(to="Author", related_name="books")

assert_type(Book().authors, ManyRelatedManager[Author, models.Model])
assert_type(Author().books, ManyRelatedManager[Book, models.Model])
"#,
);

django_testcase!(
    test_basic,
    r#"
from django.db.models.query import QuerySet
from django.db.models.fields.related_descriptors import ManyRelatedManager
from typing import assert_type
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField(Author, related_name='books')

book = Book()
assert_type(book.authors, ManyRelatedManager[Author, models.Model])
assert_type(book.authors.all(), QuerySet[Author, Author]) 

assert_type(book.authors.filter(name="Bob"), QuerySet[Author, Author])
assert_type(book.authors.create(name="Alice"), Author) 

book.authors.add("wrong type") # E: Argument `Literal['wrong type']` is not assignable to parameter `*objs` with type `Author | int` 
"#,
);
