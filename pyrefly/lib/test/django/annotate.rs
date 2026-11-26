/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_annotate_adds_keyword_attributes_to_rows,
    r#"
from typing import assert_type

from django.db import models
from django.db.models.query import QuerySet

class Article(models.Model):
    title = models.CharField(max_length=100)

annotated = Article.objects.annotate(extra_title=models.F("title"))
assert_type(annotated, QuerySet[Article, Article])
assert_type(annotated.create(title="plain"), Article)

article = annotated[0]
assert_type(article, Article)
assert_type(article.title, str)
assert_type(article.extra_title, str)
Article().extra_title  # E: Object of class `Article` has no attribute `extra_title`

filtered = annotated.filter(title="kept")[0]
assert_type(filtered, Article)
assert_type(filtered.extra_title, str)

for chained in annotated.annotate(second_title=models.F("title")):
    assert_type(chained, Article)
    assert_type(chained.extra_title, str)
    assert_type(chained.second_title, str)

for values_row in Article.objects.values("title").annotate(extra_title=models.F("title")):
    values_row.extra_title  # E: Object of class `dict` has no attribute `extra_title`
"#,
);

django_testcase!(
    test_annotate_expression_types,
    r#"
from decimal import Decimal
from typing import Any, assert_type

from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name="books")
    price = models.DecimalField(max_digits=8, decimal_places=2)

has_books = models.Exists(Book.objects.filter(author=models.OuterRef("pk")))
total = models.Sum("books__price")
total_int = models.Sum("books__price", output_field=models.IntegerField())

for author in Author.objects.annotate(
    n_books=models.Count("books"),
    has_books=has_books,
    total=total,
    total_int=total_int,
    name_copy=models.F("name"),
    constant=models.Value(1),
    unknown=models.Func("name"),
):
    assert_type(author.n_books, int)
    assert_type(author.has_books, bool)
    assert_type(author.total, Decimal | None)
    assert_type(author.total_int, int | None)
    assert_type(author.name_copy, str)
    assert_type(author.constant, int)
    assert_type(author.unknown, Any)
    author.n_books + "x"  # E: `+` is not supported between `int` and `Literal['x']`

for book in Book.objects.annotate(
    author_key=models.F("author"),
    price_copy=models.F("price"),
    total=models.Sum(models.F("price")),
    total_default=models.Sum("price", default=Decimal(0)),
    total_output=models.Sum("price", output_field=models.IntegerField(), default=0),
    text=models.Value(1, output_field=models.CharField()),
):
    assert_type(book.author_key, int)
    assert_type(book.price_copy, Decimal)
    assert_type(book.total, Decimal | None)
    assert_type(book.total_default, Decimal)
    assert_type(book.total_output, int)
    assert_type(book.text, str)
"#,
);

django_testcase!(
    test_annotate_output_fields_and_unknown_expressions,
    r#"
from typing import Any, assert_type

from django.db import models
from django.db.models.functions import Cast, Length

class Article(models.Model):
    title = models.CharField(max_length=100)
    number = models.IntegerField(null=True)

def count() -> models.Count:
    return models.Count("pk")

for article in Article.objects.annotate(
    length=Length("title"),
    text=Cast("pk", models.CharField()),
    wrapped=models.ExpressionWrapper(models.F("pk"), models.IntegerField()),
    n=count(),
    smallest=models.Min("number"),
    largest=models.Max("number", default=0),
    nullable=models.F(name="number"),
    literal=models.Value(value="hello"),
    unsupported=models.Value(object()),
    unresolved=models.F("missing"),
    nullable_default=models.Sum("number", default=models.Value(None)),
):
    assert_type(article.length, int)
    assert_type(article.text, str)
    assert_type(article.wrapped, int)
    assert_type(article.n, int)
    assert_type(article.smallest, int | None)
    assert_type(article.largest, int)
    assert_type(article.nullable, int | None)
    assert_type(article.literal, str)
    assert_type(article.unsupported, Any)
    assert_type(article.unresolved, Any)
    assert_type(article.nullable_default, int | None)
"#,
);
