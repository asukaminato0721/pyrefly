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
from django.db.models.functions import Cast, Coalesce, Concat, Length

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
    unresolved=models.F("missing"),  # E: Django field path `missing` has no field `missing` on model `Article`
    nullable_default=models.Sum("number", default=models.Value(None)),
    coalesced=Coalesce("number", models.Value(0)),
    concatenated=Concat("number", models.Value(""), output_field=models.CharField()),
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
    assert_type(article.coalesced, int)
    assert_type(article.concatenated, str)
"#,
);

django_testcase!(
    test_annotate_sql_null_and_numeric_aggregates,
    r#"
from decimal import Decimal
from typing import assert_type
from django.db import models
from django.db.models.functions import Cast, Length

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=100)
    subtitle = models.CharField(max_length=100, null=True)
    author = models.ForeignKey(Author, on_delete=models.CASCADE, null=True, related_name="books")
    pages = models.IntegerField()
    in_print = models.BooleanField()
    price = models.DecimalField(max_digits=8, decimal_places=2)

length = Length("subtitle")
for book in Book.objects.annotate(
    length=length,
    title_length=Length("title"),
    cast_subtitle=Cast("subtitle", models.IntegerField()),
    author_length=Length("author__name"),
    in_print_count=models.Sum("in_print"),
    in_print_default=models.Sum("in_print", default=0),
    mean_pages=models.Avg("pages"),
    mean_price=models.Avg("price"),
    mean_default=models.Avg("pages", default=0),
    mean_output=models.Avg("pages", output_field=models.DecimalField()),
    case=models.Case(models.When(in_print=True, then=models.Value(1))),
    case_default=models.Case(models.When(in_print=True, then=models.Value(1)), default=0),
    case_nullable=models.Case(models.When(in_print=True, then="subtitle"), default=models.Value("fallback")),
    case_output=models.Case(models.When(in_print=True, then=models.Value(1)), output_field=models.IntegerField()),
):
    assert_type(book.length, int | None)
    assert_type(book.title_length, int)
    assert_type(book.cast_subtitle, int | None)
    assert_type(book.author_length, int | None)
    assert_type(book.in_print_count, int | None)
    assert_type(book.in_print_default, int)
    assert_type(book.mean_pages, float | None)
    assert_type(book.mean_price, Decimal | None)
    assert_type(book.mean_default, float)
    assert_type(book.mean_output, Decimal | None)
    assert_type(book.case, int | None)
    assert_type(book.case_default, int)
    assert_type(book.case_nullable, str | None)
    assert_type(book.case_output, int | None)

for author in Author.objects.annotate(length=Length("books__title")):
    assert_type(author.length, int | None)
"#,
);

django_testcase!(
    test_annotate_missing_field_paths,
    r#"
from typing import Any, assert_type
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)
    pages = models.IntegerField()

missing = models.F("missing")
Book.objects.annotate(bad=missing)  # E: Django field path `missing` has no field `missing` on model `Book`
Book.objects.annotate(bad=models.Sum("missing"))  # E: Django field path `missing` has no field `missing` on model `Book`
Book.objects.annotate(bad=models.Sum("missing", output_field=models.IntegerField()))  # E: Django field path `missing` has no field `missing` on model `Book`
Book.objects.annotate(bad=models.F("missing"))  # pyrefly: ignore[missing-attribute-django-field]
Book.objects.annotate(bad=models.F("missing"))  # pyrefly: ignore[missing-attribute]

for book in Book.objects.annotate(n=models.Count("*")).annotate(copy=models.F("n")):
    assert_type(book.copy, int)
for book in Book.objects.annotate(n=models.Count("*"), copy=models.F("n")):
    assert_type(book.copy, int)
for book in Book.objects.annotate(transformed=models.F("title__custom_transform")):
    assert_type(book.transformed, Any)
"#,
);
