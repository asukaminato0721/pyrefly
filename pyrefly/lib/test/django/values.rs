/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_django_values_rows,
    r#"
from decimal import Decimal
from typing import TypedDict, assert_type
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=8, decimal_places=2)

BookRow = TypedDict("BookRow", {"author__name": str, "price": Decimal})

def rows() -> models.QuerySet[Book, BookRow]:
    return Book.objects.values("author__name", "price")

row = rows().get()
assert_type(row["author__name"], str)
assert_type(row["price"], Decimal)
projected = Book.objects.values("author__name", "price").filter(price__gt=0)
assert_type(projected.get()["author__name"], str)
assert_type(projected[0]["price"], Decimal)
for item in projected.iterator():
    assert_type(item["price"], Decimal)
assert_type(projected.create(price=1, author_id=1), Book)
projected.get()["missing"]  # E: TypedDict key `missing` does not exist
"#,
);

django_testcase!(
    test_django_values_list_rows,
    r#"
from decimal import Decimal
from typing import Any, assert_type
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    count = models.IntegerField()

assert_type(Book.objects.values_list("author__name", "price").first(), tuple[str, Decimal] | None)
assert_type(Book.objects.all().values_list("price", flat=True).first(), Decimal | None)
assert_type(Book.objects.values_list("price").get(), tuple[Decimal])
assert_type(Book.objects.values_list("id", flat=True).get(), int)
assert_type(Book.objects.values_list("pk", flat=True).get(), int)
assert_type(Book.objects.values_list("author", flat=True).get(), int)
assert_type(Book.objects.values_list("author_id", flat=True).get(), int)
named = Book.objects.values_list("author__name", "price", "count", named=True).get()
assert_type(named.author__name, str)
assert_type(named.price, Decimal)
assert_type(named.count, int)
assert_type(named[0], str)
assert_type(named[1], Decimal)
assert_type(named._fields, tuple[str, ...])
assert_type(named._asdict(), dict[str, Any])
author, price, count = named
assert_type(author, str)
assert_type(price, Decimal)
assert_type(count, int)
"#,
);

django_testcase!(
    test_django_values_annotated_rows,
    r#"
from decimal import Decimal
from typing import Any, assert_type
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=8, decimal_places=2)

grouped = Book.objects.values("author__name").annotate(total=models.Sum("price"))
assert_type(grouped.get()["author__name"], str)
assert_type(grouped.get()["total"], Decimal | None)
assert_type(grouped.annotate(n=models.Count("pk")).get()["n"], int)
assert_type(grouped.values_list("total", flat=True).get(), Decimal | None)
assert_type(Book.objects.annotate(total=models.Sum("price")).values("total").get()["total"], Decimal | None)
assert_type(Book.objects.values(name=models.F("author__name")).get()["name"], str)
assert_type(Book.objects.values("price", n=models.Value(1)).get()["n"], int)
assert_type(Book.objects.values("price").annotate(total=models.Sum("price", output_field=models.IntegerField())).get()["total"], int | None)
assert_type(Book.objects.values("price").annotate(total=models.Sum("price", default=0)).get()["total"], Decimal)
assert_type(Book.objects.values("price").annotate(unknown=object()).get()["unknown"], Any)
"#,
);

django_testcase!(
    test_django_values_fallbacks,
    r#"
from typing import Any, assert_type
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)

def dynamic(field: str, fields: list[str], flag: bool, expressions: dict[str, Any]) -> None:
    assert_type(Book.objects.values(field).get(), dict[str, Any])
    assert_type(Book.objects.values(*fields).get(), dict[str, Any])
    assert_type(Book.objects.values("title", **expressions).get(), dict[str, Any])
    assert_type(Book.objects.values_list(field).get(), Any)
    assert_type(Book.objects.values_list("title", flat=flag).get(), Any)
    assert_type(Book.objects.values_list("title", named=flag).get(), Any)

assert_type(Book.objects.values().get(), dict[str, Any])
assert_type(Book.objects.values("unresolved").get(), dict[str, Any])
assert_type(Book.objects.values_list().get(), Any)
assert_type(Book.objects.values_list("unresolved", flat=True).get(), Any)
assert_type(Book.objects.values_list("title", "title", named=True).get(), Any)
assert_type(Book.objects.values_list("title", flat=True, named=True).get(), Any)

class CustomQuerySet(models.QuerySet[Book]):
    def values_list(self, *fields: str | models.expressions.Combinable, flat: bool = False, named: bool = False) -> models.QuerySet[Book, Any]:
        return models.QuerySet[Book, Any]()

assert_type(CustomQuerySet().values_list("title", flat=True).get(), Any)
"#,
);
