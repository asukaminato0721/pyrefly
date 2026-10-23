# @generated
import functools
from enum import StrEnum
from pydantic import BaseModel, computed_field, ConfigDict, Field


class Color(StrEnum):
    red = "red"


class Model(BaseModel):
    names: list[str]
    price: float
    color: Color


class InheritedModel(Model):
    count: int = 0


class Configured(BaseModel):
    model_config = ConfigDict(strict=True, validate_by_name=True)
    internal_id: int = Field(alias="id")


class Mixed(BaseModel):
    required: int
    literal_default: int = 7
    field_with_default: int = Field(default=0)
    field_with_none: str | None = Field(default=None)
    items: list[str] = Field(default_factory=list, min_length=0)
    complex_factory: list[int] = Field(default_factory=functools.partial(list, [0]))
    field_bare: int = Field()
    field_ellipsis: int = Field(..., description="req")
    field_ellipsis_only: int = Field(...)


class WithInitFalse(BaseModel):
    id: int
    digest: bytes = Field(default=b"", init=False)


class WithValidationAlias(BaseModel):
    internal_id: int = Field(validation_alias="id")


class WithKwOnly(BaseModel):
    req: int = Field(kw_only=False)
    opt: str = Field(default="z", kw_only=True)


class WithComputed(BaseModel):
    width: int
    height: int

    @computed_field
    @property
    def area(self) -> int: ...


class CustomInit(BaseModel):
    name: str

    def __init__(self, name: str, *, eager: bool = False) -> None: ...
