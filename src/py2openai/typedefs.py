from __future__ import annotations

import inspect
from typing import Any, Literal, NotRequired

from typing_extensions import TypedDict


class PropertyBase(TypedDict, total=False):
    """Base schema property with common fields."""

    type: str
    description: str
    format: str
    default: Any


class SimpleProperty(PropertyBase):
    """Schema property for primitive types."""

    type: Literal["string", "number", "integer", "boolean"]
    enum: NotRequired[list[Any]]


class ArrayProperty(PropertyBase):
    """Schema property for array types."""

    type: Literal["array"]
    items: Property


class ObjectProperty(PropertyBase):
    """Schema property for nested object types."""

    type: Literal["object"]
    properties: NotRequired[dict[str, Property]]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool]


Property = ArrayProperty | ObjectProperty | SimpleProperty


class ToolParameters(TypedDict):
    """Schema for function parameters."""

    type: Literal["object"]
    properties: dict[str, Property]
    required: NotRequired[list[str]]


class ToolSchema(TypedDict):
    """Complete schema for an OpenAI function call."""

    name: str
    description: str
    parameters: ToolParameters


def _create_simple_property(
    type_str: Literal["string", "number", "integer", "boolean"],
    description: str | None = None,
    enum_values: list[Any] | None = None,
    default: Any = None,
    fmt: str | None = None,
) -> SimpleProperty:
    """Create a simple property."""
    prop: SimpleProperty = {"type": type_str}
    if description is not None:
        prop["description"] = description
    if enum_values is not None:
        prop["enum"] = enum_values
    if default is not inspect.Parameter.empty and default is not None:
        prop["default"] = default
    if fmt is not None:
        prop["format"] = fmt
    return prop


def _create_array_property(
    items: Property,
    description: str | None = None,
) -> ArrayProperty:
    """Create an array property."""
    prop: ArrayProperty = {
        "type": "array",
        "items": items,
    }
    if description is not None:
        prop["description"] = description
    return prop


def _create_object_property(
    description: str | None = None,
    properties: dict[str, Property] | None = None,
    required: list[str] | None = None,
    additional_properties: bool | None = None,
) -> ObjectProperty:
    """Create an object property."""
    prop: ObjectProperty = {"type": "object"}
    if description is not None:
        prop["description"] = description
    if properties is not None:
        prop["properties"] = properties
    if required is not None:
        prop["required"] = required
    if additional_properties is not None:
        prop["additionalProperties"] = additional_properties
    return prop
