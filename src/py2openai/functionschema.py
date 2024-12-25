"""Module for creating OpenAI function schemas from Python functions."""

from __future__ import annotations

from collections.abc import (
    Callable,  # noqa: TC003
    Sequence,  # noqa: F401
)
import dataclasses
from datetime import date, datetime, time, timedelta, timezone
import decimal
import enum
import inspect
import ipaddress
import logging
from pathlib import Path
import re
import types
import typing
from typing import Annotated, Any, NotRequired, Required, TypeGuard
from uuid import UUID

import docstring_parser
import pydantic

from py2openai.typedefs import (
    OpenAIFunctionDefinition,
    OpenAIFunctionTool,
    Property,
    ToolParameters,
)


logger = logging.getLogger(__name__)


class FunctionType(str, enum.Enum):
    """Enum representing different function types."""

    SYNC = "sync"
    ASYNC = "async"
    SYNC_GENERATOR = "sync_generator"
    ASYNC_GENERATOR = "async_generator"


class FunctionSchema(pydantic.BaseModel):
    """Schema representing an OpenAI function definition and metadata.

    This class encapsulates all the necessary information to describe a function to the
    OpenAI API, including its name, description, parameters, return type, and execution
    characteristics. It follows the OpenAI function calling format while adding
    additional metadata useful for Python function handling.

    The schema includes support for complex parameter types, optional values,
    and different function execution patterns (sync, async, generators).
    """

    name: str
    """The name of the function as it will be presented to the OpenAI API."""

    description: str | None = None
    """
    Optional description of what the function does. This helps the AI understand
    when and how to use the function.
    """

    parameters: ToolParameters = pydantic.Field(
        default_factory=lambda: ToolParameters(type="object", properties={}),
    )
    """
    JSON Schema object describing the function's parameters. Contains type information,
    descriptions, and constraints for each parameter.
    """

    required: list[str] = pydantic.Field(default_factory=list)
    """
    List of parameter names that are required (do not have default values).
    These parameters must be provided when calling the function.
    """

    returns: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {"type": "object"},
    )
    """
    JSON Schema object describing the function's return type. Used for type checking
    and documentation purposes.
    """

    function_type: FunctionType = FunctionType.SYNC
    """
    The execution pattern of the function (sync, async, generator, or async generator).
    Used to determine how to properly invoke the function.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    def model_dump_openai(self) -> OpenAIFunctionTool:
        """Convert the schema to OpenAI's function calling format.

        Returns:
            A dictionary matching OpenAI's complete function tool definition format.

        Example:
            ```python
            schema = FunctionSchema(
                name="get_weather",
                description="Get weather information for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["C", "F"]}
                    }
                },
                required=["location"]
            )

            openai_schema = schema.model_dump_openai()
            # Result:
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "get_weather",
            #         "description": "Get weather information for a location",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "location": {"type": "string"},
            #                 "unit": {"type": "string", "enum": ["C", "F"]}
            #             },
            #             "required": ["location"]
            #         }
            #     }
            # }
            ```
        """
        parameters: ToolParameters = {
            "type": "object",
            "properties": self.parameters["properties"],
            "required": self.required,
        }

        # First create the function definition
        function_def = OpenAIFunctionDefinition(
            name=self.name,
            description=self.description or "",
            parameters=parameters,
        )

        # Then wrap it in the tool
        return OpenAIFunctionTool(
            type="function",
            function=function_def,
        )


def _is_optional_type(typ: type) -> TypeGuard[type]:
    """Check if a type is Optional[T] or T | None.

    Args:
        typ: Type to check

    Returns:
        True if the type is Optional, False otherwise
    """
    origin = typing.get_origin(typ)

    # Not a Union/UnionType, can't be Optional
    if origin not in (typing.Union, types.UnionType):  # pyright: ignore
        return False

    args = typing.get_args(typ)
    # Check if any of the union members is None or NoneType
    return any(arg is type(None) for arg in args)


def _resolve_type_annotation(
    typ: Any,
    description: str | None = None,
    default: Any = inspect.Parameter.empty,
    is_parameter: bool = True,
) -> Property:
    """Resolve a type annotation into an OpenAI schema type.

    Args:
        typ: Type to resolve
        description: Optional description
        default: Default value if any
        is_parameter: Whether this is for a parameter (affects dict schema)
    """
    schema: dict[str, Any] = {}

    # Handle Annotated types first
    if typing.get_origin(typ) is Annotated:
        # Get the underlying type (first argument)
        base_type = typing.get_args(typ)[0]
        return _resolve_type_annotation(
            base_type,
            description=description,
            default=default,
            is_parameter=is_parameter,
        )

    origin = typing.get_origin(typ)
    args = typing.get_args(typ)

    # Handle Union types (including Optional)
    if origin in (typing.Union, types.UnionType):  # pyright: ignore
        # For Optional (union with None), filter out None type
        non_none_types = [t for t in args if t is not type(None)]
        if non_none_types:
            # Use the first non-None type
            schema.update(
                _resolve_type_annotation(
                    non_none_types[0],
                    description=description,
                    default=default,
                    is_parameter=is_parameter,
                )
            )
        else:
            schema["type"] = "string"  # Fallback for Union[]

    # Handle dataclasses
    elif dataclasses.is_dataclass(typ):
        schema["type"] = "object"
    elif typing.is_typeddict(typ):
        properties = {}
        required = []
        for field_name, field_type in typ.__annotations__.items():
            # Check if field is wrapped in Required/NotRequired
            origin = typing.get_origin(field_type)
            if origin is Required:
                is_required = True
                field_type = typing.get_args(field_type)[0]
            elif origin is NotRequired:
                is_required = False
                field_type = typing.get_args(field_type)[0]
            else:
                # Fall back to checking __required_keys__
                is_required = field_name in getattr(
                    typ, "__required_keys__", {field_name}
                )

            properties[field_name] = _resolve_type_annotation(
                field_type,
                is_parameter=is_parameter,
            )
            if is_required:
                required.append(field_name)

        schema.update({
            "type": "object",
            "properties": properties,
        })
        if required:
            schema["required"] = required
    # Handle mappings - updated check
    elif (
        origin in (dict, typing.Dict)  # noqa: UP006
        or (origin is not None and isinstance(origin, type) and issubclass(origin, dict))
    ):
        schema["type"] = "object"
        if is_parameter:  # Only add additionalProperties for parameters
            schema["additionalProperties"] = True

    # Handle sequences
    elif origin in (
        list,
        set,
        tuple,
        frozenset,
        typing.List,  # noqa: UP006  # pyright: ignore
        typing.Set,  # noqa: UP006  # pyright: ignore
    ) or (
        origin is not None
        and origin.__module__ == "collections.abc"
        and origin.__name__ in {"Sequence", "MutableSequence", "Collection"}
    ):
        schema["type"] = "array"
        item_type = args[0] if args else Any
        schema["items"] = _resolve_type_annotation(
            item_type,
            is_parameter=is_parameter,
        )

    # Handle literals
    elif origin is typing.Literal:
        schema["type"] = "string"
        schema["enum"] = list(args)

    # Handle basic types
    elif isinstance(typ, type):
        if issubclass(typ, enum.Enum):
            schema["type"] = "string"
            schema["enum"] = [e.value for e in typ]

        # Basic types
        elif typ in (str, Path, UUID, re.Pattern):
            schema["type"] = "string"
        elif typ is int:
            schema["type"] = "integer"
        elif typ in (float, decimal.Decimal):
            schema["type"] = "number"
        elif typ is bool:
            schema["type"] = "boolean"

        # String formats
        elif typ is datetime:
            schema["type"] = "string"
            schema["format"] = "date-time"
            if description:
                description = f"{description} (ISO 8601 format)"
        elif typ is date:
            schema["type"] = "string"
            schema["format"] = "date"
            if description:
                description = f"{description} (ISO 8601 format)"
        elif typ is time:
            schema["type"] = "string"
            schema["format"] = "time"
            if description:
                description = f"{description} (ISO 8601 format)"
        elif typ is timedelta:
            schema["type"] = "string"
            if description:
                description = f"{description} (ISO 8601 duration)"
        elif typ is timezone:
            schema["type"] = "string"
            if description:
                description = f"{description} (IANA timezone name)"
        elif typ is UUID:
            schema["type"] = "string"
        elif typ in (bytes, bytearray):
            schema["type"] = "string"
            if description:
                description = f"{description} (base64 encoded)"
        elif typ is ipaddress.IPv4Address or typ is ipaddress.IPv6Address:
            schema["type"] = "string"
        elif typ is complex:
            schema.update({
                "type": "object",
                "properties": {
                    "real": {"type": "number"},
                    "imag": {"type": "number"},
                },
            })
        # Default to object for unknown types
        else:
            schema["type"] = "object"
    else:
        # Default for unmatched types
        schema["type"] = "string"

    # Add description if provided
    if description is not None:
        schema["description"] = description

    # Add default if provided and not empty
    if default is not inspect.Parameter.empty:
        schema["default"] = default

    from py2openai.typedefs import (
        _create_array_property,
        _create_object_property,
        _create_simple_property,
    )

    if schema["type"] == "array":
        return _create_array_property(
            items=schema["items"],
            description=schema.get("description"),
        )
    if schema["type"] == "object":
        prop = _create_object_property(description=schema.get("description"))
        if "properties" in schema:
            prop["properties"] = schema["properties"]
        if "additionalProperties" in schema:
            prop["additionalProperties"] = schema["additionalProperties"]
        if "required" in schema:
            prop["required"] = schema["required"]
        return prop

    return _create_simple_property(
        type_str=schema["type"],
        description=schema.get("description"),
        enum_values=schema.get("enum"),
        default=default if default is not inspect.Parameter.empty else None,
        fmt=schema.get("format"),
    )


def _determine_function_type(func: Callable[..., Any]) -> FunctionType:
    """Determine the type of the function.

    Args:
        func: Function to check

    Returns:
        FunctionType indicating the function's type
    """
    if inspect.isasyncgenfunction(func):
        return FunctionType.ASYNC_GENERATOR
    if inspect.isgeneratorfunction(func):
        return FunctionType.SYNC_GENERATOR
    if inspect.iscoroutinefunction(func):
        return FunctionType.ASYNC
    return FunctionType.SYNC


def create_schema(
    func: Callable[..., Any],
    name_override: str | None = None,
) -> FunctionSchema:
    """Create an OpenAI function schema from a Python function.

    Args:
        func: Function to create schema for
        name_override: Optional name override (otherwise the function name)

    Returns:
        Schema representing the function

    Raises:
        TypeError: If input is not callable

    Note:
        Variable arguments (*args) and keyword arguments (**kwargs) are not
        supported in OpenAI function schemas and will be ignored with a warning.
    """
    if not callable(func):
        msg = f"Expected callable, got {type(func)}"
        raise TypeError(msg)

    # Parse function signature and docstring
    sig = inspect.signature(func)
    docstring = docstring_parser.parse(func.__doc__ or "")

    # Get clean type hints without extras
    try:
        hints = typing.get_type_hints(func, localns=locals())
    except NameError:
        msg = "Unable to resolve type hints for function %s, skipping"
        logger.warning(msg, func.__name__)
        hints = {}
    # Process parameters
    parameters: ToolParameters = {"type": "object", "properties": {}}
    required: list[str] = []
    params = list(sig.parameters.items())
    skip_first = (
        inspect.isfunction(func)
        and not inspect.ismethod(func)
        and params
        and params[0][0] == "self"
    )
    for i, (name, param) in enumerate(sig.parameters.items()):
        # Skip the first parameter for bound methods
        if skip_first and i == 0:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        param_doc = next(
            (p.description for p in docstring.params if p.arg_name == name),
            None,
        )

        param_type = hints.get(name, Any)
        parameters["properties"][name] = _resolve_type_annotation(
            param_type,
            description=param_doc,
            default=param.default,
            is_parameter=True,
        )

        if param.default is inspect.Parameter.empty:
            required.append(name)

    # Handle return type with is_parameter=False
    function_type = _determine_function_type(func)
    return_hint = hints.get("return", Any)

    if function_type in (FunctionType.SYNC_GENERATOR, FunctionType.ASYNC_GENERATOR):
        element_type = next(
            (t for t in typing.get_args(return_hint) if t is not type(None)),
            Any,
        )
        returns_dct = {
            "type": "array",
            "items": _resolve_type_annotation(element_type, is_parameter=False),
        }
    else:
        returns = _resolve_type_annotation(return_hint, is_parameter=False)
        returns_dct = dict(returns)  # type: ignore
    return FunctionSchema(
        name=name_override or func.__name__,
        description=docstring.short_description,
        parameters=parameters,
        required=required,
        returns=returns_dct,
        function_type=function_type,
    )


if __name__ == "__main__":
    import json

    def get_weather(
        location: str,
        unit: typing.Literal["C", "F"] = "C",
        detailed: bool = False,
    ) -> dict[str, str | float]:
        """Get the weather for a location.

        Args:
            location: City or address to get weather for
            unit: Temperature unit (Celsius or Fahrenheit)
            detailed: Include extended forecast
        """
        return {"temp": 22.5, "conditions": "sunny"}

    # Create schema and executable function
    schema = create_schema(get_weather)

    # Print the schema
    print("OpenAI Function Schema:")
    print(json.dumps(schema.model_dump_openai(), indent=2))
