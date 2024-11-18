"""Module for creating OpenAI function schemas from Python functions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable  # noqa: TCH003
import dataclasses
import enum
import inspect
import types
import typing
from typing import Annotated, Any, TypeGuard, TypeVar

import docstring_parser
import pydantic


T = TypeVar("T")


class FunctionType(str, enum.Enum):
    """Enum representing different function types."""

    SYNC = "sync"
    ASYNC = "async"
    SYNC_GENERATOR = "sync_generator"
    ASYNC_GENERATOR = "async_generator"


class FunctionSchema(pydantic.BaseModel):
    """Schema representing an OpenAI function."""

    name: str
    description: str | None = None
    parameters: dict[str, Any]
    required: list[str]
    returns: dict[str, Any]

    # Internal metadata - not part of OpenAI schema
    function_type: FunctionType

    def model_dump_openai(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                **self.parameters,
                "required": self.required,
            },
        }


class ExecutableFunction[T]:
    """Wrapper for executing functions with different calling patterns."""

    def __init__(self, schema: FunctionSchema, func: Callable[..., T]) -> None:
        """Initialize with schema and function.

        Args:
            schema: OpenAI function schema
            func: The actual function to execute
        """
        self.schema = schema
        self.func = func

    def run(self, *args: Any, **kwargs: Any) -> T | list[T]:
        """Run the function synchronously."""
        match self.schema.function_type:
            case FunctionType.SYNC:
                return self.func(*args, **kwargs)
            case FunctionType.ASYNC:
                try:
                    # Try to get running loop
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop, create new one
                    return asyncio.run(self.func(*args, **kwargs))
                else:
                    # Have a running loop, create new one if needed
                    if loop.is_running():
                        new_loop = asyncio.new_event_loop()
                        try:
                            return new_loop.run_until_complete(self.func(*args, **kwargs))
                        finally:
                            new_loop.close()
                    return loop.run_until_complete(self.func(*args, **kwargs))
            case FunctionType.SYNC_GENERATOR:
                return list(self.func(*args, **kwargs))
            case FunctionType.ASYNC_GENERATOR:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(self._collect_async_gen(*args, **kwargs))
                else:
                    if loop.is_running():
                        new_loop = asyncio.new_event_loop()
                        try:
                            return new_loop.run_until_complete(
                                self._collect_async_gen(*args, **kwargs),
                            )
                        finally:
                            new_loop.close()
                    return loop.run_until_complete(
                        self._collect_async_gen(*args, **kwargs),
                    )

    async def _collect_async_gen(self, *args: Any, **kwargs: Any) -> list[T]:
        """Collect async generator results into a list."""
        return [x async for x in self.func(*args, **kwargs)]

    async def arun(self, *args: Any, **kwargs: Any) -> T | list[T]:
        """Run the function asynchronously.

        Returns:
            Function result or list of results for generators
        """
        match self.schema.function_type:
            case FunctionType.SYNC:
                return self.func(*args, **kwargs)
            case FunctionType.ASYNC:
                return await self.func(*args, **kwargs)
            case FunctionType.SYNC_GENERATOR:
                return list(self.func(*args, **kwargs))
            case FunctionType.ASYNC_GENERATOR:
                return [x async for x in self.func(*args, **kwargs)]
            case _:
                msg = f"Unknown function type: {self.schema.function_type}"
                raise ValueError(msg)

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[T]:
        """Stream results from the function.

        Yields:
            Individual results as they become available
        """
        match self.schema.function_type:
            case FunctionType.SYNC_GENERATOR:
                for x in self.func(*args, **kwargs):
                    yield x
            case FunctionType.ASYNC_GENERATOR:
                async for x in self.func(*args, **kwargs):
                    yield x
            case FunctionType.SYNC:
                yield self.func(*args, **kwargs)
            case FunctionType.ASYNC:
                yield await self.func(*args, **kwargs)
            case _:
                msg = f"Unknown function type: {self.schema.function_type}"
                raise ValueError(msg)


def _is_optional_type(typ: type) -> TypeGuard[type]:
    """Check if a type is Optional[T] or T | None.

    Args:
        typ: Type to check

    Returns:
        True if the type is Optional, False otherwise
    """
    origin = typing.get_origin(typ)

    # Not a Union/UnionType, can't be Optional
    if origin not in (typing.Union, types.UnionType):
        return False

    args = typing.get_args(typ)
    # Check if any of the union members is None or NoneType
    return any(arg is type(None) for arg in args)


def _resolve_type_annotation(
    typ: type,
    description: str | None = None,
    default: Any = inspect.Parameter.empty,
    is_parameter: bool = True,
) -> dict[str, Any]:
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
    if origin in (typing.Union, types.UnionType):
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
        typing.List,  # noqa: UP006
        typing.Set,  # noqa: UP006
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
        elif typ is str:
            schema["type"] = "string"
        elif typ is int:
            schema["type"] = "integer"
        elif typ is float:
            schema["type"] = "number"
        elif typ is bool:
            schema["type"] = "boolean"
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

    return schema


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


def create_schema(func: Callable[..., Any]) -> FunctionSchema:
    """Create an OpenAI function schema from a Python function.

    Args:
        func: Function to create schema for

    Returns:
        Schema representing the function

    Raises:
        TypeError: If input is not callable
    """
    if not callable(func):
        msg = f"Expected callable, got {type(func)}"
        raise TypeError(msg)

    # Parse function signature and docstring
    sig = inspect.signature(func)
    docstring = docstring_parser.parse(func.__doc__ or "")

    # Get clean type hints without extras
    hints = typing.get_type_hints(func)

    # Process parameters
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    required: list[str] = []

    for name, param in sig.parameters.items():
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
        returns = {
            "type": "array",
            "items": _resolve_type_annotation(element_type, is_parameter=False),
        }
    else:
        returns = _resolve_type_annotation(return_hint, is_parameter=False)

    return FunctionSchema(
        name=func.__name__,
        description=docstring.short_description,
        parameters=parameters,
        required=required,
        returns=returns,
        function_type=function_type,
    )


def create_executable(func: Callable[..., T]) -> ExecutableFunction[T]:
    """Create an executable function wrapper with schema.

    Args:
        func: Function to wrap

    Returns:
        Executable wrapper with schema
    """
    schema = create_schema(func)
    return ExecutableFunction(schema, func)


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
    exe = create_executable(get_weather)

    # Print the schema
    print("OpenAI Function Schema:")
    print(json.dumps(schema.model_dump_openai(), indent=2))

    # Execute the function
    result = exe.run("London", unit="C")
    print("\nFunction Result:")
    print(result)
