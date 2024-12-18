from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
import importlib
import inspect
import types
from typing import Any, Literal

from py2openai.functionschema import FunctionSchema, create_schema


def create_schemas_from_callables(
    callables: dict[str, Callable[..., Any]],
    prefix: str | Literal[False] | None = None,
    exclude_private: bool = True,
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas from a dictionary of callables.

    Args:
        callables: Dictionary mapping names to callable objects
        prefix: Schema name prefix to prepend to function names.
               If None, no prefix. If False, use raw name.
               If string, uses that prefix.
        exclude_private: Whether to exclude callables starting with underscore

    Returns:
        Dictionary mapping qualified names to FunctionSchema objects

    Example:
        >>> def foo(x: int) -> str: ...
        >>> def bar(y: float) -> int: ...
        >>> callables = {'foo': foo, 'bar': bar}
        >>> schemas = create_schemas_from_callables(callables, prefix='math')
        >>> print(schemas['math.foo'])
    """
    schemas = {}

    for name, callable_obj in callables.items():
        # Skip private members if requested
        if exclude_private and name.startswith("_"):
            continue

        # Generate schema key based on prefix setting
        key = name if prefix is False else f"{prefix}.{name}" if prefix else name
        schemas[key] = create_schema(callable_obj)

    return schemas


def create_schemas_from_class(
    cls: type,
    prefix: str | Literal[False] | None = None,
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas for all public methods in a class.

    Args:
        cls: The class to generate schemas from
        prefix: Schema name prefix. If None, uses class name.
               If False, no prefix. If string, uses that prefix.

    Returns:
        Dictionary mapping qualified method names to FunctionSchema objects

    Example:
        >>> class MyClass:
        ...     def my_method(self, x: int) -> str:
        ...         return str(x)
        >>> schemas = create_schemas_from_class(MyClass)
        >>> print(schemas['MyClass.my_method'])
    """
    callables: dict[str, Callable[..., Any]] = {}

    # Get all attributes of the class
    for name, attr in inspect.getmembers(cls):
        # Handle different method types
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            callables[name] = attr
        elif isinstance(attr, classmethod | staticmethod):
            callables[name] = attr.__get__(None, cls)

    # Use default prefix of class name if not specified
    effective_prefix = cls.__name__ if prefix is None else prefix
    return create_schemas_from_callables(callables, prefix=effective_prefix)


def create_schemas_from_module(
    module: types.ModuleType | str,
    include_functions: list[str] | None = None,
    prefix: str | Literal[False] | None = None,
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas from a Python module's functions.

    Args:
        module: Either a ModuleType object or string name of module to analyze
        include_functions: Optional list of function names to specifically include
        prefix: Schema name prefix. If None, uses module name.
                If False, no prefix. If string, uses that prefix.

    Returns:
        Dictionary mapping function names to FunctionSchema objects

    Raises:
        ImportError: If module string name cannot be imported

    Example:
        >>> import math
        >>> schemas = create_schemas_from_module(math, ['sin', 'cos'])
        >>> print(schemas['math.sin'])
    """
    # Resolve module if string name provided
    mod = (
        module
        if isinstance(module, types.ModuleType)
        else importlib.import_module(module)
    )

    # Get all functions from module
    callables: dict[str, Callable[..., Any]] = {
        name: func
        for name, func in inspect.getmembers(mod, predicate=inspect.isfunction)
        if include_functions is None
        or (name in include_functions and func.__module__.startswith(mod.__name__))
    }

    # Use default prefix of module name if not specified
    effective_prefix = mod.__name__ if prefix is None else prefix
    return create_schemas_from_callables(callables, prefix=effective_prefix)


if __name__ == "__main__":
    schemas = create_schemas_from_module(__name__)
    print(schemas)
