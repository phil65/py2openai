from __future__ import annotations

import importlib
import inspect
import types

from py2openai.functionschema import FunctionSchema, create_schema


def create_schemas_from_class(cls: type) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas for all public methods in a class.

    This function analyzes a class and creates OpenAI function schemas for all of its
    public methods, including instance methods, class methods, static methods, and
    async methods. Private methods (starting with '_') are excluded.

    Args:
        cls: The class to generate schemas from. Must be a type object.

    Returns:
        A dictionary mapping qualified method names (e.g. 'ClassName.method_name')
        to their corresponding FunctionSchema objects. Only includes public methods.

    Example:
        >>> class MyClass:
        ...     def my_method(self, x: int) -> str:
        ...         return str(x)
        >>> schemas = create_schemas_from_class(MyClass)
        >>> print(schemas['MyClass.my_method'])
    """
    schemas = {}

    # Get all attributes of the class
    for name, attr in inspect.getmembers(cls):
        # Skip private/special methods
        if name.startswith("_"):
            continue

        # Handle different method types
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            # Regular methods
            schemas[f"{cls.__name__}.{name}"] = create_schema(attr)
        elif isinstance(attr, classmethod | staticmethod):
            # Class methods and static methods
            # Get the underlying function
            func = attr.__get__(None, cls)
            schemas[f"{cls.__name__}.{name}"] = create_schema(func)

    return schemas


def create_schemas_from_module(
    module: types.ModuleType | str, include_functions: list[str] | None = None
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas from a Python module's functions.

    This function analyzes a module and creates OpenAI function schemas for its public
    functions. It can accept either a module object or a string name to import.
    Private functions (starting with '_') are excluded by default.

    Args:
        module: Either a ModuleType object or string name of module to analyze
        include_functions: Optional list of function names to specifically include.
            If None, all public functions are included.

    Returns:
        A dictionary mapping function names to their corresponding FunctionSchema
        objects. Only includes public functions unless specified in include_functions.

    Raises:
        ImportError: If the module string name cannot be imported
        ValueError: If the module argument is neither a ModuleType nor string

    Example:
        >>> import math
        >>> schemas = create_schemas_from_module(math, ['sin', 'cos'])
        >>> print(schemas['sin'])
    """
    mod = (
        module
        if isinstance(module, types.ModuleType)
        else importlib.import_module(module)
    )
    schemas = {}
    for name, func in inspect.getmembers(mod, predicate=inspect.isfunction):
        if (
            include_functions is None or name in include_functions
        ) and not name.startswith("_"):
            schemas[name] = create_schema(func)
    return schemas


if __name__ == "__main__":
    schemas = create_schemas_from_module(__name__)
    print(schemas)
