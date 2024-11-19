"""Simple tests for class and module schema generation functions."""

import enum
from typing import Any

from py2openai.functionschema import (
    FunctionSchema,
    FunctionType,
    create_schema,
)
from py2openai.schema_generators import (
    create_schemas_from_class,
    create_schemas_from_module,
)


class TestClass:
    """Test class with various method types."""

    def __init__(self, value: int) -> None:
        self.value = value

    def simple_method(self, x: int) -> int:
        """A simple bound method.

        Args:
            x: Input value

        Returns:
            Sum of input and instance value
        """
        return x + self.value

    @classmethod
    def class_method(cls, y: str) -> str:
        """A class method.

        Args:
            y: Input string

        Returns:
            Modified string
        """
        return f"{cls.__name__}_{y}"

    @staticmethod
    def static_method(z: float) -> float:
        """A static method.

        Args:
            z: Input number

        Returns:
            Doubled input
        """
        return z * 2.0

    async def async_method(self, data: dict[str, Any]) -> dict[str, Any]:
        """An async method.

        Args:
            data: Input dictionary

        Returns:
            Modified dictionary
        """
        return {**data, "processed": True}


def test_bound_method_schema() -> None:
    """Test schema generation for bound instance methods."""
    instance = TestClass(42)
    schema = create_schema(instance.simple_method)

    assert isinstance(schema, FunctionSchema)
    assert schema.name == "simple_method"
    assert schema.function_type == FunctionType.SYNC
    assert "x" in schema.parameters["properties"]
    assert schema.returns == {"type": "integer"}


def test_class_method_schema() -> None:
    """Test schema generation for class methods."""
    schema = create_schema(TestClass.class_method)

    assert isinstance(schema, FunctionSchema)
    assert schema.name == "class_method"
    assert schema.function_type == FunctionType.SYNC
    assert "y" in schema.parameters["properties"]
    assert schema.returns == {"type": "string"}


def test_static_method_schema() -> None:
    """Test schema generation for static methods."""
    schema = create_schema(TestClass.static_method)

    assert isinstance(schema, FunctionSchema)
    assert schema.name == "static_method"
    assert schema.function_type == FunctionType.SYNC
    assert "z" in schema.parameters["properties"]
    assert schema.returns == {"type": "number"}


def test_async_method_schema() -> None:
    """Test schema generation for async methods."""
    instance = TestClass(42)
    schema = create_schema(instance.async_method)

    assert isinstance(schema, FunctionSchema)
    assert schema.name == "async_method"
    assert schema.function_type == FunctionType.ASYNC
    assert "data" in schema.parameters["properties"]
    assert schema.returns == {"type": "object"}


def test_create_schemas_from_class_methods() -> None:
    """Test creating schemas from all methods in a class."""
    schemas = create_schemas_from_class(TestClass)

    assert isinstance(schemas, dict)
    assert len(schemas) == 4  # noqa: PLR2004
    assert "TestClass.simple_method" in schemas
    assert "TestClass.class_method" in schemas
    assert "TestClass.static_method" in schemas
    assert "TestClass.async_method" in schemas

    # Private methods should be excluded
    assert not any(name.startswith("TestClass._") for name in schemas)

    # Verify method types are correct
    assert schemas["TestClass.simple_method"].function_type == FunctionType.SYNC
    assert schemas["TestClass.class_method"].function_type == FunctionType.SYNC
    assert schemas["TestClass.static_method"].function_type == FunctionType.SYNC
    assert schemas["TestClass.async_method"].function_type == FunctionType.ASYNC


class Color(enum.Enum):
    """Test enum that already exists in our tests."""

    RED = "red"
    BLUE = "blue"


def test_create_schemas_from_class() -> None:
    """Test creating schemas from an existing enum class."""
    schemas = create_schemas_from_class(Color)
    assert isinstance(schemas, dict)
    assert all(isinstance(schema, FunctionSchema) for schema in schemas.values())


def test_create_schemas_from_module() -> None:
    """Test creating schemas from the functionschema module itself."""
    import py2openai.functionschema as schema_module

    schemas = create_schemas_from_module(schema_module)
    assert isinstance(schemas, dict)
    assert all(isinstance(schema, FunctionSchema) for schema in schemas.values())
