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
    # Default prefix (class name)
    schemas = create_schemas_from_class(TestClass)
    assert "TestClass.simple_method" in schemas
    assert "TestClass.class_method" in schemas
    assert "TestClass.static_method" in schemas
    assert "TestClass.async_method" in schemas

    # No prefix
    schemas_no_prefix = create_schemas_from_class(TestClass, prefix=False)
    assert "simple_method" in schemas_no_prefix
    assert "class_method" in schemas_no_prefix
    assert "static_method" in schemas_no_prefix
    assert "async_method" in schemas_no_prefix

    # Custom prefix
    schemas_custom = create_schemas_from_class(TestClass, prefix="MyAPI")
    assert "MyAPI.simple_method" in schemas_custom
    assert "MyAPI.class_method" in schemas_custom
    assert "MyAPI.static_method" in schemas_custom
    assert "MyAPI.async_method" in schemas_custom

    # Verify contents are the same regardless of prefix
    for name in ["simple_method", "class_method", "static_method", "async_method"]:
        default_schema = schemas[f"TestClass.{name}"]
        no_prefix_schema = schemas_no_prefix[name]
        custom_schema = schemas_custom[f"MyAPI.{name}"]

        # Names will differ but contents should be identical
        assert default_schema.parameters == no_prefix_schema.parameters
        assert default_schema.returns == custom_schema.returns
        assert default_schema.function_type == custom_schema.function_type

    # Verify function types
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
    """Test schema generation from modules with different prefix options."""
    import py2openai.schema_generators as schema_module

    # Test default prefix (module name)
    schemas = create_schemas_from_module(schema_module)
    assert "py2openai.schema_generators.create_schema" in schemas
    assert "py2openai.schema_generators.create_schemas_from_class" in schemas
    assert "py2openai.schema_generators.create_schemas_from_module" in schemas

    # Test with no prefix
    no_prefix = create_schemas_from_module(schema_module, prefix=False)
    assert "create_schema" in no_prefix
    assert "create_schemas_from_class" in no_prefix
    assert "create_schemas_from_module" in no_prefix

    # Test with custom prefix
    custom = create_schemas_from_module(schema_module, prefix="API")
    assert "API.create_schema" in custom
    assert "API.create_schemas_from_class" in custom
    assert "API.create_schemas_from_module" in custom

    # Test with include_functions
    included = create_schemas_from_module(
        schema_module, include_functions=["create_schema"], prefix="API"
    )
    assert len(included) == 1
    assert "API.create_schema" in included

    # Verify schemas are identical regardless of prefix
    base_schema = schemas["py2openai.schema_generators.create_schema"]
    no_prefix_schema = no_prefix["create_schema"]
    custom_schema = custom["API.create_schema"]

    assert base_schema.parameters == no_prefix_schema.parameters
    assert base_schema.returns == custom_schema.returns
    assert base_schema.function_type == custom_schema.function_type
