"""Tests for schema conversion between OpenAI and Python formats."""

from __future__ import annotations

import inspect
from typing import Any, Literal

import pytest

from py2openai.functionschema import FunctionSchema, create_schema


def test_from_dict_complete_tool() -> None:
    """Test creating schema from complete OpenAI tool definition."""
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "default": "C",
                    },
                },
                "required": ["location"],
            },
        },
    }
    schema = FunctionSchema.from_dict(tool_schema)

    assert schema.name == "get_weather"
    assert schema.description == "Get weather information"
    props = schema.parameters["properties"]
    assert props["location"]["type"] == "string"
    assert props["unit"]["enum"] == ["C", "F"]
    assert schema.required == ["location"]


def test_from_dict_function_only() -> None:
    """Test creating schema from function definition only."""
    func_schema = {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["C", "F"],
                    "default": "C",
                },
            },
            "required": ["location"],
        },
    }
    schema = FunctionSchema.from_dict(func_schema)

    assert schema.name == "get_weather"
    assert schema.parameters["properties"]["unit"]["enum"] == ["C", "F"]
    assert schema.required == ["location"]


def test_from_dict_invalid_schema() -> None:
    """Test creating schema from invalid input."""
    with pytest.raises(KeyError, match="function"):
        FunctionSchema.from_dict({"type": "function"})  # missing function def

    with pytest.raises(KeyError, match="name"):
        FunctionSchema.from_dict({"parameters": {}})  # missing name


def test_roundtrip_conversion() -> None:
    """Test converting Python function to OpenAI schema and back."""

    def get_weather(
        location: str,
        unit: Literal["C", "F"] = "C",
        detailed: bool = False,
    ) -> dict[str, Any]:
        """Get weather information.

        Args:
            location: The city name
            unit: Temperature unit
            detailed: Include details
        """
        return {"temp": 22.5}

    # Create schema from function
    schema1 = create_schema(get_weather)

    # Convert to OpenAI format
    openai_schema = schema1.model_dump_openai()

    # Convert back to our schema
    schema2 = FunctionSchema.from_dict(openai_schema)  # type: ignore

    # Compare key attributes
    assert schema1.name == schema2.name
    assert schema1.description == schema2.description
    assert schema1.parameters == schema2.parameters
    assert schema1.required == schema2.required


def test_schema_to_signature_roundtrip() -> None:
    """Test converting between function signatures and schemas."""

    def original_func(
        x: int,
        y: Literal["a", "b"] = "a",
        z: bool | None = None,
    ) -> dict[str, Any]:
        return {"x": x, "y": y, "z": z}

    # Create schema from original function
    schema = create_schema(original_func)

    # Get signature back from schema
    sig = schema.to_python_signature()

    # Compare signatures
    original_sig = inspect.signature(original_func)

    # Compare parameter names and kinds
    assert sig.parameters.keys() == original_sig.parameters.keys()

    # Compare parameter defaults
    for name, param in sig.parameters.items():
        orig_param = original_sig.parameters[name]
        if name in schema.required:
            assert param.default is inspect.Parameter.empty
        else:
            assert param.default == orig_param.default

        # Type comparisons
        if name == "y":
            assert param.annotation == Literal["a", "b"]
        else:
            # Handle forward references vs resolved types
            expected = orig_param.annotation
            if isinstance(expected, str):
                # Convert string annotation to actual type
                expected = eval(expected, globals(), locals())
            assert param.annotation == expected


def test_model_dump_openai_format() -> None:
    """Test the format of OpenAI schema output."""

    def func(x: int, y: str = "default") -> None:
        """Test function."""

    schema = create_schema(func)
    openai_schema = schema.model_dump_openai()

    assert isinstance(openai_schema, dict)
    assert openai_schema["type"] == "function"
    assert isinstance(openai_schema["function"], dict)

    function_def = openai_schema["function"]
    assert isinstance(function_def["name"], str)
    assert isinstance(function_def["description"], str)
    assert isinstance(function_def["parameters"], dict)

    params = function_def["parameters"]
    assert params["type"] == "object"
    assert isinstance(params["properties"], dict)
    assert isinstance(params.get("required"), list)


if __name__ == "__main__":
    pytest.main([__file__])
