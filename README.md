# Py2OpenAI

[![PyPI License](https://img.shields.io/pypi/l/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Package status](https://img.shields.io/pypi/status/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Daily downloads](https://img.shields.io/pypi/dd/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Weekly downloads](https://img.shields.io/pypi/dw/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Monthly downloads](https://img.shields.io/pypi/dm/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Distribution format](https://img.shields.io/pypi/format/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Wheel availability](https://img.shields.io/pypi/wheel/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Python version](https://img.shields.io/pypi/pyversions/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Implementation](https://img.shields.io/pypi/implementation/py2openai.svg)](https://pypi.org/project/py2openai/)
[![Releases](https://img.shields.io/github/downloads/phil65/py2openai/total.svg)](https://github.com/phil65/py2openai/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/py2openai)](https://github.com/phil65/py2openai/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/py2openai)](https://github.com/phil65/py2openai/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/py2openai)](https://github.com/phil65/py2openai/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/py2openai)](https://github.com/phil65/py2openai/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/py2openai)](https://github.com/phil65/py2openai/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/py2openai)](https://github.com/phil65/py2openai/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/py2openai)](https://github.com/phil65/py2openai/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/py2openai)](https://github.com/phil65/py2openai)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/py2openai)](https://github.com/phil65/py2openai/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/py2openai)](https://github.com/phil65/py2openai/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/py2openai)](https://github.com/phil65/py2openai)
[![Github commits this week](https://img.shields.io/github/commit-activity/w/phil65/py2openai)](https://github.com/phil65/py2openai)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/py2openai)](https://github.com/phil65/py2openai)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/py2openai)](https://github.com/phil65/py2openai)
[![Package status](https://codecov.io/gh/phil65/py2openai/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/py2openai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/py2openai/shield.svg)](https://pyup.io/repos/github/phil65/py2openai/)

[Read the documentation!](https://phil65.github.io/py2openai/)

# OpenAI Function Schema Generator

Convert Python functions to OpenAI-compatible function schemas automatically.

## Installation

```bash
pip install openai-function-schema  # not yet published
```

## Basic Usage

```python
from openai_function_schema import create_schema
from typing import Literal

def get_weather(
    location: str,
    unit: Literal["C", "F"] = "C",
    detailed: bool = False,
) -> dict[str, str | float]:
    """Get the weather for a location.

    Args:
        location: City or address to get weather for
        unit: Temperature unit (Celsius or Fahrenheit)
        detailed: Include extended forecast
    """
    return {"temp": 22.5, "conditions": "sunny"}

# Create schema
schema = create_schema(get_weather)

# Use with OpenAI
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    functions=[schema.model_dump_openai()],
    function_call="auto"
)
```

## Supported Types

### Basic Types
```python
def func(
    text: str,              # -> "type": "string"
    number: int,            # -> "type": "integer"
    amount: float,          # -> "type": "number"
    enabled: bool,          # -> "type": "boolean"
    anything: Any,          # -> "type": "string"
) -> None: ...
```

### Container Types
```python
def func(
    items: list[str],                    # -> "type": "array", "items": {"type": "string"}
    numbers: set[int],                   # -> same as list
    mapping: dict[str, Any],            # -> "type": "object", "additionalProperties": true
    nested: list[dict[str, int]],       # -> nested array/object types
    sequence: Sequence[str],            # -> "type": "array"
    collection: Collection[int],        # -> "type": "array"
) -> None: ...
```

### Enums and Literals
```python
class Color(Enum):
    RED = "red"
    BLUE = "blue"

def func(
    color: Color,                       # -> "type": "string", "enum": ["red", "blue"]
    mode: Literal["fast", "slow"],      # -> "type": "string", "enum": ["fast", "slow"]
) -> None: ...
```

### Optional and Union Types
```python
def func(
    opt1: str | None,                   # -> "type": "string"
    opt2: int | None,                   # -> "type": "integer"
    union: str | int,                   # -> "type": "string" (first type)
) -> None: ...
```

### Custom Types
```python
@dataclass
class User:
    name: str
    age: int

def func(
    user: User,                         # -> "type": "object"
    data: JsonDict,                     # -> "type": "object"
) -> None: ...
```

### Type Aliases
```python
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonDict = dict[str, JsonValue]

def func(
    data: JsonDict,                     # -> "type": "object"
    values: list[JsonValue],            # -> "type": "array"
) -> None: ...
```

### Recursive Types
```python
def func(
    tree: dict[str, "dict[str, Any] | str"],  # -> "type": "object"
    nested: dict[str, list["dict[str, Any]"]], # -> "type": "object"
) -> None: ...
```

## Generated Schema Example

```python
{
    "name": "get_weather",
    "description": "Get the weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or address to get weather for"
            },
            "unit": {
                "type": "string",
                "enum": ["C", "F"],
                "description": "Temperature unit (Celsius or Fahrenheit)",
                "default": "C"
            },
            "detailed": {
                "type": "boolean",
                "description": "Include extended forecast",
                "default": false
            }
        },
        "required": ["location"]
    }
}
```

## Schema Generators

### Module Schemas

You can generate schemas for all public functions in a module using `create_schemas_from_module`:

```python
from py2openai import create_schemas_from_module
import math

# Generate schemas for all public functions
schemas = create_schemas_from_module(math)

# Generate schemas for specific functions only
schemas = create_schemas_from_module(math, include_functions=['sin', 'cos'])

# Import module by string name
schemas = create_schemas_from_module('math')
```

### Class Schemas

Generate schemas for all public methods in a class using `create_schemas_from_class`:

```python
from py2openai import create_schemas_from_class

class Calculator:
    def add(self, x: int, y: int) -> int:
        """Add two numbers.

        Args:
            x: First number
            y: Second number

        Returns:
            Sum of x and y
        """
        return x + y

    @classmethod
    def multiply(cls, x: int, y: int) -> int:
        """Multiply two numbers.

        Args:
            x: First number
            y: Second number

        Returns:
            Product of x and y
        """
        return x * y

    @staticmethod
    def divide(x: float, y: float) -> float:
        """Divide two numbers.

        Args:
            x: Numerator
            y: Denominator

        Returns:
            Result of x divided by y
        """
        return x / y

# Generate schemas for all public methods
schemas = create_schemas_from_class(Calculator)

# Access individual method schemas
add_schema = schemas['Calculator.add']
multiply_schema = schemas['Calculator.multiply']
divide_schema = schemas['Calculator.divide']
```

The schema generators support:

- Regular instance methods
- Class methods
- Static methods
- Async methods
- Property methods
- All supported type annotations
- Method docstrings for descriptions
- Default values
- Return type hints
