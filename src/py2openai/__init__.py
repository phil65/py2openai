"""Py2OpenAI: Create OpenAI-compatible function schemas from python functions"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("py2openai")
__title__ = "Py2OpenAI"
__description__ = "Create OpenAI-compatible function schemas from python functions"
__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2024 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/py2openai"

from py2openai.executable import create_executable, ExecutableFunction
from py2openai.functionschema import FunctionType, create_schema
from py2openai.schema_generators import (
    create_schemas_from_callables,
    create_schemas_from_module,
    create_schemas_from_class,
    create_constructor_schema,
)
from py2openai.typedefs import OpenAIFunctionDefinition, OpenAIFunctionTool

__all__ = [
    "ExecutableFunction",
    "FunctionType",
    "OpenAIFunctionDefinition",
    "OpenAIFunctionTool",
    "__version__",
    "create_constructor_schema",
    "create_executable",
    "create_schema",
    "create_schemas_from_callables",
    "create_schemas_from_class",
    "create_schemas_from_module",
]
