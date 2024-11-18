from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable  # noqa: TCH003
from typing import Any, TypeVar

from py2openai.functionschema import FunctionSchema, FunctionType, create_schema


T = TypeVar("T")


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

    def run(self, *args: Any, **kwargs: Any) -> T | list[T]:  # noqa: PLR0911
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
    import typing

    def get_weather(
        location: str,
        unit: typing.Literal["C", "F"] = "C",
        detailed: bool = False,
    ) -> dict[str, str | float]:
        return {"temp": 22.5, "conditions": "sunny"}

    exe = create_executable(get_weather)
    # Execute the function
    result = exe.run("London", unit="C")
    print("\nFunction Result:")
    print(result)
