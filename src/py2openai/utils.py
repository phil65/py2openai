from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import sys
from typing import Any, get_type_hints as _get_type_hints


def get_type_hints(
    fn: Callable[..., Any],
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = sys.modules[fn.__module__]

    result_ns = {
        **module.__dict__,
        "Sequence": Sequence,
        "Iterator": Iterator,
        "Mapping": Mapping,
        "List": list,
        "Dict": dict,
        "Set": set,
        "Tuple": tuple,
        "Any": Any,
    }
    if globalns is not None:
        result_ns = {**globalns, **result_ns}
    return _get_type_hints(
        fn,
        include_extras=True,
        localns=localns or locals(),
        globalns=result_ns,
    )
