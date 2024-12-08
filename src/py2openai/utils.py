from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import sys
from typing import Any, get_type_hints as _get_type_hints


def get_type_hints(
    fn: Callable[..., Any],
    localns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = sys.modules[fn.__module__]

    globalns = {
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

    return _get_type_hints(
        fn,
        include_extras=True,
        localns=localns or locals(),
        globalns=globalns,
    )
