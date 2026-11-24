"""Explicit per-node output trace for Studio demonstrations."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

from concept_platform.state import RunState


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=_json_default))
    except (TypeError, ValueError):
        return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return str(value)


_TRACE_BLOB_KEYS = frozenset(
    {
        "layer_agent_trace",
        "layer_agent_assessments",
        "layer_agent_router_output",
        "layer_agent_tool_calls",
        "property_agent_trace",
        "flow_trace",
    }
)


def _slim_output(output: dict[str, Any]) -> dict[str, Any]:
    """Keep step/status visibility without duplicating the full node traces."""

    coverage = dict(output.get("coverage", {}))
    for key in _TRACE_BLOB_KEYS:
        coverage.pop(key, None)
    return {**output, "coverage": coverage}


def _append_trace(
    state: RunState, node_id: str, output: dict[str, Any]
) -> dict[str, Any]:
    existing = state.get("coverage", {}).get("flow_trace", ())
    if not isinstance(existing, (list, tuple)):
        existing = ()
    entry = {
        "step": len(existing) + 1,
        "node_id": node_id,
        "output": _jsonable(_slim_output(output)),
    }
    coverage = dict(output.get("coverage", {}))
    coverage["flow_trace"] = [*existing, entry]
    return {**output, "coverage": coverage}


def trace_node_output(node_id: str, inner: Callable[[RunState], dict[str, Any]]):
    """Wrap a sync or async node and append its exact output to coverage.flow_trace."""

    if inspect.iscoroutinefunction(inner):

        async def traced_async_node(state: RunState) -> dict[str, Any]:
            output = await inner(state)
            return _append_trace(state, node_id, output)

        return traced_async_node

    def traced_node(state: RunState) -> dict[str, Any]:
        output = inner(state)
        return _append_trace(state, node_id, output)

    return traced_node


def latest_node_output(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the latest recorded node output from state.coverage.flow_trace."""

    trace = state.get("coverage", {}).get("flow_trace", ())
    if not trace or not isinstance(trace[-1], Mapping):
        return None
    return dict(trace[-1])
