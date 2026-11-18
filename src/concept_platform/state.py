"""The deliberately small control-plane state contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class NodeStatus(StrEnum):
    SUPPORTED = "supported"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"
    FAIL = "FAIL"


def merge_mappings(
    current: Mapping[str, Any] | None, update: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge parallel node updates by key without adding state fields."""

    return {**(current or {}), **(update or {})}


def merge_unique_ids(current: list[str] | None, update: list[str] | None) -> list[str]:
    """Merge parallel identifier updates while preserving first-seen order."""

    return list(dict.fromkeys([*(current or []), *(update or [])]))


class RunState(TypedDict):
    """LangGraph state. Keep this field set aligned with the platform contract."""

    run_id: str
    brief_version: str
    context_manifest_hash: str
    node_status: Annotated[dict[str, NodeStatus], merge_mappings]
    coverage: Annotated[dict[str, Any], merge_mappings]
    result_ids: Annotated[list[str], merge_unique_ids]
    approval_version: str | None
    pending_writes: Annotated[list[str], merge_unique_ids]


class RunStateModel(BaseModel):
    """Pydantic v2 boundary validator for API, queue, and checkpoint state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    brief_version: str = Field(min_length=1)
    context_manifest_hash: str = Field(min_length=1)
    node_status: dict[str, NodeStatus]
    coverage: dict[str, Any]
    result_ids: list[str]
    approval_version: str | None
    pending_writes: list[str]


def validate_run_state(value: Mapping[str, Any]) -> RunState:
    """Validate an incoming state and return the exact graph state shape."""

    return RunStateModel.model_validate(value).model_dump()  # type: ignore[return-value]
