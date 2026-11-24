from __future__ import annotations

from concept_platform.flow_trace import latest_node_output, trace_node_output
from concept_platform.state import NodeStatus, RunState


def _state() -> RunState:
    return {
        "run_id": "run-flow-trace",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "node_status": {},
        "coverage": {},
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


def test_trace_node_output_records_explicit_node_patch() -> None:
    node = trace_node_output(
        "orchestrator",
        lambda state: {
            "node_status": {"orchestrator": NodeStatus.SUPPORTED},
            "coverage": {"errors": []},
        },
    )

    patch = node(_state())
    entry = patch["coverage"]["flow_trace"][0]

    assert entry["step"] == 1
    assert entry["node_id"] == "orchestrator"
    assert entry["output"]["node_status"]["orchestrator"] == "supported"
    assert entry["output"]["coverage"]["errors"] == []


def test_trace_node_output_does_not_duplicate_full_node_traces() -> None:
    node = trace_node_output(
        "layer_agent",
        lambda state: {
            "node_status": {"layer_agent": NodeStatus.SUPPORTED},
            "coverage": {
                "layer_agent_trace": {"tool_results": {"body": {"summary": "body"}}},
                "errors": [],
            },
        },
    )

    entry = node(_state())["coverage"]["flow_trace"][0]

    assert entry["output"]["coverage"]["errors"] == []
    for key in ("layer_agent_trace", "layer_agent_assessments", "layer_agent_router_output"):
        assert key not in entry["output"]["coverage"]


def test_trace_node_output_preserves_previous_steps() -> None:
    first = trace_node_output(
        "first",
        lambda state: {"node_status": {"first": NodeStatus.SUPPORTED}},
    )(_state())
    state_after_first = {
        **_state(),
        "coverage": {**_state()["coverage"], **first["coverage"]},
        "node_status": first.get("node_status", {}),
    }
    second = trace_node_output(
        "second",
        lambda state: {"node_status": {"second": NodeStatus.ASSUMPTION}},
    )(state_after_first)

    assert [item["node_id"] for item in second["coverage"]["flow_trace"]] == [
        "first",
        "second",
    ]
    assert latest_node_output(state_after_first)["node_id"] == "first"
