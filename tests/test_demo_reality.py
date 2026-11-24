from __future__ import annotations

import asyncio

from langgraph.graph import END

from concept_platform.demo_reality import (
    build_demo_reality_check_node,
    route_after_demo_reality_check,
)
from concept_platform.state import NodeStatus, RunState


def _state(*, decision: str | None = None) -> RunState:
    return {
        "run_id": "run-demo-reality",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "node_status": {"property_agent": NodeStatus.SUPPORTED},
        "coverage": {"reality_check_decision": decision} if decision else {},
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


def test_demo_reality_check_passes_with_synthetic_fixture() -> None:
    node = build_demo_reality_check_node()

    patch = asyncio.run(node(_state()))

    assert patch["node_status"]["reality_check"] is NodeStatus.SUPPORTED
    assert patch["coverage"]["reality_check_decision"] == "PASS"
    assert patch["result_ids"][0].startswith("sha256:")

    evaluations = patch["coverage"]["reality_check_evaluations"]
    assert len(evaluations) == 1
    assert evaluations[0]["rule_id"] == "demo-mass-hard-limit"
    assert evaluations[0]["decision"] == "PASS"
    assert evaluations[0]["actual_value"] == "1580"
    assert evaluations[0]["threshold"] == "1600"
    assert evaluations[0]["delta"] == "-20"
    assert evaluations[0]["reason"] == "typed_rule_passed"
    assert patch["coverage"]["reality_check_rule_snapshot"] == "demo-rules-v1"


def test_demo_reality_route_ends_on_pass_and_loops_on_fail() -> None:
    assert route_after_demo_reality_check(_state(decision="PASS")) == END
    assert route_after_demo_reality_check(_state(decision="UNKNOWN")) == END
    assert route_after_demo_reality_check(_state(decision="FAIL")) == "orchestrator"
