from concept_platform.graph import (
    GRAPH_EDGES,
    GRAPH_NODES,
    build_graph,
    graph_snapshot,
    make_orchestrator_node,
    route_after_reality_check,
)
from concept_platform.state import NodeStatus, RunState, validate_run_state


def initial_state() -> RunState:
    return {
        "run_id": "run-1",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:manifest",
        "node_status": {},
        "coverage": {},
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


def test_graph_snapshot_matches_md_section_5_nodes_and_edges() -> None:
    nodes, edges = graph_snapshot(build_graph())

    assert nodes == tuple(sorted(GRAPH_NODES))
    assert edges == tuple(sorted(GRAPH_EDGES))


def test_run_state_has_only_contract_fields_and_fixed_status_vocabulary() -> None:
    validated = validate_run_state(initial_state())

    assert set(validated) == {
        "run_id",
        "brief_version",
        "context_manifest_hash",
        "node_status",
        "coverage",
        "result_ids",
        "approval_version",
        "pending_writes",
    }
    assert {status.value for status in NodeStatus} == {
        "supported",
        "assumption",
        "unknown",
        "blocked",
        "FAIL",
    }


def test_graph_executes_skeleton_with_unknowns_and_reality_stop() -> None:
    result = build_graph().invoke(initial_state(), {"recursion_limit": 50})

    assert result["node_status"]["layer_agent"] == NodeStatus.UNKNOWN
    assert result["node_status"]["property_agent"] == NodeStatus.UNKNOWN
    assert result["node_status"]["reality_check"] == NodeStatus.UNKNOWN
    assert "expert_review" not in result["node_status"]
    assert "engineering_core" not in GRAPH_NODES
    assert "request_engineering_run" not in GRAPH_NODES


def test_reality_check_pass_advances_and_unknown_exits() -> None:
    state = initial_state()
    state["node_status"]["reality_check"] = NodeStatus.SUPPORTED
    state["coverage"]["reality_check_decision"] = "PASS"
    assert route_after_reality_check(state) == "expert_review"

    state["node_status"]["reality_check"] = NodeStatus.UNKNOWN
    state["coverage"]["reality_check_decision"] = "UNKNOWN"
    assert route_after_reality_check(state) == "stop"


def test_orchestrator_selects_candidate_and_collects_sourced_errors() -> None:
    state = initial_state()
    state["coverage"]["candidate_concepts"] = ["concept-a", "concept-b", "concept-c"]
    state["node_status"]["property_agent"] = NodeStatus.FAIL

    patch = make_orchestrator_node()(state)

    assert patch["node_status"]["orchestrator"] is NodeStatus.SUPPORTED
    assert patch["coverage"]["selected_concept"] == "concept-a"
    assert patch["coverage"]["errors"] == [{"source": "property_agent", "status": "FAIL"}]
