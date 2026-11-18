from concept_platform.graph import GRAPH_EDGES, GRAPH_NODES, build_graph, graph_snapshot
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


def test_graph_executes_skeleton_with_unknowns_and_unregistered_model_blocked() -> None:
    result = build_graph().invoke(initial_state(), {"recursion_limit": 50})

    assert result["node_status"]["engineering_core"] == NodeStatus.BLOCKED
    assert result["node_status"]["reality_check"] == NodeStatus.UNKNOWN
    assert result["node_status"]["package"] == NodeStatus.UNKNOWN
    assert result["node_status"]["other_specialists"] == NodeStatus.UNKNOWN
    assert "request_engineering_run" not in GRAPH_NODES
