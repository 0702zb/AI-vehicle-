"""LangGraph skeleton matching Agent Inputs §5 business data flow."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from concept_platform.state import NodeStatus, RunState

# Node IDs map one-to-one to the boxes in Agent Inputs, Outputs, and
# Implementation.md §5. request_engineering_run is deliberately not a node.
GRAPH_NODES = (
    "user_request",
    "mira",
    "brief_confirmation",
    "scenario_snapshot",
    "innovator",
    "candidate_concept",
    "orchestrator",
    "package",
    "drive_energy",
    "engineering_core",
    "reality_check",
    "other_specialists",
    "expert_review",
    "concept_paper",
    "report_issuance",
)

GRAPH_EDGES = (
    ("user_request", "mira"),
    ("mira", "brief_confirmation"),
    ("brief_confirmation", "scenario_snapshot"),
    ("scenario_snapshot", "innovator"),
    ("innovator", "candidate_concept"),
    ("candidate_concept", "orchestrator"),
    ("orchestrator", "package"),
    ("orchestrator", "drive_energy"),
    ("orchestrator", "other_specialists"),
    ("package", "orchestrator"),
    ("drive_energy", "engineering_core"),
    ("engineering_core", "reality_check"),
    ("reality_check", "orchestrator"),
    ("other_specialists", "orchestrator"),
    ("orchestrator", "expert_review"),
    ("expert_review", "concept_paper"),
    ("concept_paper", "report_issuance"),
)


def route_after_orchestration(state: RunState) -> Any:
    """Pure state-only route; no model, database, or network access."""

    statuses = state["node_status"]
    specialist_nodes = {"package", "drive_energy", "other_specialists"}
    started = specialist_nodes.intersection(statuses)
    if not started:
        branch_state = dict(state)
        return [
            Send("package", branch_state),
            Send("drive_energy", branch_state),
            Send("other_specialists", branch_state),
        ]

    required = {"package", "drive_energy", "engineering_core", "reality_check", "other_specialists"}
    if not required.issubset(statuses):
        # Other branches may still be running in parallel. Their final return
        # reaches this pure router again; this partial branch terminates here.
        return "stop"
    if (
        statuses["reality_check"] != NodeStatus.SUPPORTED
        or state["coverage"].get("reality_check_decision") != "PASS"
    ):
        return "stop"
    return "expert_review"


def route_after_mira(state: RunState) -> str:
    """Stop before confirmation if the shared runtime cannot produce a M.I.R.A. draft."""

    if state["node_status"].get("mira") == NodeStatus.BLOCKED:
        return "stop"
    return "brief_confirmation"


def route_after_expert_review(state: RunState) -> str:
    """Only an explicit human approval can advance to the paper agent."""

    if state["node_status"].get("expert_review") == NodeStatus.SUPPORTED and state["approval_version"]:
        return "concept_paper"
    return "stop"


def route_after_concept_paper(state: RunState) -> str:
    if state["node_status"].get("concept_paper") == NodeStatus.SUPPORTED:
        return "report_issuance"
    return "stop"


def _placeholder_node(node_id: str):
    """Keep the graph executable until each node receives its step-specific logic."""

    def run(state: RunState) -> dict[str, Any]:
        if node_id == "engineering_core":
            # No model adapter is registered in the skeleton: explicit BLOCKED.
            status = NodeStatus.BLOCKED
        elif node_id == "reality_check":
            status = NodeStatus.UNKNOWN
        else:
            status = NodeStatus.UNKNOWN
        return {"node_status": {node_id: status}}

    return run


def build_graph(
    *,
    checkpointer: Any = None,
    node_overrides: Mapping[str, Callable[..., Any]] | None = None,
):
    """Build the runnable control-flow skeleton with an optional checkpointer."""

    builder = StateGraph(RunState)
    overrides = dict(node_overrides or {})
    invalid_overrides = set(overrides) - set(GRAPH_NODES)
    if invalid_overrides:
        raise ValueError("Node overrides contain unregistered graph nodes: " + ", ".join(sorted(invalid_overrides)))
    for node_id in GRAPH_NODES:
        builder.add_node(node_id, overrides.get(node_id, _placeholder_node(node_id)))

    builder.add_edge(START, "user_request")
    for source, target in GRAPH_EDGES:
        if source in {"orchestrator", "mira", "expert_review", "concept_paper"}:
            continue
        builder.add_edge(source, target)
    builder.add_conditional_edges(
        "mira",
        route_after_mira,
        {"brief_confirmation": "brief_confirmation", "stop": END},
    )
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestration,
        {
            "package": "package",
            "drive_energy": "drive_energy",
            "other_specialists": "other_specialists",
            "expert_review": "expert_review",
            "stop": END,
        },
    )
    builder.add_conditional_edges(
        "expert_review",
        route_after_expert_review,
        {"concept_paper": "concept_paper", "stop": END},
    )
    builder.add_conditional_edges(
        "concept_paper",
        route_after_concept_paper,
        {"report_issuance": "report_issuance", "stop": END},
    )
    builder.add_edge("report_issuance", END)
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def postgres_checkpointer(database_url: str) -> AsyncIterator[Any]:
    """Open and initialize the production PostgreSQL checkpointer."""

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        await saver.setup()
        yield saver


@asynccontextmanager
async def graph_with_postgres_checkpointer(
    database_url: str,
) -> AsyncIterator[Any]:
    """Yield a compiled graph while its asynchronous saver remains open."""

    async with postgres_checkpointer(database_url) as saver:
        yield build_graph(checkpointer=saver)


def graph_snapshot(graph: Any) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return normalized graph node IDs and business edges for contract tests."""

    snapshot = graph.get_graph()
    nodes = tuple(sorted(node_id for node_id in GRAPH_NODES if node_id in snapshot.nodes))
    edges = tuple(
        sorted(
            (edge.source, edge.target)
            for edge in snapshot.edges
            if edge.source in GRAPH_NODES and edge.target in GRAPH_NODES
        )
    )
    return nodes, edges
