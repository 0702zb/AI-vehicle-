"""LangGraph skeleton matching Agent Inputs §5 business data flow."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph

from concept_platform.state import NodeStatus, RunState

# Node IDs follow the requested flowchart: orchestrator -> layer_agent ->
# property_agent -> reality_check -> orchestrator.
# request_engineering_run remains an in-node tool, not a graph node.
GRAPH_NODES = (
    "user_request",
    "mira",
    "brief_confirmation",
    "scenario_snapshot",
    "innovator",
    "candidate_concept",
    "orchestrator",
    "layer_agent",
    "property_agent",
    "reality_check",
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
    ("orchestrator", "layer_agent"),
    ("layer_agent", "property_agent"),
    ("layer_agent", "orchestrator"),
    ("property_agent", "reality_check"),
    ("property_agent", "orchestrator"),
    ("reality_check", "orchestrator"),
    ("reality_check", "expert_review"),
    ("expert_review", "concept_paper"),
    ("concept_paper", "report_issuance"),
)


def route_after_orchestration(state: RunState) -> str:
    """Start the chain on first entry; error returns terminate here."""

    if "layer_agent" not in state["node_status"]:
        return "layer_agent"
    return "stop"


def _route_after_chain_node(node_id: str, success_target: str):
    """Return to orchestrator on explicit failure; otherwise advance the chain."""

    def route(state: RunState) -> str:
        status = state["node_status"].get(node_id)
        if status in (NodeStatus.FAIL, NodeStatus.BLOCKED):
            return "orchestrator"
        return success_target

    return route


route_after_layer_agent = _route_after_chain_node("layer_agent", "property_agent")
route_after_property_agent = _route_after_chain_node("property_agent", "reality_check")


def route_after_reality_check(state: RunState) -> str:
    """PASS advances directly; UNKNOWN exits; FAIL/BLOCKED returns to orchestrator."""

    status = state["node_status"].get("reality_check")
    decision = state["coverage"].get("reality_check_decision")
    if status in (NodeStatus.FAIL, NodeStatus.BLOCKED) or decision == "FAIL":
        return "orchestrator"
    if status is NodeStatus.SUPPORTED and decision == "PASS":
        return "expert_review"
    return "stop"


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


def make_orchestrator_node():
    """State/error manager: collect sourced errors and select a candidate concept."""

    def orchestrator_node(state: RunState) -> dict[str, Any]:
        errors = list(state["coverage"].get("errors", []))
        seen = {(item.get("source"), item.get("status")) for item in errors}
        for node_id, status in state["node_status"].items():
            if status in (NodeStatus.FAIL, NodeStatus.BLOCKED):
                key = (node_id, status.value)
                if key not in seen:
                    errors.append({"source": node_id, "status": status.value})
                    seen.add(key)

        coverage: dict[str, Any] = {"errors": errors}
        concepts = state["coverage"].get("candidate_concepts", [])
        if concepts and not state["coverage"].get("selected_concept"):
            coverage["selected_concept"] = concepts[0]

        return {
            "node_status": {"orchestrator": NodeStatus.SUPPORTED},
            "coverage": coverage,
        }

    return orchestrator_node


def _placeholder_node(node_id: str):
    """Keep the graph executable until each node receives its step-specific logic."""

    def run(state: RunState) -> dict[str, Any]:
        return {"node_status": {node_id: NodeStatus.UNKNOWN}}

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
    orchestrator_node = make_orchestrator_node()
    for node_id in GRAPH_NODES:
        default_node = orchestrator_node if node_id == "orchestrator" else _placeholder_node(node_id)
        builder.add_node(node_id, overrides.get(node_id, default_node))

    builder.add_edge(START, "user_request")
    conditional_sources = {
        "orchestrator",
        "mira",
        "layer_agent",
        "property_agent",
        "reality_check",
        "expert_review",
        "concept_paper",
    }
    for source, target in GRAPH_EDGES:
        if source in conditional_sources:
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
        {"layer_agent": "layer_agent", "stop": END},
    )
    for source, router, targets in (
        ("layer_agent", route_after_layer_agent, {"orchestrator": "orchestrator", "property_agent": "property_agent"}),
        ("property_agent", route_after_property_agent, {"orchestrator": "orchestrator", "reality_check": "reality_check"}),
    ):
        builder.add_conditional_edges(source, router, targets)
    builder.add_conditional_edges(
        "reality_check",
        route_after_reality_check,
        {"orchestrator": "orchestrator", "expert_review": "expert_review", "stop": END},
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
