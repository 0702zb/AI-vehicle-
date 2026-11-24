from __future__ import annotations

import asyncio

from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from concept_platform.concept_orchestrator import (
    ConceptStatus,
    build_concept_orchestrator,
    initial_state,
)
from concept_platform.state import NodeStatus


def _agent(status: NodeStatus, calls: list[str], name: str):
    async def run(state):
        calls.append(name)
        return {"status": status, "result_ids": [f"{name}:{state['concept_id']}"]}

    return run


def _state(concepts):
    return initial_state(
        run_id="run-orch",
        brief_version="brief-v1",
        context_manifest_hash="sha256:manifest",
        concepts=concepts,
    )


def test_five_concepts_run_in_parallel_and_aggregate() -> None:
    layer_calls: list[str] = []
    property_calls: list[str] = []
    reality_calls: list[str] = []
    graph = build_concept_orchestrator(
        layer_agent=_agent(NodeStatus.SUPPORTED, layer_calls, "layer"),
        property_agent=_agent(NodeStatus.SUPPORTED, property_calls, "property"),
        reality_agent=_agent(NodeStatus.SUPPORTED, reality_calls, "reality"),
    )

    result = asyncio.run(graph.ainvoke(_state([f"concept-{i}" for i in range(5)])))

    assert len(result["concept_portfolio"]) == 5
    assert len(layer_calls) == len(property_calls) == len(reality_calls) == 5
    assert all(r["status"] == ConceptStatus.COMPLETE for r in result["concept_portfolio"].values())
    assert all(r["node_status"]["layer_agent"] is NodeStatus.SUPPORTED
               for r in result["concept_portfolio"].values())


def test_failed_concept_does_not_advance_but_others_succeed() -> None:
    property_calls: list[str] = []

    # Concept 2 fails at layer; others pass.
    async def selective_layer(state):
        status = NodeStatus.FAIL if state["concept_id"] == "concept-2" else NodeStatus.SUPPORTED
        return {"status": status, "result_ids": []}

    graph = build_concept_orchestrator(
        layer_agent=selective_layer,
        property_agent=_agent(NodeStatus.SUPPORTED, property_calls, "property"),
        reality_agent=_agent(NodeStatus.SUPPORTED, [], "reality"),
    )

    result = asyncio.run(graph.ainvoke(_state([f"concept-{i}" for i in range(1, 6)])))

    assert result["concept_portfolio"]["concept-2"]["status"] == ConceptStatus.FAILED
    assert "property_agent" not in result["concept_portfolio"]["concept-2"]["node_status"]
    assert result["concept_portfolio"]["concept-1"]["status"] == ConceptStatus.COMPLETE
    assert len(property_calls) == 4


def test_human_selection_gate() -> None:
    graph = build_concept_orchestrator(
        layer_agent=_agent(NodeStatus.SUPPORTED, [], "layer"),
        property_agent=_agent(NodeStatus.SUPPORTED, [], "property"),
        reality_agent=_agent(NodeStatus.SUPPORTED, [], "reality"),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "select-run"}}

    interrupted = asyncio.run(graph.ainvoke(_state(["A", "B", "C"]), config))
    assert "__interrupt__" in interrupted

    result = asyncio.run(
        graph.ainvoke(Command(resume={"concept_id": "concept-2", "action": "select"}), config)
    )
    assert result["selection"]["concept_id"] == "concept-2"
    assert result["concept_portfolio"]["concept-2"]["status"] == ConceptStatus.SELECTED
    assert result["concept_portfolio"]["concept-1"]["status"] == ConceptStatus.REJECTED


def test_checkpoint_recovery_skips_completed_concepts() -> None:
    calls: list[str] = []
    saver = InMemorySaver()
    graph = build_concept_orchestrator(
        layer_agent=_agent(NodeStatus.SUPPORTED, calls, "layer"),
        property_agent=_agent(NodeStatus.SUPPORTED, calls, "property"),
        reality_agent=_agent(NodeStatus.SUPPORTED, calls, "reality"),
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "recovery-run"}}

    asyncio.run(graph.ainvoke(_state(["A", "B"]), config))
    assert len(calls) == 6  # 2 concepts x 3 stages

    snapshot = asyncio.run(graph.aget_state(config))
    portfolio = snapshot.values["concept_portfolio"]
    assert all(r["status"] == ConceptStatus.COMPLETE for r in portfolio.values())

    # Re-running on the same thread is blocked at the human gate (already interrupted).
    assert "__interrupt__" in asyncio.run(graph.ainvoke(None, config))
    assert len(calls) == 6  # no agent re-executed
