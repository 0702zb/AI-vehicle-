"""Deterministic multi-concept orchestrator built on LangGraph.

Design boundary:
  * The orchestrator never scores or ranks concepts and never calls an LLM.
  * It owns state, fan-out, aggregation and the human gate only.
  * Layer / property / reality agents are injected callables; they do the work.

LangGraph capabilities used here:
  1. ``Send`` dynamic fan-out       - one independent task per concept
  2. reducer aggregation           - merge_mappings keyed by concept_id
  3. checkpoint recovery           - only PENDING concepts are re-dispatched
  4. ``interrupt`` human-in-loop    - post-evaluation selection by a person
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from concept_platform.state import NodeStatus, merge_mappings, merge_unique_ids

ConceptAgent = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


class ConceptStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SELECTED = "selected"
    REJECTED = "rejected"


class ConceptRecord(TypedDict):
    concept_id: str
    text: str
    status: str
    node_status: dict[str, NodeStatus]
    result_ids: list[str]
    error: str | None
    analysis: dict[str, Any]


def _extract_analysis(coverage: Mapping[str, Any]) -> dict[str, Any]:
    """Project each agent's actual output from state.coverage for display.

    Agents write into state (never return to the orchestrator). This function reads
    those writes so the UI can show what layer / property / reality actually produced.
    """
    analysis: dict[str, Any] = {"layers": [], "property": {}, "reality": {}}

    layer_trace = coverage.get("layer_agent_trace")
    if isinstance(layer_trace, Mapping):
        results = layer_trace.get("tool_results") or {}
        if isinstance(results, Mapping):
            for domain, result in results.items():
                if isinstance(result, Mapping):
                    analysis["layers"].append({
                        "domain": str(domain),
                        "summary": str(result.get("summary") or ""),
                        "status": str(result.get("status") or ""),
                        "evidence_refs": [str(r) for r in (result.get("evidence_refs") or [])],
                    })

    prop_trace = coverage.get("property_agent_trace")
    if isinstance(prop_trace, Mapping):
        analysis["property"] = {
            "property_id": str(prop_trace.get("property_id") or ""),
            "rubric_result": str(prop_trace.get("rubric_result") or ""),
            "aggregation_summary": str(prop_trace.get("aggregation_summary") or ""),
            "target_comparison": str(prop_trace.get("target_comparison") or ""),
            "properties": [
                {
                    "name": str(item.get("name") or ""),
                    "status": str(item.get("status") or ""),
                    "rationale": str(item.get("rationale") or ""),
                }
                for item in (prop_trace.get("properties") or [])
                if isinstance(item, Mapping)
            ],
        }

    analysis["reality"] = {
        "decision": str(coverage.get("reality_check_decision") or ""),
        "evaluations": [
            {
                "rule_id": str(item.get("rule_id") or ""),
                "decision": str(item.get("decision") or ""),
                "actual_value": str(item.get("actual_value") or ""),
                "threshold": str(item.get("threshold") or ""),
                "reason": str(item.get("reason") or ""),
            }
            for item in (coverage.get("reality_check_evaluations") or [])
            if isinstance(item, Mapping)
        ],
    }
    return analysis


class ConceptTask(TypedDict):
    concept_id: str
    text: str
    run_id: str
    brief_version: str
    context_manifest_hash: str


class ConceptChainState(TypedDict):
    concept_id: str
    text: str
    run_id: str
    brief_version: str
    context_manifest_hash: str
    node_status: Annotated[dict[str, NodeStatus], merge_mappings]
    result_ids: Annotated[list[str], merge_unique_ids]
    coverage: Annotated[dict[str, Any], merge_mappings]
    error: str | None


class ConceptState(TypedDict):
    run_id: str
    brief_version: str
    context_manifest_hash: str
    concepts: list[str]
    concept_portfolio: Annotated[dict[str, ConceptRecord], merge_mappings]
    selection: dict[str, Any]


def _new_record(concept_id: str, text: str) -> ConceptRecord:
    return {
        "concept_id": concept_id,
        "text": text,
        "status": ConceptStatus.PENDING,
        "node_status": {},
        "result_ids": [],
        "error": None,
        "analysis": {"layers": [], "property": {}, "reality": {}},
    }


def make_concept_subgraph(
    layer_agent: ConceptAgent,
    property_agent: ConceptAgent,
    reality_agent: ConceptAgent,
):
    """Deterministic per-concept chain: layer -> property -> reality.

    Stage order is fixed in the graph, not decided by any model. Failure short-circuits
    to END so a bad concept cannot silently advance to the next stage.
    """

    def make_agent_node(node_id: str, agent: ConceptAgent):
        """Uniform node wrapper: agents write state, never return results upward."""
        async def run(state: ConceptChainState) -> dict[str, Any]:
            outcome = await agent(state)
            return {
                "node_status": {node_id: outcome["status"]},
                "result_ids": outcome.get("result_ids", []),
                "coverage": outcome.get("coverage", {}),
            }
        return run

    def route_after(node_id: str):
        def route(state: ConceptChainState) -> str:
            status = state["node_status"].get(node_id)
            return "stop" if status in (NodeStatus.FAIL, NodeStatus.BLOCKED) else "next"

        return route

    builder = StateGraph(ConceptChainState)
    builder.add_node("layer_agent", make_agent_node("layer_agent", layer_agent))
    builder.add_node("property_agent", make_agent_node("property_agent", property_agent))
    builder.add_node("reality_check", make_agent_node("reality_check", reality_agent))
    builder.add_edge(START, "layer_agent")
    builder.add_conditional_edges(
        "layer_agent",
        route_after("layer_agent"),
        {"next": "property_agent", "stop": END},
    )
    builder.add_conditional_edges(
        "property_agent",
        route_after("property_agent"),
        {"next": "reality_check", "stop": END},
    )
    builder.add_edge("reality_check", END)
    return builder.compile()


def build_concept_orchestrator(
    *,
    layer_agent: ConceptAgent,
    property_agent: ConceptAgent,
    reality_agent: ConceptAgent,
    checkpointer: Any = None,
):
    """Build the orchestrator graph.

    Returns the compiled graph. The caller supplies concept texts via
    ``state["concepts"]``-style input handled by ``initial_state`` below.
    """

    chain = make_concept_subgraph(layer_agent, property_agent, reality_agent)

    async def intake(state: ConceptState) -> dict[str, Any]:
        concepts: Sequence[str] = state["concepts"]
        portfolio = {f"concept-{index + 1}": _new_record(f"concept-{index + 1}", text)
                     for index, text in enumerate(concepts)}
        return {"concept_portfolio": portfolio}

    def dispatch(state: ConceptState):
        """LangGraph Send fan-out - only concepts not already completed."""
        sends = [
            Send("evaluate_concept", {
                "concept_id": concept_id,
                "text": record["text"],
                "run_id": state["run_id"],
                "brief_version": state["brief_version"],
                "context_manifest_hash": state["context_manifest_hash"],
            })
            for concept_id, record in state["concept_portfolio"].items()
            if record["status"] in (ConceptStatus.PENDING, ConceptStatus.RUNNING)
        ]
        return sends or END

    async def evaluate_concept(task: ConceptTask) -> dict[str, Any]:
        """Run one isolated concept chain; write result into keyed reducer."""
        result = await chain.ainvoke({
            **dict(task),
            "coverage": {"selected_concept": task["text"]},
        })
        statuses = result["node_status"]
        failed = any(s in (NodeStatus.FAIL, NodeStatus.BLOCKED) for s in statuses.values())
        record: ConceptRecord = {
            "concept_id": task["concept_id"],
            "text": task["text"],
            "status": ConceptStatus.FAILED if failed else ConceptStatus.COMPLETE,
            "node_status": statuses,
            "result_ids": result["result_ids"],
            "error": result.get("error"),
            "analysis": _extract_analysis(result.get("coverage") or {}),
        }
        return {"concept_portfolio": {task["concept_id"]: record}}

    def aggregate(state: ConceptState) -> dict[str, Any]:
        """Pure aggregation: count outcomes, never rank or score."""
        statuses = [r["status"] for r in state["concept_portfolio"].values()]
        return {
            "selection": {
                "phase": "post_evaluation",
                "complete_count": statuses.count(ConceptStatus.COMPLETE),
                "failed_count": statuses.count(ConceptStatus.FAILED),
            }
        }

    def human_selection(state: ConceptState) -> dict[str, Any]:
        """Post-evaluation human gate via LangGraph interrupt."""
        options = [
            {
                "concept_id": record["concept_id"],
                "text": record["text"],
                "status": record["status"],
            }
            for record in state["concept_portfolio"].values()
            if record["status"] is ConceptStatus.COMPLETE
        ]
        response = interrupt(
            {
                "kind": "concept_selection",
                "options": options,
                "instruction": "Select one concept to continue, or reject to restart.",
            }
        )
        if not isinstance(response, Mapping) or "concept_id" not in response:
            raise ValueError("Selection resume must contain concept_id.")
        chosen = str(response["concept_id"])
        if chosen not in state["concept_portfolio"]:
            raise ValueError("Selected concept_id is not in the portfolio.")
        if state["concept_portfolio"][chosen]["status"] != ConceptStatus.COMPLETE:
            raise ValueError("Only a COMPLETE concept can be selected.")

        portfolio_update = {
            cid: {
                **record,
                "status": ConceptStatus.SELECTED if cid == chosen else ConceptStatus.REJECTED,
            }
            for cid, record in state["concept_portfolio"].items()
        }
        return {
            "concept_portfolio": portfolio_update,
            "selection": {
                **state.get("selection", {}),
                "concept_id": chosen,
                "action": str(response.get("action", "select")),
            },
        }

    builder = StateGraph(ConceptState)
    builder.add_node("intake", intake)
    builder.add_node("evaluate_concept", evaluate_concept)
    builder.add_node("aggregate", aggregate)
    builder.add_node("human_selection", human_selection)
    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake", dispatch, {"evaluate_concept": "evaluate_concept", END: END}
    )
    builder.add_edge("evaluate_concept", "aggregate")
    builder.add_edge("aggregate", "human_selection")
    builder.add_edge("human_selection", END)
    return builder.compile(checkpointer=checkpointer)


def initial_state(
    *,
    run_id: str,
    brief_version: str,
    context_manifest_hash: str,
    concepts: Sequence[str],
) -> dict[str, Any]:
    """Build the input state for the orchestrator graph."""
    return {
        "run_id": run_id,
        "brief_version": brief_version,
        "context_manifest_hash": context_manifest_hash,
        "concepts": list(concepts),
        "concept_portfolio": {},
        "selection": {},
    }
