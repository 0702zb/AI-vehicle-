"""Demo: orchestrator -> Layer Agent -> Body/Chassis tools, with/without RAG.

Run: python demo_layer_flow.py

The script runs the flow twice (RAG on / RAG off) and writes a concise
presentation log to layer_agent_demo.log.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langgraph.graph import END, START, StateGraph

from concept_platform.agents import MockProvider, ProviderAdapter, ProviderResponse
from concept_platform.graph import make_orchestrator_node
from concept_platform.layer_agents import HttpProviderAdapter, RAGHit, create_layer_agent_node
from concept_platform.state import NodeStatus, RunState


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).parent / ".env")

DEDICATED_API_KEY = os.environ.get("DEDICATED_API_KEY", "")
DEDICATED_BASE_URL = os.environ.get("DEDICATED_BASE_URL", "")
DEDICATED_MODEL_ID = os.environ.get("DEDICATED_MODEL_ID", "layer-model")
USE_REAL_LLM = bool(
    DEDICATED_API_KEY
    and DEDICATED_BASE_URL
    and "REPLACE_WITH" not in DEDICATED_API_KEY
)

CONCEPT = (
    "概念A：轻量城市通勤舱面向一线城市高频短途出行，强调短前后悬与长轴距带来的空间感受，"
    "外观以低风阻水滴形轮廓、隐藏式门把手和主动进气格栅形成简洁气质。车身结构侧重钢铝混合与可维修性，"
    "座舱采用二加一布局并保留少量物理按键以保证盲操作安全。动力与热管理围绕城市通勤的快充便利性和"
    "冬季续航稳定性展开；底盘需在城市低速工况下优化滤振与转向响应并保留机械冗余。"
    "工程上需在轻量化、风阻、碰撞安全、底盘舒适性与成本之间给出文字性权衡建议，"
    "关键结论应来自冻结概念或检索证据，不得凭空设定数值目标。"
)

HITS = (
    RAGHit(
        "ev-body-01", "demo-project", "v1", "sha256:demo",
        "低风阻轮廓与轻量化骨架的权衡：优先保证碰撞传力通道与可维修性。",
        domain="body",
    ),
    RAGHit(
        "ev-chassis-01", "demo-project", "v1", "sha256:demo",
        "城市工况底盘建议：强调低速滤振与转向响应，保留机械冗余。",
        domain="chassis",
    ),
)


class DemoRAG:
    """Domain-scoped RAG stub; records every retrieval for the demo log."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events

    def retrieve(
        self, query: str, *, top_k: int = 5, domain: str | None = None
    ) -> tuple[RAGHit, ...]:
        hits = tuple(hit for hit in HITS if domain is None or hit.domain == domain)[:top_k]
        self.events.append(
            {
                "stage": "rag",
                "domain": domain or "all",
                "hits": [hit.evidence_id for hit in hits],
            }
        )
        return hits


class TracingAdapter:
    """Keep only each model's JSON output for the presentation log."""

    def __init__(self, inner: ProviderAdapter, events: list[dict[str, Any]]) -> None:
        self.inner = inner
        self.events = events

    async def complete(self, request):
        response = await self.inner.complete(request)
        self.events.append(
            {
                "stage": "model_output",
                "agent_id": request.agent_id,
                "structured_output": response.structured_output,
            }
        )
        return response


def _base_proposal() -> dict[str, Any]:
    return {
        "project_id": "demo-project",
        "concept_id": "demo-concept",
        "concept_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "status": "supported",
        "summary": "",
        "evidence_refs": [],
    }


ROUTER_OUTPUT = {
    **_base_proposal(),
    "summary": "概念以文字描述为主，调用车身与底盘工具给出工程权衡建议。",
    "reasoning": "概念强调轻量通勤与城市工况，需要 body 与 chassis 细化文字性建议。",
    "tool_calls": [
        {"tool_name": "body", "arguments": {"focus": "低风阻轮廓与轻量化骨架的文字性权衡"}},
        {"tool_name": "chassis", "arguments": {"focus": "城市工况滤振与转向响应的文字性建议"}},
    ],
}

BODY_OUTPUT = {
    **_base_proposal(),
    "summary": (
        "车身设计以水滴形低风阻轮廓和隐藏式门把手为主，配合钢铝混合骨架按载荷路径选材。"
        "造型收益需要与模具、装配和维修可达性权衡；钢铝连接工艺与短前后悬的碰撞传力空间"
        "是主要工程约束。当前仅有文字描述，结论限于方向性可行，仍需碰撞与风洞验证。"
    ),
    "evidence_refs": [
        {
            "evidence_id": "ev-body-01",
            "project_id": "demo-project",
            "version": "v1",
            "context_manifest_hash": "sha256:demo",
        }
    ],
    "domain": "body",
    "layer_id": "body",
    "property_impacts": [
        {
            "property_id": "mass",
            "description": "钢铝混合骨架按载荷路径选材，理论上比全钢方案更利于减重。",
            "engineering_implications": "减重收益取决于连接工艺与加强件数量，需与碰撞传力要求一并确认。",
            "evidence_class": "ASSUMED",
            "source_ids": [],
            "confidence": "LOW",
        }
    ],
    "overall_assessment": "FEASIBLE_WITH_CONDITIONS",
}

CHASSIS_OUTPUT = {
    **_base_proposal(),
    "summary": (
        "底盘设计以悬架几何、阻尼与转向匹配改善城市低速滤振和转向响应，并保留制动与转向冗余。"
        "长轴距和轻量化车身改变了簧载质量分布，需要在舒适性与侧倾支撑之间取舍。"
        "冗余机构会带来布置和质量代价；没有确定性动力学结果时，结论仅为方向性建议。"
    ),
    "evidence_refs": [
        {
            "evidence_id": "ev-chassis-01",
            "project_id": "demo-project",
            "version": "v1",
            "context_manifest_hash": "sha256:demo",
        }
    ],
    "domain": "chassis",
    "layer_id": "chassis",
    "property_impacts": [
        {
            "property_id": "mass",
            "description": "增加阻尼元件与冗余执行机构会带来底盘质量上升方向的影响。",
            "engineering_implications": "质量增加会抵消部分轻量化收益，需结合确定性结果确认。",
            "evidence_class": "UNKNOWN",
            "source_ids": [],
            "confidence": "LOW",
        }
    ],
    "overall_assessment": "FEASIBLE_WITH_CONDITIONS",
}


def _mock_adapter(use_rag: bool) -> ProviderAdapter:
    body = deepcopy(BODY_OUTPUT)
    chassis = deepcopy(CHASSIS_OUTPUT)
    if not use_rag:
        body["evidence_refs"] = []
        chassis["evidence_refs"] = []
    return MockProvider(
        [
            ProviderResponse(structured_output=ROUTER_OUTPUT, model_version="layer-model-v1"),
            ProviderResponse(structured_output=body, model_version="layer-model-v1"),
            ProviderResponse(structured_output=chassis, model_version="layer-model-v1"),
        ]
    )


def _adapter(use_rag: bool, events: list[dict[str, Any]]) -> ProviderAdapter:
    inner: ProviderAdapter
    if USE_REAL_LLM:
        inner = HttpProviderAdapter(
            api_key=DEDICATED_API_KEY,
            base_url=DEDICATED_BASE_URL,
            model_id=DEDICATED_MODEL_ID,
        )
    else:
        inner = _mock_adapter(use_rag)
    return TracingAdapter(inner, events)


def _initial_state() -> RunState:
    return {
        "run_id": "run-tool-flow",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "node_status": {},
        "coverage": {"candidate_concepts": [CONCEPT], "selected_concept": CONCEPT},
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


async def run_flow(use_rag: bool) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    rag = DemoRAG(events) if use_rag else None

    def traced_orchestrator(state: RunState) -> dict[str, Any]:
        patch = make_orchestrator_node()(state)
        events.append({"stage": "orchestrator", "patch": patch})
        return patch

    def traced_route(state: RunState) -> str:
        decision = "layer_agent"
        events.append({"stage": "route", "decision": decision})
        return decision

    def traced_layer_route(state: RunState) -> str:
        status = state["node_status"].get("layer_agent")
        decision = "orchestrator" if status in (NodeStatus.FAIL, NodeStatus.BLOCKED) else END
        events.append({"stage": "layer_route", "decision": decision})
        return decision

    builder = StateGraph(RunState)
    builder.add_node("orchestrator", traced_orchestrator)
    builder.add_node("layer_agent", create_layer_agent_node(adapter=_adapter(use_rag, events), rag=rag))
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        traced_route,
        {"layer_agent": "layer_agent"},
    )
    builder.add_conditional_edges(
        "layer_agent",
        traced_layer_route,
        {"orchestrator": "orchestrator", END: END},
    )
    graph = builder.compile()
    result = await graph.ainvoke(_initial_state(), {"recursion_limit": 50})
    return {"events": events, "result": result}


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _slim(entry: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: entry[key] for key in keys if key in entry}


def _write_demo_log(runs: list[tuple[str, dict[str, Any]]]) -> None:
    sections = ["=== Layer Agent 工具调用演示 ==="]

    for title, payload in runs:
        events = payload["events"]
        result = payload["result"]
        outputs = {
            event["agent_id"]: event["structured_output"]
            for event in events
            if event["stage"] == "model_output"
        }
        orchestrator = next(event["patch"] for event in events if event["stage"] == "orchestrator")
        decision = next(event["decision"] for event in events if event["stage"] == "route")
        rag_hits = {
            event["domain"]: event["hits"]
            for event in events
            if event["stage"] == "rag"
        }
        router = _slim(outputs["layer_router"], ("status", "summary", "reasoning", "tool_calls"))
        trace = result["coverage"]["layer_agent_trace"]
        final = {
            "node_status": result["node_status"],
            "tool_calls": trace["tool_calls"],
            "tool_results": trace["tool_results"],
        }

        sections.extend(
            [
                "",
                f"=== 模式：{title} ===",
                "[1] Orchestrator 输出",
                _pretty(orchestrator),
                f"[2] 路由决策: orchestrator -> {decision}",
                "[3] Layer Agent（路由 LLM）输出 JSON",
                _pretty(router),
            ]
        )
        for domain, agent_id, label in (
            ("body", "tool_body", "车身"),
            ("chassis", "tool_chassis", "底盘"),
        ):
            if agent_id not in outputs:
                sections.extend(
                    [
                        f"[4.{domain}] {label}子 Agent RAG 命中: 未调用",
                        f"[5.{domain}] {label}子 Agent 输出 JSON",
                        _pretty({"status": "not_called"}),
                    ]
                )
                continue
            tool = _slim(
                outputs[agent_id],
                ("status", "domain", "summary", "evidence_refs", "directives"),
            )
            if "evidence_refs" in tool:
                tool["evidence_refs"] = [ref["evidence_id"] for ref in tool["evidence_refs"]]
            sections.extend(
                [
                    f"[4.{domain}] {label}子 Agent RAG 命中: "
                    + (", ".join(rag_hits.get(domain, [])) or "无"),
                    f"[5.{domain}] {label}子 Agent 输出 JSON",
                    _pretty(tool),
                ]
            )
        sections.extend(["[6] 最终结果摘要", _pretty(final)])

    Path("layer_agent_demo.log").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    runs = [
        ("RAG 开启", asyncio.run(run_flow(True))),
        ("RAG 关闭", asyncio.run(run_flow(False))),
    ]
    _write_demo_log(runs)
    print("wrote layer_agent_demo.log")
    for title, payload in runs:
        print(f"{title}: {payload['result']['node_status']}")


if __name__ == "__main__":
    main()
