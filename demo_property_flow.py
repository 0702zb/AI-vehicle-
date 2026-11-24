"""Demo: orchestrator -> Layer Agent -> Property Agent -> Reality Check -> END.

Layer Agent and Property Agent return to orchestrator on FAIL/BLOCKED.

Run: python demo_property_flow.py

Layer Agent and Property Agent both call the API configured in .env by default.
The real LayerAssessment trace feeds the selected Property Rubric. The log is
property_agent_demo.log.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langgraph.graph import END, START, StateGraph

from concept_platform.agents import MockProvider, ProviderAdapter, ProviderRequest, ProviderResponse
from concept_platform.demo_reality import (
    build_demo_reality_check_node,
    route_after_demo_reality_check,
)
from concept_platform.graph import (
    make_orchestrator_node,
    route_after_layer_agent,
    route_after_property_agent,
)
from concept_platform.layer_agents import HttpProviderAdapter, RAGHit, create_layer_agent_node
from concept_platform.property_agent import create_property_agent_node
from concept_platform.state import RunState


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
DEDICATED_PROPERTY_MODEL_ID = os.environ.get(
    "DEDICATED_PROPERTY_MODEL_ID", DEDICATED_MODEL_ID
)
DEDICATED_PROPERTY_MODEL_VERSION = os.environ.get(
    "DEDICATED_PROPERTY_MODEL_VERSION", "property-model-v1"
)
FORCE_MOCK = os.environ.get("PROPERTY_DEMO_USE_MOCK", "").strip().lower() in {
    "1", "true", "yes"
}
USE_REAL_LLM = not FORCE_MOCK and bool(
    DEDICATED_API_KEY
    and DEDICATED_BASE_URL
    and "REPLACE_WITH" not in DEDICATED_API_KEY
)
LAYER_FORCE_MOCK = os.environ.get(
    "PROPERTY_DEMO_LAYER_USE_MOCK", ""
).strip().lower() in {"1", "true", "yes"}
LAYER_USE_REAL_LLM = USE_REAL_LLM and not LAYER_FORCE_MOCK

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
    def __init__(
        self, inner: ProviderAdapter, events: list[dict[str, Any]], stage: str
    ) -> None:
        self.inner = inner
        self.events = events
        self.stage = stage

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        response = await self.inner.complete(request)
        self.events.append(
            {
                "stage": self.stage,
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
            "evidence_class": "INFERRED",
            "source_ids": ["ev-body-01"],
            "engineering_implications": "轻量化与非轻量化材料选择会影响整车质量。",
            "confidence": "MEDIUM",
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
            "description": "冗余机构与阻尼元件会带来底盘质量上升方向的影响。",
            "evidence_class": "INFERRED",
            "source_ids": ["ev-chassis-01"],
            "engineering_implications": "冗余方案与底盘结构变化可能带来质量影响。",
            "confidence": "MEDIUM",
        }
    ],
    "overall_assessment": "FEASIBLE_WITH_CONDITIONS",
}

PROPERTY_OUTPUT = {
    **_base_proposal(),
    "summary": "汇总 body 与 chassis 的 mass impact 记录，形成整车质量属性判断。",
    "property_id": "mass",
    "evaluated_layers": ["body", "chassis"],
    "aggregation_summary": "两个领域均给出定性质量影响，尚无确定性质量计算结果。",
    "target_comparison": "缺少注册质量目标和确定性计算结果，不能判定达标。",
    "rubric_result": "INSUFFICIENT_INFORMATION",
    "confidence": "LOW",
    "missing_information": [
        "缺少注册的质量目标值及其来源版本。",
        "缺少确定性质量计算结果 artifact 及其适用条件。",
        "缺少与质量相关的冻结证据或检索命中。",
    ],
    "properties": [
        {
            "property_id": "prop-aero-mass",
            "name": "低风阻与轻量化结构权衡",
            "status": "supported",
            "source_domains": ["body"],
            "basis": ["用文字说明低风阻轮廓与轻量化骨架之间的取舍"],
            "rationale": "车身 directive 直接要求说明低风阻轮廓与轻量化骨架的取舍。",
            "validation_items": ["检查低风阻轮廓与轻量化骨架的取舍是否被明确描述。"],
        },
        {
            "property_id": "prop-repair-safety",
            "name": "碰撞传力与可维修性",
            "status": "supported",
            "source_domains": ["body"],
            "basis": ["补充碰撞传力通道与可维修性的工程建议"],
            "rationale": "车身 directive 要求补充碰撞传力通道与可维修性建议。",
            "validation_items": ["检查碰撞传力通道与可维修性是否同时被说明。"],
        },
        {
            "property_id": "prop-ride-steer",
            "name": "城市工况滤振与转向响应",
            "status": "supported",
            "source_domains": ["chassis"],
            "basis": ["描述城市工况下滤振与转向响应的权衡，保留机械冗余方案"],
            "rationale": "底盘 directive 直接给出低速滤振、转向响应与冗余要求。",
            "validation_items": ["检查滤振、转向响应与机械冗余是否均有文字性说明。"],
        },
        {
            "property_id": "prop-thermal-interface",
            "name": "热管理与底盘布置接口",
            "status": "supported",
            "source_domains": ["chassis"],
            "basis": ["说明热管理与底盘布置的接口关系"],
            "rationale": "底盘 directive 要求说明热管理与底盘布置的接口关系。",
            "validation_items": ["检查热管理与底盘布置接口是否被描述且不编造参数。"],
        },
    ],
}


def _layer_adapter(use_rag: bool, events: list[dict[str, Any]]) -> ProviderAdapter:
    body = deepcopy(BODY_OUTPUT)
    chassis = deepcopy(CHASSIS_OUTPUT)
    if not use_rag:
        body["evidence_refs"] = []
        chassis["evidence_refs"] = []
    if LAYER_USE_REAL_LLM:
        inner: ProviderAdapter = HttpProviderAdapter(
            api_key=DEDICATED_API_KEY,
            base_url=DEDICATED_BASE_URL,
            model_id=DEDICATED_MODEL_ID,
            model_version="layer-model-v1",
        )
    else:
        inner = MockProvider(
            [
                ProviderResponse(structured_output=ROUTER_OUTPUT, model_version="layer-model-v1"),
                ProviderResponse(structured_output=body, model_version="layer-model-v1"),
                ProviderResponse(structured_output=chassis, model_version="layer-model-v1"),
            ]
        )
    return TracingAdapter(inner, events, "layer_output")


def _property_adapter(events: list[dict[str, Any]]) -> ProviderAdapter:
    if USE_REAL_LLM:
        inner: ProviderAdapter = HttpProviderAdapter(
            api_key=DEDICATED_API_KEY,
            base_url=DEDICATED_BASE_URL,
            model_id=DEDICATED_PROPERTY_MODEL_ID,
            model_version=DEDICATED_PROPERTY_MODEL_VERSION,
        )
    else:
        inner = MockProvider(
            [
                ProviderResponse(
                    structured_output=deepcopy(PROPERTY_OUTPUT),
                    model_version=DEDICATED_PROPERTY_MODEL_VERSION,
                )
            ]
        )
    return TracingAdapter(inner, events, "property_output")


def _initial_state() -> RunState:
    return {
        "run_id": "run-property-flow",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "node_status": {},
        "coverage": {"candidate_concepts": [CONCEPT], "selected_concept": CONCEPT},
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


def build_demo_graph(use_rag: bool = False, events: list[dict[str, Any]] | None = None):
    """Build the 4-node demo graph with mock or real adapters."""
    if events is None:
        events = []
    orchestrator_node = make_orchestrator_node()

    def traced_orchestrator(state: RunState) -> dict[str, Any]:
        patch = orchestrator_node(state)
        events.append({"stage": "orchestrator", "patch": patch})
        return patch

    def traced_route(state: RunState) -> str:
        events.append({"stage": "route", "decision": "layer_agent"})
        return "layer_agent"

    def traced_layer_route(state: RunState) -> str:
        decision = route_after_layer_agent(state)
        events.append({"stage": "layer_route", "decision": decision})
        return decision

    def traced_property_route(state: RunState) -> str:
        decision = route_after_property_agent(state)
        events.append({"stage": "property_route", "decision": decision})
        return decision

    builder = StateGraph(RunState)
    builder.add_node("orchestrator", traced_orchestrator)
    builder.add_node(
        "layer_agent",
        create_layer_agent_node(
            adapter=_layer_adapter(use_rag, events),
            rag=DemoRAG(events) if use_rag else None,
        ),
    )
    builder.add_node(
        "property_agent",
        create_property_agent_node(
            adapter=_property_adapter(events),
            property_id="mass",
            model_id=DEDICATED_PROPERTY_MODEL_ID,
            model_version=DEDICATED_PROPERTY_MODEL_VERSION,
        ),
    )
    reality_check = build_demo_reality_check_node()

    async def traced_reality_check(state: RunState) -> dict[str, Any]:
        patch = await reality_check(state)
        events.append({"stage": "reality_check", "patch": patch})
        return patch

    builder.add_node("reality_check", traced_reality_check)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator", traced_route, {"layer_agent": "layer_agent"}
    )
    builder.add_conditional_edges(
        "layer_agent",
        traced_layer_route,
        {"orchestrator": "orchestrator", "property_agent": "property_agent"},
    )
    builder.add_conditional_edges(
        "property_agent",
        traced_property_route,
        {"orchestrator": "orchestrator", "reality_check": "reality_check"},
    )
    builder.add_conditional_edges(
        "reality_check",
        route_after_demo_reality_check,
        {END: END, "orchestrator": "orchestrator"},
    )
    return builder.compile()


async def run_flow(use_rag: bool, state: RunState | None = None) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    graph = build_demo_graph(use_rag=use_rag, events=events)
    result = await graph.ainvoke(state or _initial_state(), {"recursion_limit": 50})
    return {"events": events, "result": result}


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _slim(entry: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: entry[key] for key in keys if key in entry}


def _write_demo_log(runs: list[tuple[str, dict[str, Any]]]) -> None:
    property_provider = "API" if USE_REAL_LLM else "mock"
    layer_provider = "API" if LAYER_USE_REAL_LLM else "mock"
    sections = [
        "=== Property Agent 演示：orchestrator -> layer_agent -> property_agent -> reality_check -> END ===",
        f"provider: property_agent={property_provider}, layer_agent={layer_provider}",
        "property_rubric: mass",
    ]
    statuses: dict[str, str] = {}

    for title, payload in runs:
        events = payload["events"]
        result = payload["result"]
        orchestrator_patch = next(
            event["patch"] for event in events if event["stage"] == "orchestrator"
        )
        route_decision = next(
            event["decision"] for event in events if event["stage"] == "route"
        )
        layer_outputs = {
            event["agent_id"]: event["structured_output"]
            for event in events
            if event["stage"] == "layer_output"
        }
        router_output = layer_outputs["layer_router"]
        property_output = next(
            event["structured_output"] for event in events if event["stage"] == "property_output"
        )
        layer_trace = result["coverage"]["layer_agent_trace"]

        tool_calls = layer_trace.get("tool_calls", ())
        called_tools = [call.get("tool_name") for call in tool_calls]
        tool_call_status = {
            call.get("tool_name"): call.get("status")
            for call in tool_calls
            if isinstance(call, Mapping)
        }
        layer_assessments = {}
        for domain, item in layer_trace["tool_results"].items():
            output = layer_outputs.get(f"tool_{domain}", {})
            layer_assessments[domain] = _slim(
                output,
                (
                    "layer_id",
                    "status",
                    "summary",
                    "property_impacts",
                    "overall_assessment",
                ),
            )
            layer_assessments[domain]["tool_call_status"] = tool_call_status.get(domain)

        property_trace = result["coverage"]["property_agent_trace"]
        statuses[title] = property_trace["status"]

        sections.extend(
            [
                "",
                f"=== Layer RAG：{title} ===",
                "[1] 流程",
                "START -> orchestrator -> layer_agent -> property_agent -> reality_check -> END "
                "(FAIL/BLOCKED 返回 orchestrator)",
                "[2] Orchestrator 输出",
                _pretty(orchestrator_patch),
                "[3] 路由决策",
                f"orchestrator -> {route_decision}",
                "[4] Layer Router 输出（工具调用决策）",
                _pretty(
                    _slim(router_output, ("status", "summary", "reasoning", "tool_calls"))
                ),
                "[5] 工具调用与 LayerAssessment 输出",
                "called_tools: " + ", ".join(str(name) for name in called_tools),
                _pretty(layer_assessments),
                "[6] PropertyVerdict 模型输出 JSON",
                _pretty(
                    _slim(
                        property_output,
                        (
                            "status",
                            "summary",
                            "property_id",
                            "rubric_result",
                            "confidence",
                            "aggregation_summary",
                            "target_comparison",
                            "properties",
                        ),
                    )
                ),
                "[7] PropertyVerdict 归一化 trace",
                _pretty(property_trace),
                "[8] Reality Check（确定性规则，synthetic demo fixture）",
                _pretty(
                    {
                        "decision": result["coverage"].get("reality_check_decision"),
                        "node_status": result["node_status"].get("reality_check"),
                        "result_ids": result["result_ids"],
                    }
                ),
                "[9] 最终 node_status",
                _pretty(result["node_status"]),
            ]
        )

    sections.extend(
        [
            "",
            "[10] 各 RAG 模式下的 property_agent 状态",
            _pretty(statuses),
        ]
    )
    Path("property_agent_demo.log").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    compare_rag = os.environ.get(
        "PROPERTY_DEMO_COMPARE_RAG", ""
    ).strip().lower() in {"1", "true", "yes"}
    runs = (
        [("开启", asyncio.run(run_flow(True))), ("关闭", asyncio.run(run_flow(False)))]
        if compare_rag
        else [("关闭", asyncio.run(run_flow(False)))]
    )
    _write_demo_log(runs)
    print("wrote property_agent_demo.log")
    for title, payload in runs:
        print(f"Layer RAG {title}: {payload['result']['node_status']}")


if __name__ == "__main__":
    main()
