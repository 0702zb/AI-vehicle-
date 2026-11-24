"""Layer agent with domain tool agents (Body/Chassis/Drive/Energy/EE/Thermal/Interior).

The concept is sent to the layer agent once. The layer agent is an LLM that
emits tool calls; each tool is a domain agent sharing a common rule set plus
domain-specific refinement. Only Body and Chassis are implemented here; the
other domains are registered as reserved tool names.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from concept_platform.agents import (
    AgentDefinition,
    AgentExecutionResult,
    AgentExecutor,
    AgentInput,
    AgentProposal,
    AgentRegistry,
    AgentTaskContext,
    ProviderAdapter,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
)
from concept_platform.assessment_schemas import LayerAssessment
from concept_platform.config import Environment
from concept_platform.engineering_knowledge import DOMAIN_PACKS
from concept_platform.prompts import (
    LAYER_SHARED_SYSTEM_PROMPT,
    SCOPE_ENVELOPE_RULE,
    render_domain_pack,
)
from concept_platform.state import NodeStatus, RunState

LAYER_AGENT_VERSION = "layer-v1"
DEFAULT_PROVIDER_ID = "layer-llm"
DEFAULT_MODEL_ID = "layer-model"
DEFAULT_MODEL_VERSION = "layer-model-v1"
ROUTER_AGENT_ID = "layer_router"

# One shared rule set for every domain tool agent.
SHARED_RULES = (
    "所有领域工具只生成本领域架构报告：描述设计、组件与本领域的工程取舍；"
    "只基于已冻结的概念文本与 RAG 检索到的证据；"
    "不得编造数值目标，概念或证据中确有价值时必须引用来源并说明适用条件；"
    "rag_context 为空时 evidence_refs 必须为空数组；"
    "输出必须是一个简洁 JSON：summary 写唯一的架构报告，property_impacts 至少一条；"
    "不得引用、比较或推断其它领域，也不得引用、假设或索取其它子 Agent 的输出。"
)

RESERVED_TOOLS = ("drive", "ee", "thermal", "interior")

# Higher value means a worse outcome. Used only to reduce tool results to one node status.
_STATUS_SEVERITY = {
    NodeStatus.SUPPORTED: 0,
    NodeStatus.ASSUMPTION: 1,
    NodeStatus.UNKNOWN: 2,
    NodeStatus.BLOCKED: 3,
    NodeStatus.FAIL: 4,
}


def build_task_context(
    state: RunState, allowed_evidence: dict[str, str] | None = None
) -> AgentTaskContext:
    """Build the task boundary from frozen state, not hardcoded demo IDs."""
    return AgentTaskContext(
        project_id=str(state["coverage"].get("project_id", "demo-project")),
        concept_id=str(state["coverage"].get("concept_id", "demo-concept")),
        concept_version=state["brief_version"],
        context_manifest_hash=state["context_manifest_hash"],
        allowed_evidence=allowed_evidence or {},
        allowed_models={},
    )


def _blocked(source: str) -> dict[str, Any]:
    return {
        "node_status": {"layer_agent": NodeStatus.BLOCKED},
        "coverage": {"errors": [{"source": source, "status": "BLOCKED"}]},
    }


@dataclass(frozen=True)
class RAGHit:
    evidence_id: str
    project_id: str
    version: str
    context_manifest_hash: str
    text: str
    domain: str = ""


class RAGProvider(Protocol):
    """Domain-scoped retrieval hook for frozen engineering evidence."""

    def retrieve(
        self, query: str, *, top_k: int = 5, domain: str | None = None
    ) -> tuple[RAGHit, ...]: ...


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class LayerRouterInput(AgentInput):
    concept: str
    available_tools: list[str]
    rag_context: list[dict[str, Any]] = Field(default_factory=list)


class LayerRouterProposal(AgentProposal):
    reasoning: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolAgentInput(AgentInput):
    concept: str
    focus: str = ""
    rag_context: list[dict[str, Any]] = Field(default_factory=list)


ToolAgentProposal = LayerAssessment


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    agent_id: str
    description: str


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "body": ToolDefinition(
        "body", "tool_body", "车身结构与空气动力学设计工具"
    ),
    "chassis": ToolDefinition(
        "chassis", "tool_chassis", "悬架、转向、制动与动力学设计工具"
    ),
    "energy": ToolDefinition(
        "energy", "tool_energy", "电池、高压架构、充电与能量管理工具"
    ),
}

LAYER_ROUTER_SYSTEM_INSTRUCTION = (
    "你是 Layer Router。概念只输入一次。"
    "仅根据概念文字与工具职责决定调用哪些领域工具；证据检索由各子 Agent 自行完成。"
    "可选工具：" + ", ".join(TOOL_DEFINITIONS) + "。"
    "每个工具由其 Domain Pack 定义职责；未注册领域不得调用；低风阻外观属于 body。\n"
    "reasoning 必须给出工程选择依据：说明为何调用这些领域、哪些概念线索触发了它们，"
    "以及为何其它领域不需要本次分析，不得只复述概念。\n"
    "每个 tool_calls.arguments.focus 必须是具体的本领域架构关注点（含对象与权衡方向），"
    "不得只写领域名称，不得要求跨域比较或其它领域信息，不得编造数值参数。\n"
    + SCOPE_ENVELOPE_RULE
)


class HttpProviderAdapter(ProviderAdapter):
    """OpenAI-compatible /chat/completions adapter for structured JSON output."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str = DEFAULT_MODEL_ID,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        self.api_key = api_key.strip().strip('"').strip("'")
        url = base_url.strip().strip('"').strip("'").rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        self.base_url = url
        self.model_id = model_id.strip().strip('"').strip("'")
        self.model_version = model_version

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Call an OpenAI-compatible /chat/completions endpoint for structured JSON."""

        payload: dict[str, Any] = {
            "task_input": request.task_input,
            "task_context": request.task_context,
            "output_schema": request.output_schema,
            "tool_allowlist": list(request.tool_allowlist),
            "token_budget": request.token_budget,
        }
        if request.repair_errors:
            payload["repair_errors"] = list(request.repair_errors)
            payload["repair_candidate"] = request.repair_candidate

        messages = [
            {
                "role": "system",
                "content": (
                    request.system_instruction
                    + "\n只输出一个 JSON 对象，且必须符合下面的 JSON Schema：\n"
                    + json.dumps(request.output_schema, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": request.token_budget,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.base_url, headers=headers, json=body)
            if response.status_code == 400 and "response_format" in body:
                body.pop("response_format")
                response = await client.post(self.base_url, headers=headers, json=body)
            if response.is_error:
                raise RuntimeError(
                    f"LLM HTTP {response.status_code}: {response.text}"
                ) from None
            content = response.json()["choices"][0]["message"]["content"]

        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        return ProviderResponse(
            structured_output=json.loads(text),
            model_version=self.model_version,
        )


def _definition(
    *,
    agent_id: str,
    input_schema: type[BaseModel],
    output_schema: type[AgentProposal],
    system_instruction: str,
    provider_id: str,
    model_id: str,
    model_version: str,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        version=LAYER_AGENT_VERSION,
        input_schema=input_schema,
        output_schema=output_schema,
        prompt_template_version="layer-prompt-v3",
        system_instruction=system_instruction,
        provider_id=provider_id,
        model_id=model_id,
        model_version=model_version,
        tool_allowlist=frozenset(TOOL_DEFINITIONS),
        token_budget=4096,
    )


def create_layer_agent_node(
    *,
    adapter: ProviderAdapter | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider_id: str = DEFAULT_PROVIDER_ID,
    model_id: str = DEFAULT_MODEL_ID,
    model_version: str = DEFAULT_MODEL_VERSION,
    rag: RAGProvider | None = None,
) -> Callable[[RunState], dict[str, Any]]:
    """Build the layer_agent node.

    Pass ``adapter`` for local mocks, or pass ``api_key``/``base_url`` to use the
    real-LLM entry point. ``rag`` injects frozen engineering evidence into both
    the router and the domain tool agents. The returned node is provider-neutral.
    """

    if adapter is None:
        if not api_key or not base_url:
            raise ValueError("Provide adapter, or both api_key and base_url.")
        adapter = HttpProviderAdapter(
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
            model_version=model_version,
        )

    agents = AgentRegistry()
    providers = ProviderRegistry(Environment.DEV)
    providers.register(provider_id, model_id, model_version, adapter)
    executor = AgentExecutor(agents, providers)

    agents.register(
        _definition(
            agent_id=ROUTER_AGENT_ID,
            input_schema=LayerRouterInput,
            output_schema=LayerRouterProposal,
            system_instruction=LAYER_ROUTER_SYSTEM_INSTRUCTION,
            provider_id=provider_id,
            model_id=model_id,
            model_version=model_version,
        )
    )
    for name, tool in TOOL_DEFINITIONS.items():
        agents.register(
            _definition(
                agent_id=tool.agent_id,
                input_schema=ToolAgentInput,
                output_schema=ToolAgentProposal,
                system_instruction=(
                    LAYER_SHARED_SYSTEM_PROMPT
                    + "\n"
                    + render_domain_pack(DOMAIN_PACKS[name])
                    + "\n"
                    + SHARED_RULES
                    + "\nlayer_id 必须等于 Domain Pack 的 domain_id。"
                    + "property_impacts.property_id 只能使用 mass、package_space、"
                    + "efficiency_range、cost_mek。"
                ),
                provider_id=provider_id,
                model_id=model_id,
                model_version=model_version,
            )
        )

    async def run_router(state: RunState, concept: str) -> AgentExecutionResult:
        return await executor.execute(
            agent_id=ROUTER_AGENT_ID,
            agent_version=LAYER_AGENT_VERSION,
            task_input={
                "concept": concept,
                "available_tools": list(TOOL_DEFINITIONS),
                "rag_context": [],
            },
            context=build_task_context(state),
        )

    async def run_tool(
        state: RunState, concept: str, call: ToolCall
    ) -> tuple[dict[str, Any], dict[str, Any] | None, NodeStatus]:
        tool = TOOL_DEFINITIONS.get(call.tool_name)
        if tool is None:
            return (
                {"tool_name": call.tool_name, "arguments": call.arguments, "status": "unknown_tool"},
                None,
                NodeStatus.BLOCKED,
            )

        focus = str(call.arguments.get("focus", ""))
        hits = rag.retrieve(f"{concept}\n{focus}", domain=call.tool_name) if rag else ()
        rag_context = [
            {
                "evidence_id": hit.evidence_id,
                "project_id": hit.project_id,
                "version": hit.version,
                "context_manifest_hash": hit.context_manifest_hash,
                "text": hit.text,
                "domain": hit.domain or call.tool_name,
            }
            for hit in hits
        ]
        result = await executor.execute(
            agent_id=tool.agent_id,
            agent_version=LAYER_AGENT_VERSION,
            task_input={"concept": concept, "focus": focus, "rag_context": rag_context},
            context=build_task_context(
                state, allowed_evidence={hit.evidence_id: hit.version for hit in hits}
            ),
        )
        trace: dict[str, Any] | None = None
        if isinstance(result.proposal, ToolAgentProposal):
            assessment = result.proposal.model_dump(mode="json", exclude_unset=True)
            trace = {
                "status": result.status.value,
                "summary": result.proposal.summary,
                "directives": [],
                "evidence_refs": [ref.evidence_id for ref in result.proposal.evidence_refs],
                "rag_context": rag_context,
                "assessment": assessment,
            }
        return (
            {"tool_name": call.tool_name, "arguments": call.arguments, "status": result.status.value},
            trace,
            result.status,
        )

    async def layer_agent_node(state: RunState) -> dict[str, Any]:
        concept = state["coverage"].get("selected_concept")
        if not concept:
            return _blocked("layer_agent")

        router = await run_router(state, concept)
        if router.status is NodeStatus.BLOCKED or not isinstance(router.proposal, LayerRouterProposal):
            return _blocked("layer_router")

        tool_calls: list[dict[str, Any]] = []
        tool_results: dict[str, Any] = {}
        statuses: list[NodeStatus] = []
        for call in router.proposal.tool_calls:
            call_trace, result_trace, call_status = await run_tool(state, concept, call)
            tool_calls.append(call_trace)
            statuses.append(call_status)
            if result_trace is not None:
                tool_results[call.tool_name] = result_trace

        status = max(statuses, key=_STATUS_SEVERITY.__getitem__) if statuses else NodeStatus.BLOCKED
        return {
            "node_status": {"layer_agent": status},
            "coverage": {
                "layer_agent_trace": {
                    "reasoning": router.proposal.reasoning,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                },
                "layer_agent_router_output": {
                    "status": router.proposal.status.value,
                    "summary": router.proposal.summary,
                    "reasoning": router.proposal.reasoning,
                    "tool_calls": [
                        call.model_dump(mode="json") for call in router.proposal.tool_calls
                    ],
                },
                "layer_agent_tool_calls": tool_calls,
                "layer_agent_assessments": {
                    name: result["assessment"]
                    for name, result in tool_results.items()
                    if "assessment" in result
                },
            },
        }

    return layer_agent_node
