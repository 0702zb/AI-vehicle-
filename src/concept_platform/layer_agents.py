"""Layer agent with domain tool agents (Body/Chassis/Drive/Energy/EE/Thermal/Interior).

The concept is sent to the layer agent once. The layer agent is an LLM that
emits tool calls; each tool is a domain agent sharing a common rule set plus
domain-specific refinement. Only Body and Chassis are implemented here; the
other domains are registered as reserved tool names.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from concept_platform.agents import (
    AgentDefinition,
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
from concept_platform.config import Environment
from concept_platform.state import NodeStatus, RunState

LAYER_AGENT_VERSION = "layer-v1"
DEFAULT_PROVIDER_ID = "layer-llm"
DEFAULT_MODEL_ID = "layer-model"
DEFAULT_MODEL_VERSION = "layer-model-v1"
ROUTER_AGENT_ID = "layer_router"

# One shared rule set for every domain tool agent.
SHARED_RULES = (
    "所有领域工具必须遵守同一套规则：概念以文字描述和工程权衡为主；"
    "只基于已冻结的概念文本与 RAG 检索到的证据输出文字性工程建议；"
    "不得编造数值目标，概念或证据中确有价值时必须引用来源并说明适用条件；"
    "rag_context 为空时 evidence_refs 必须为空数组；"
    "输出必须是结构化指令且 directives 至少给出一条文字性建议；"
    "区分事实、假设与风险；不得给出 PASS/FAIL 结论；"
    "不得修改其它领域职责范围。"
)

DOMAIN_REFINEMENTS = {
    "body": "车身领域：负责车身结构、材料、空气动力学、开闭件与座舱布置。",
    "chassis": "底盘领域：负责悬架、转向、制动、轮胎、副车架与整车动力学。",
}

RESERVED_TOOLS = ("drive", "energy", "ee", "thermal", "interior")


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


class ToolAgentProposal(AgentProposal):
    domain: str
    directives: list[str] = Field(default_factory=list)


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
}


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
        prompt_template_version="layer-prompt-v1",
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
            system_instruction=(
                "你是 Layer Agent。概念只输入一次。用大模型理解概念文字描述与工程权衡，"
                "仅根据概念与工具职责决定调用哪些领域工具，证据检索由各子 Agent 自行完成。"
                "tool_calls.arguments 只给出文字性 focus，不要编造数值参数；"
                "focus 必须落在所选工具的职责范围内。"
                "可选工具：" + ", ".join(TOOL_DEFINITIONS) + "。"
                + "body 负责车身/材料/空气动力学/座舱，chassis 负责悬架/转向/制动/轮胎/底盘动力学；"
                + "低风阻外观属于 body，不属于 chassis。未实现的领域不得调用。"
            ),
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
                system_instruction=SHARED_RULES + " " + DOMAIN_REFINEMENTS[name],
                provider_id=provider_id,
                model_id=model_id,
                model_version=model_version,
            )
        )

    def layer_agent_node(state: RunState) -> dict[str, Any]:
        concept = state["coverage"].get("selected_concept")
        if not concept:
            return {
                "node_status": {"layer_agent": NodeStatus.BLOCKED},
                "coverage": {"errors": [{"source": "layer_agent", "status": "BLOCKED"}]},
            }

        router_context = AgentTaskContext(
            project_id="demo-project",
            concept_id="demo-concept",
            concept_version=state["brief_version"],
            context_manifest_hash=state["context_manifest_hash"],
            allowed_evidence={},
            allowed_models={},
        )

        router = asyncio.run(
            executor.execute(
                agent_id=ROUTER_AGENT_ID,
                agent_version=LAYER_AGENT_VERSION,
                task_input={
                    "concept": concept,
                    "available_tools": list(TOOL_DEFINITIONS),
                    "rag_context": [],
                },
                context=router_context,
            )
        )
        if router.status is NodeStatus.BLOCKED or not isinstance(router.proposal, LayerRouterProposal):
            return {
                "node_status": {"layer_agent": NodeStatus.BLOCKED},
                "coverage": {"errors": [{"source": "layer_router", "status": "BLOCKED"}]},
            }

        tool_calls: list[dict[str, Any]] = []
        tool_results: dict[str, Any] = {}
        for call in router.proposal.tool_calls:
            tool = TOOL_DEFINITIONS.get(call.tool_name)
            if tool is None:
                tool_calls.append(
                    {"tool_name": call.tool_name, "arguments": call.arguments, "status": "unknown_tool"}
                )
                continue

            focus = str(call.arguments.get("focus", ""))
            tool_hits = (
                rag.retrieve(f"{concept}\n{focus}", domain=call.tool_name)
                if rag is not None
                else ()
            )
            tool_rag_context = [
                {
                    "evidence_id": hit.evidence_id,
                    "project_id": hit.project_id,
                    "version": hit.version,
                    "context_manifest_hash": hit.context_manifest_hash,
                    "text": hit.text,
                    "domain": hit.domain or call.tool_name,
                }
                for hit in tool_hits
            ]
            tool_context = AgentTaskContext(
                project_id="demo-project",
                concept_id="demo-concept",
                concept_version=state["brief_version"],
                context_manifest_hash=state["context_manifest_hash"],
                allowed_evidence={hit.evidence_id: hit.version for hit in tool_hits},
                allowed_models={},
            )
            result = asyncio.run(
                executor.execute(
                    agent_id=tool.agent_id,
                    agent_version=LAYER_AGENT_VERSION,
                    task_input={
                        "concept": concept,
                        "focus": focus,
                        "rag_context": tool_rag_context,
                    },
                    context=tool_context,
                )
            )
            tool_calls.append(
                {"tool_name": call.tool_name, "arguments": call.arguments, "status": result.status.value}
            )
            if isinstance(result.proposal, ToolAgentProposal):
                tool_results[call.tool_name] = {
                    "status": result.status.value,
                    "summary": result.proposal.summary,
                    "directives": list(result.proposal.directives),
                    "evidence_refs": [ref.evidence_id for ref in result.proposal.evidence_refs],
                    "rag_context": tool_rag_context,
                }

        status = NodeStatus.SUPPORTED if tool_results else NodeStatus.BLOCKED
        return {
            "node_status": {"layer_agent": status},
            "coverage": {
                "layer_agent_trace": {
                    "reasoning": router.proposal.reasoning,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                }
            },
        }

    return layer_agent_node
