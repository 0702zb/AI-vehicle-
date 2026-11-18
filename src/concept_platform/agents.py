"""Shared, provider-neutral agent execution and backend output validation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from concept_platform.artifacts import ArtifactStore
from concept_platform.config import Environment
from concept_platform.state import NodeStatus, RunState


class ProposalStatus(StrEnum):
    """Statuses a language model may propose; PASS and FAIL are intentionally absent."""

    SUPPORTED = "supported"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


class AgentInput(BaseModel):
    """Base input shape for registered agent tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    context_manifest_hash: str = Field(min_length=1)


class ModelReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    context_manifest_hash: str = Field(min_length=1)


class AgentProposal(BaseModel):
    """Common structured output contract for all LLM-backed agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    concept_version: str = Field(min_length=1)
    context_manifest_hash: str = Field(min_length=1)
    status: ProposalStatus
    summary: str
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    model_refs: list[ModelReference] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)

class AgentDefinition(BaseModel):
    """Immutable, version-pinned configuration shared by all agent tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema: type[BaseModel]
    output_schema: type[AgentProposal]
    prompt_template_version: str = Field(min_length=1)
    system_instruction: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    tool_allowlist: frozenset[str] = frozenset()
    context_scope: frozenset[str] = frozenset()
    token_budget: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    stop_conditions: tuple[str, ...] = ()
    max_schema_repairs: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def output_schema_is_proposal(self) -> AgentDefinition:
        if not issubclass(self.output_schema, AgentProposal):
            raise TypeError("output_schema must extend AgentProposal.")
        return self


@dataclass(frozen=True)
class AgentTaskContext:
    """Trusted, server-built task scope; it is not part of LangGraph RunState."""

    project_id: str
    concept_id: str
    concept_version: str
    context_manifest_hash: str
    allowed_evidence: Mapping[str, str]
    allowed_models: Mapping[str, str]
    context_scope: frozenset[str] = frozenset()
    requested_tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ProviderRequest:
    provider_id: str
    model_id: str
    model_version: str
    agent_id: str
    agent_version: str
    prompt_template_version: str
    system_instruction: str
    task_input: dict[str, Any]
    task_context: dict[str, Any]
    output_schema: dict[str, Any]
    tool_allowlist: tuple[str, ...]
    token_budget: int
    repair_errors: tuple[str, ...] = ()
    repair_candidate: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderResponse:
    structured_output: dict[str, Any]
    model_version: str


class ProviderAdapter(Protocol):
    """Provider-neutral asynchronous structured-output interface."""

    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...


@dataclass(frozen=True)
class RegisteredProvider:
    model_version: str
    adapter: ProviderAdapter


class ProviderRegistry:
    def __init__(self, environment: Environment = Environment.DEV) -> None:
        self._providers: dict[tuple[str, str], RegisteredProvider] = {}
        self._environment = environment

    def register(
        self, provider_id: str, model_id: str, model_version: str, adapter: ProviderAdapter
    ) -> None:
        key = (provider_id, model_id)
        if self._environment is Environment.PROD and isinstance(adapter, MockProvider):
            raise ValueError("MockProvider is restricted to the development environment.")
        if key in self._providers:
            raise ValueError(f"Provider model already registered: {provider_id}/{model_id}")
        self._providers[key] = RegisteredProvider(model_version, adapter)

    def resolve(self, provider_id: str, model_id: str) -> RegisteredProvider | None:
        return self._providers.get((provider_id, model_id))


class AgentRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        key = (definition.agent_id, definition.version)
        if key in self._definitions:
            raise ValueError(f"Agent definition already registered: {key}")
        self._definitions[key] = definition

    def resolve(self, agent_id: str, version: str) -> AgentDefinition | None:
        return self._definitions.get((agent_id, version))


class AgentBackendValidationError(ValueError):
    """Raised when the model response violates a trusted task boundary."""


@dataclass(frozen=True)
class AgentExecutionResult:
    status: NodeStatus
    proposal: AgentProposal | None
    reason: str | None
    agent_id: str
    agent_version: str | None
    prompt_template_version: str | None
    provider_id: str | None
    model_id: str | None
    model_version: str | None


def _provider_context(context: AgentTaskContext) -> dict[str, Any]:
    """Expose only the task's pinned identities and reference allowlists."""

    return {
        "project_id": context.project_id,
        "concept_id": context.concept_id,
        "concept_version": context.concept_version,
        "context_manifest_hash": context.context_manifest_hash,
        "allowed_evidence": dict(context.allowed_evidence),
        "allowed_models": dict(context.allowed_models),
        "context_scope": sorted(context.context_scope),
    }


def _validate_references(proposal: AgentProposal, context: AgentTaskContext) -> None:
    if (
        proposal.project_id != context.project_id
        or proposal.concept_id != context.concept_id
        or proposal.concept_version != context.concept_version
        or proposal.context_manifest_hash != context.context_manifest_hash
    ):
        raise AgentBackendValidationError("Output project/concept/snapshot scope mismatch.")

    for ref in proposal.evidence_refs:
        if ref.project_id != context.project_id:
            raise AgentBackendValidationError("Cross-project evidence reference rejected.")
        if ref.context_manifest_hash != context.context_manifest_hash:
            raise AgentBackendValidationError("Evidence reference is outside the frozen manifest.")
        if context.allowed_evidence.get(ref.evidence_id) != ref.version:
            raise AgentBackendValidationError("Evidence reference is absent or version-mismatched.")

    for ref in proposal.model_refs:
        if ref.project_id != context.project_id:
            raise AgentBackendValidationError("Cross-project model reference rejected.")
        if ref.context_manifest_hash != context.context_manifest_hash:
            raise AgentBackendValidationError("Model reference is outside the frozen manifest.")
        if context.allowed_models.get(ref.model_id) != ref.version:
            raise AgentBackendValidationError("Model reference is absent or version-mismatched.")


class AgentExecutor:
    """One execution implementation for every AgentDefinition."""

    def __init__(self, agents: AgentRegistry, providers: ProviderRegistry) -> None:
        self._agents = agents
        self._providers = providers

    async def execute(
        self,
        *,
        agent_id: str,
        agent_version: str,
        task_input: Mapping[str, Any],
        context: AgentTaskContext,
    ) -> AgentExecutionResult:
        definition = self._agents.resolve(agent_id, agent_version)
        if definition is None:
            return self._blocked(agent_id, None, "Agent definition is not registered.")

        registered = self._providers.resolve(definition.provider_id, definition.model_id)
        if registered is None or registered.model_version != definition.model_version:
            return self._blocked(
                agent_id, definition, "Pinned provider model is not registered or version-matched."
            )

        disallowed_scope = context.context_scope - definition.context_scope
        if disallowed_scope:
            raise AgentBackendValidationError(
                "Task context exceeds AgentDefinition scope: "
                + ", ".join(sorted(disallowed_scope))
            )

        disallowed_tools = context.requested_tools - definition.tool_allowlist
        if disallowed_tools:
            raise AgentBackendValidationError(
                "Requested tools are outside AgentDefinition allowlist: "
                + ", ".join(sorted(disallowed_tools))
            )

        validated_input = definition.input_schema.model_validate(task_input)
        base = ProviderRequest(
            provider_id=definition.provider_id,
            model_id=definition.model_id,
            model_version=definition.model_version,
            agent_id=definition.agent_id,
            agent_version=definition.version,
            prompt_template_version=definition.prompt_template_version,
            system_instruction=definition.system_instruction,
            task_input=validated_input.model_dump(mode="json"),
            task_context=_provider_context(context),
            output_schema=definition.output_schema.model_json_schema(),
            tool_allowlist=tuple(sorted(context.requested_tools)),
            token_budget=definition.token_budget,
        )

        async with asyncio.timeout(definition.timeout_seconds):
            response = await registered.adapter.complete(base)
            if response.model_version != definition.model_version:
                raise AgentBackendValidationError("Provider returned an unpinned model version.")

            repair_errors: tuple[str, ...] = ()
            for repair_index in range(definition.max_schema_repairs + 1):
                try:
                    proposal = definition.output_schema.model_validate(response.structured_output)
                except ValidationError as exc:
                    if repair_index >= definition.max_schema_repairs:
                        raise AgentBackendValidationError(
                            "Agent output failed schema validation after the allowed repair."
                        ) from exc
                    repair_errors = tuple(
                        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                        for error in exc.errors()
                    )
                    repair_request = replace(
                        base,
                        repair_errors=repair_errors,
                        repair_candidate=response.structured_output,
                    )
                    response = await registered.adapter.complete(repair_request)
                    if response.model_version != definition.model_version:
                        raise AgentBackendValidationError(
                            "Provider returned an unpinned model version during schema repair."
                        )
                    continue

                _validate_references(proposal, context)
                return AgentExecutionResult(
                    status=NodeStatus(proposal.status.value),
                    proposal=proposal,
                    reason=None,
                    agent_id=definition.agent_id,
                    agent_version=definition.version,
                    prompt_template_version=definition.prompt_template_version,
                    provider_id=definition.provider_id,
                    model_id=definition.model_id,
                    model_version=definition.model_version,
                )

        raise AssertionError("schema repair loop exited without a result")

    @staticmethod
    def _blocked(
        agent_id: str, definition: AgentDefinition | None, reason: str
    ) -> AgentExecutionResult:
        return AgentExecutionResult(
            status=NodeStatus.BLOCKED,
            proposal=None,
            reason=reason,
            agent_id=agent_id,
            agent_version=definition.version if definition else None,
            prompt_template_version=definition.prompt_template_version if definition else None,
            provider_id=definition.provider_id if definition else None,
            model_id=definition.model_id if definition else None,
            model_version=definition.model_version if definition else None,
        )


class MockProvider:
    """Deterministic provider for local development and contract tests."""

    def __init__(self, responses: Sequence[ProviderResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("Mock provider response queue is empty.")
        return self._responses.pop(0)


class AgentNodeInputResolver(Protocol):
    """Builds a role-specific input from pinned artifacts, never prior agent prose."""

    async def resolve(
        self, state: RunState
    ) -> tuple[Mapping[str, Any], AgentTaskContext]: ...


def make_agent_node(
    *,
    executor: AgentExecutor,
    input_resolver: AgentNodeInputResolver,
    artifact_store: ArtifactStore,
    agent_id: str,
    agent_version: str,
):
    """Shared LangGraph node implementation for all configured AgentDefinitions."""

    async def agent_node(state: RunState) -> dict[str, Any]:
        task_input, context = await input_resolver.resolve(state)
        if context.context_manifest_hash != state["context_manifest_hash"]:
            raise AgentBackendValidationError("Agent task is outside the active evidence manifest.")
        result = await executor.execute(
            agent_id=agent_id,
            agent_version=agent_version,
            task_input=task_input,
            context=context,
        )
        patch: dict[str, Any] = {"node_status": {agent_id: result.status}}
        if result.proposal is not None:
            artifact_ref = await artifact_store.put_json({
                "artifact_kind": "agent_proposal",
                "run_id": state["run_id"],
                "project_id": context.project_id,
                "concept_id": context.concept_id,
                "concept_version": context.concept_version,
                "context_manifest_hash": context.context_manifest_hash,
                "agent_id": result.agent_id,
                "agent_version": result.agent_version,
                "prompt_template_version": result.prompt_template_version,
                "provider_id": result.provider_id,
                "model_id": result.model_id,
                "model_version": result.model_version,
                "proposal": result.proposal.model_dump(mode="json"),
            })
            patch["result_ids"] = [artifact_ref]
        return patch

    return agent_node
