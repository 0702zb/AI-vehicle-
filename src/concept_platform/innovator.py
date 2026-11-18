"""Innovator candidate generation and deterministic boundary checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from concept_platform.agents import (
    AgentExecutionResult,
    AgentExecutor,
    AgentInput,
    AgentProposal,
    AgentTaskContext,
    EvidenceReference,
    ModelReference,
)
from concept_platform.artifacts import ArtifactStore
from concept_platform.state import NodeStatus, RunState


class VariableDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=1)
    value_type: str
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    unit: str | None = None
    allowed_values: tuple[str, ...] = ()
    definition_version: str = Field(min_length=1)


class ConceptVariable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=1)
    value: str | int | float | bool


class IdeaCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    design_principle: str = Field(min_length=1)
    variables: tuple[ConceptVariable, ...]
    assumptions: tuple[str, ...] = ()
    benefits: tuple[str, ...] = ()
    tradeoffs: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    validation_items: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
    model_refs: tuple[ModelReference, ...] = ()


class InnovatorInput(AgentInput):
    frozen_brief_ref: str
    brief_version: str
    context_manifest_hash: str
    candidate_budget: int = Field(ge=1, le=20)
    variable_definitions: tuple[VariableDefinition, ...]


class InnovatorProposal(AgentProposal):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[IdeaCandidate, ...]


class InnovatorInputResolver(Protocol):
    async def resolve(
        self, run_id: str, state: RunState
    ) -> tuple[dict[str, Any], AgentTaskContext]: ...


@dataclass(frozen=True)
class InnovatorOutcome:
    status: NodeStatus
    artifact_ref: str | None
    reason: str | None = None


class InnovatorService:
    def __init__(
        self,
        *,
        executor: AgentExecutor,
        input_resolver: InnovatorInputResolver,
        artifact_store: ArtifactStore,
        variable_definitions: Sequence[VariableDefinition],
        agent_id: str = "innovator",
        agent_version: str,
        candidate_budget: int,
    ) -> None:
        if candidate_budget < 1:
            raise ValueError("Candidate budget must be positive.")
        self._executor = executor
        self._input_resolver = input_resolver
        self._artifact_store = artifact_store
        self._variable_definitions = tuple(variable_definitions)
        self._agent_id = agent_id
        self._agent_version = agent_version
        self._candidate_budget = candidate_budget

    async def generate(self, state: RunState) -> InnovatorOutcome:
        task_input, context = await self._input_resolver.resolve(state["run_id"], state)
        if context.context_manifest_hash != state["context_manifest_hash"]:
            raise ValueError("Innovator context does not match the frozen run manifest.")
        if task_input.get("context_manifest_hash") != state["context_manifest_hash"]:
            raise ValueError("Innovator input does not match the frozen run manifest.")
        if task_input.get("brief_version") != state["brief_version"]:
            raise ValueError("Innovator input does not match the frozen Brief version.")

        result: AgentExecutionResult = await self._executor.execute(
            agent_id=self._agent_id,
            agent_version=self._agent_version,
            task_input=task_input,
            context=context,
        )
        if result.status is NodeStatus.BLOCKED or result.proposal is None:
            return InnovatorOutcome(NodeStatus.BLOCKED, None, result.reason)
        if not isinstance(result.proposal, InnovatorProposal):
            raise TypeError("Innovator AgentDefinition must use InnovatorProposal output.")

        self._validate_candidates(result.proposal, context)
        candidate_versions = []
        for candidate in result.proposal.candidates:
            candidate_payload = {
                "artifact_kind": "candidate_concept",
                "run_id": state["run_id"],
                "project_id": context.project_id,
                "concept_id": context.concept_id,
                "parent_concept_version": context.concept_version,
                "brief_version": state["brief_version"],
                "context_manifest_hash": state["context_manifest_hash"],
                "candidate": candidate.model_dump(mode="json"),
                "agent_version": result.agent_version,
                "prompt_template_version": result.prompt_template_version,
                "provider_id": result.provider_id,
                "model_id": result.model_id,
                "model_version": result.model_version,
            }
            digest = hashlib.sha256(
                json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            candidate_payload["concept_version"] = "concept:" + digest
            candidate_versions.append(await self._artifact_store.put_json(candidate_payload))

        portfolio = {
            "artifact_kind": "candidate_portfolio",
            "run_id": state["run_id"],
            "brief_version": state["brief_version"],
            "context_manifest_hash": state["context_manifest_hash"],
            "candidate_refs": candidate_versions,
        }
        portfolio_ref = await self._artifact_store.put_json(portfolio)
        return InnovatorOutcome(result.status, portfolio_ref)

    def _validate_candidates(
        self, proposal: InnovatorProposal, context: AgentTaskContext
    ) -> None:
        if len(proposal.candidates) > self._candidate_budget:
            raise ValueError("Innovator exceeded the server-owned candidate budget.")

        definitions = {item.variable_id: item for item in self._variable_definitions}
        if len(definitions) != len(self._variable_definitions):
            raise ValueError("Variable registry contains duplicate IDs.")

        combinations: set[str] = set()
        for candidate in proposal.candidates:
            if not candidate.variables:
                raise ValueError("Every candidate must specify at least one registered variable.")
            normalized: dict[str, str | int | float | bool] = {}
            for variable in candidate.variables:
                definition = definitions.get(variable.variable_id)
                if definition is None:
                    raise ValueError(f"Unregistered design variable: {variable.variable_id}")
                if variable.variable_id in normalized:
                    raise ValueError(f"Duplicate design variable: {variable.variable_id}")
                self._validate_variable_value(definition, variable.value)
                normalized[variable.variable_id] = variable.value

            combination_hash = hashlib.sha256(
                json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if combination_hash in combinations:
                raise ValueError("Innovator produced duplicate normalized variable combinations.")
            combinations.add(combination_hash)

            for ref in candidate.evidence_refs:
                if ref.project_id != context.project_id:
                    raise ValueError("Candidate contains cross-project evidence reference.")
                if ref.context_manifest_hash != context.context_manifest_hash:
                    raise ValueError("Candidate evidence reference is outside the active manifest.")
                if context.allowed_evidence.get(ref.evidence_id) != ref.version:
                    raise ValueError("Candidate evidence reference is absent or version-mismatched.")
            for ref in candidate.model_refs:
                if ref.project_id != context.project_id:
                    raise ValueError("Candidate contains cross-project model reference.")
                if ref.context_manifest_hash != context.context_manifest_hash:
                    raise ValueError("Candidate model reference is outside the active manifest.")
                if context.allowed_models.get(ref.model_id) != ref.version:
                    raise ValueError("Candidate model reference is absent or version-mismatched.")

    @staticmethod
    def _validate_variable_value(definition: VariableDefinition, value: Any) -> None:
        if definition.value_type == "enum":
            if str(value) not in definition.allowed_values:
                raise ValueError(f"Value is outside allowed options for {definition.variable_id}.")
            return
        if definition.value_type == "text":
            if not isinstance(value, str) or not value:
                raise ValueError(f"Variable {definition.variable_id} must be non-empty text.")
            return
        if definition.value_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Variable {definition.variable_id} must be boolean.")
            return
        if definition.value_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Variable {definition.variable_id} must be numeric.")
            try:
                number = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(f"Variable {definition.variable_id} is not finite numeric input.") from exc
            if not number.is_finite():
                raise ValueError(f"Variable {definition.variable_id} is not finite numeric input.")
            if definition.minimum is not None and number < definition.minimum:
                raise ValueError(f"Variable {definition.variable_id} is below its registered minimum.")
            if definition.maximum is not None and number > definition.maximum:
                raise ValueError(f"Variable {definition.variable_id} is above its registered maximum.")
            return
        raise ValueError(f"Unsupported registered variable type: {definition.value_type}")


def make_innovator_node(service: InnovatorService):
    async def innovator_node(state: RunState) -> dict[str, Any]:
        outcome = await service.generate(state)
        patch: dict[str, Any] = {"node_status": {"innovator": outcome.status}}
        if outcome.artifact_ref:
            patch["result_ids"] = [outcome.artifact_ref]
        return patch

    return innovator_node


def make_candidate_concept_node():
    """Deterministic boundary after Innovator; records whether a portfolio exists."""

    async def candidate_concept_node(state: RunState) -> dict[str, Any]:
        if state["node_status"].get("innovator") is NodeStatus.BLOCKED or not state["result_ids"]:
            status = NodeStatus.BLOCKED
        else:
            status = state["node_status"].get("innovator", NodeStatus.UNKNOWN)
        return {"node_status": {"candidate_concept": status}}

    return candidate_concept_node
