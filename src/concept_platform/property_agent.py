"""Property agent: cross-domain assessment against a versioned Property Rubric."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from concept_platform.agents import (
    AgentDefinition,
    AgentExecutor,
    AgentInput,
    AgentRegistry,
    AgentTaskContext,
    ProposalStatus,
    ProviderAdapter,
    ProviderRegistry,
)
from concept_platform.assessment_schemas import PropertyImpact, PropertyRecord, PropertyVerdict
from concept_platform.config import Environment
from concept_platform.engineering_knowledge import (
    PROPERTY_RUBRICS,
    PropertyRubric,
    PropertyVerdictStatus,
)
from concept_platform.layer_agents import HttpProviderAdapter, build_task_context
from concept_platform.prompts import PROPERTY_SHARED_SYSTEM_PROMPT
from concept_platform.state import NodeStatus, RunState

PROPERTY_AGENT_VERSION = "property-v1"
PROPERTY_AGENT_ID = "property_agent"
DEFAULT_PROPERTY_ID = "mass"
DEFAULT_PROVIDER_ID = "property-llm"
DEFAULT_MODEL_ID = "property-model"
DEFAULT_MODEL_VERSION = "property-model-v1"

SYSTEM_INSTRUCTION = PROPERTY_SHARED_SYSTEM_PROMPT + (
    "The Property Rubric is injected in task_input.rubric and is authoritative.\n"
    "Use task_input.layer_assessments as primary input; layer_directives are legacy fallback only.\n"
    "The JSON envelope must include project_id, concept_id, concept_version, "
    "context_manifest_hash, status and summary exactly as given in task_context.\n"
    "Return property_id, evaluated_layers, extracted_findings, aggregation_summary, "
    "target_comparison, rubric_result, confidence, and the properties array.\n"
    "The field name is exactly properties. Never output legacy_properties or other invented field names.\n"
    "For every property record, basis must be verbatim text from concept, a Layer "
    "directive, or a LayerAssessment property impact statement.\n"
    "rubric_result must be PASS, CONDITIONAL, FAIL or INSUFFICIENT_INFORMATION.\n"
    "Do not create numerical verdicts without registered deterministic tool records.\n"
)

_STATUS_RANK = {
    ProposalStatus.SUPPORTED: 0,
    ProposalStatus.ASSUMPTION: 1,
    ProposalStatus.UNKNOWN: 2,
    ProposalStatus.BLOCKED: 3,
}
_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?(?![\w])")


class LayerDirective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str = Field(min_length=1)
    directive: str = Field(min_length=1)
    focus: str = ""


class LayerAssessmentInput(AgentInput):
    layer_id: str = Field(min_length=1)
    property_impacts: tuple[PropertyImpact, ...] = Field(default_factory=tuple)
    directives: tuple[str, ...] = Field(default_factory=tuple)


class PropertyAgentInput(AgentInput):
    concept: str = Field(min_length=1)
    property_id: str = Field(min_length=1)
    rubric: PropertyRubric
    layer_assessments: tuple[LayerAssessmentInput, ...] = Field(default_factory=tuple)
    directives: tuple[LayerDirective, ...] = Field(default_factory=tuple)


class PropertyAgentProposal(PropertyVerdict):
    @model_validator(mode="after")
    def unique_property_ids(self) -> PropertyAgentProposal:
        ids = [item.property_id for item in self.properties]
        if len(ids) != len(set(ids)):
            raise ValueError("PropertyAgent returned duplicate property_id values.")
        return self


@dataclass(frozen=True)
class PropertyAgentOutcome:
    status: NodeStatus
    trace: dict[str, Any]


def build_property_agent_input(
    state: RunState, property_id: str = DEFAULT_PROPERTY_ID
) -> tuple[PropertyAgentInput | None, str | None]:
    """Build a RAG-free input from frozen concept and real LayerAssessment records."""

    rubric = PROPERTY_RUBRICS.get(property_id)
    if rubric is None:
        return None, f"Unregistered property_id: {property_id}."

    concept = state.get("coverage", {}).get("selected_concept")
    if not isinstance(concept, str) or not concept.strip():
        return None, "selected_concept is missing."

    layer_trace = state.get("coverage", {}).get("layer_agent_trace")
    if not isinstance(layer_trace, Mapping):
        return None, "layer_agent_trace is missing."

    tool_results = layer_trace.get("tool_results")
    if not isinstance(tool_results, Mapping) or not tool_results:
        return None, "layer_agent_trace.tool_results is empty."

    focus_by_domain: dict[str, str] = {}
    tool_calls = layer_trace.get("tool_calls", ())
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            domain = call.get("tool_name")
            arguments = call.get("arguments")
            if isinstance(domain, str) and isinstance(arguments, Mapping):
                focus = arguments.get("focus")
                if isinstance(focus, str) and focus.strip():
                    focus_by_domain[domain] = focus.strip()

    assessments: list[LayerAssessmentInput] = []
    directives: list[LayerDirective] = []
    for domain, result in tool_results.items():
        if not isinstance(domain, str) or not domain or not isinstance(result, Mapping):
            continue

        assessment_data = result.get("assessment")
        if isinstance(assessment_data, Mapping):
            impacts: list[PropertyImpact] = []
            for raw_impact in assessment_data.get("property_impacts", ()) or ():
                try:
                    impact = PropertyImpact.model_validate(raw_impact)
                except Exception:
                    continue
                if impact.property_id == property_id:
                    impacts.append(impact)
            assessment_directives = tuple(
                value.strip()
                for value in (assessment_data.get("directives", ()) or ())
                if isinstance(value, str) and value.strip()
            )
            assessments.append(
                LayerAssessmentInput(
                    layer_id=str(assessment_data.get("layer_id") or domain),
                    property_impacts=tuple(impacts),
                    directives=assessment_directives,
                )
            )

        for value in result.get("directives", ()) or ():
            if isinstance(value, str) and value.strip():
                directives.append(
                    LayerDirective(
                        domain=domain,
                        directive=value.strip(),
                        focus=focus_by_domain.get(domain, ""),
                    )
                )

    if not assessments and not directives:
        return None, "layer_agent_trace contains neither assessments nor directives."

    return (
        PropertyAgentInput(
            concept=concept.strip(),
            property_id=property_id,
            rubric=rubric,
            layer_assessments=tuple(assessments),
            directives=tuple(directives),
        ),
        None,
    )


def _trusted_texts(task_input: PropertyAgentInput) -> tuple[str, ...]:
    texts = [task_input.concept]
    for directive in task_input.directives:
        texts.append(directive.directive)
    for assessment in task_input.layer_assessments:
        texts.extend(assessment.directives)
        for impact in assessment.property_impacts:
            texts.append(impact.description)
    return tuple(texts)


def _available_domains(task_input: PropertyAgentInput) -> set[str]:
    domains = {directive.domain for directive in task_input.directives}
    domains.update(assessment.layer_id for assessment in task_input.layer_assessments)
    return domains


def _is_traceable(basis: str, trusted_texts: tuple[str, ...]) -> bool:
    candidate = basis.strip()
    return bool(candidate) and any(candidate in text for text in trusted_texts)


def _has_unbacked_numeric_claim(
    claims: tuple[str, ...], trusted_texts: tuple[str, ...]
) -> bool:
    """A number is unbacked unless the exact token appears in trusted evidence."""
    trusted_tokens = {
        token for text in trusted_texts for token in _NUMBER_RE.findall(text)
    }
    return any(
        token not in trusted_tokens
        for claim in claims
        for token in _NUMBER_RE.findall(claim or "")
    )


def _aggregate_status(statuses: list[ProposalStatus]) -> ProposalStatus:
    if not statuses:
        return ProposalStatus.UNKNOWN
    return max(statuses, key=lambda status: _STATUS_RANK[status])


def _status_to_verdict(status: ProposalStatus) -> PropertyVerdictStatus:
    if status is ProposalStatus.SUPPORTED:
        return PropertyVerdictStatus.PASS
    if status is ProposalStatus.ASSUMPTION:
        return PropertyVerdictStatus.CONDITIONAL
    return PropertyVerdictStatus.INSUFFICIENT_INFORMATION


def _verdict_to_node_status(verdict: PropertyVerdictStatus) -> NodeStatus:
    if verdict is PropertyVerdictStatus.PASS:
        return NodeStatus.SUPPORTED
    if verdict is PropertyVerdictStatus.CONDITIONAL:
        return NodeStatus.ASSUMPTION
    if verdict is PropertyVerdictStatus.FAIL:
        return NodeStatus.FAIL
    return NodeStatus.UNKNOWN


def _final_verdict(status: ProposalStatus, proposed: PropertyVerdictStatus | None) -> PropertyVerdictStatus:
    """The rule layer decides: keep the more conservative of evidence status and model verdict."""
    deterministic = _status_to_verdict(status)
    if proposed is None:
        return deterministic
    severity = {
        PropertyVerdictStatus.PASS: 0,
        PropertyVerdictStatus.CONDITIONAL: 1,
        PropertyVerdictStatus.INSUFFICIENT_INFORMATION: 2,
        PropertyVerdictStatus.FAIL: 3,
    }
    return max((deterministic, proposed), key=severity.__getitem__)


def _normalize_proposal(
    proposal: PropertyAgentProposal, task_input: PropertyAgentInput
) -> tuple[ProposalStatus, list[dict[str, Any]]]:
    available_domains = _available_domains(task_input)
    trusted_texts = _trusted_texts(task_input)
    shared_claims = (
        proposal.aggregation_summary,
        proposal.target_comparison,
        *proposal.cross_domain_tradeoffs,
        *proposal.critical_risks,
    )

    if not proposal.properties:
        return ProposalStatus.UNKNOWN, []

    normalized: list[dict[str, Any]] = []
    final_statuses: list[ProposalStatus] = []
    for item in proposal.properties:
        adjustments: list[str] = []
        invalid_domains = sorted(set(item.source_domains) - available_domains)
        basis_traceable = bool(item.basis) and all(
            _is_traceable(basis, trusted_texts) for basis in item.basis
        )
        claims = (*shared_claims, item.name, item.rationale, *item.validation_items)
        has_unbacked_number = _has_unbacked_numeric_claim(claims, trusted_texts)

        if invalid_domains:
            adjustments.append("source_domains_outside_layer_trace")
        if not basis_traceable:
            adjustments.append("basis_not_traceable")

        # Evidence can only lower the status floor; it can never upgrade the model's own
        # UNKNOWN/BLOCKED. The more conservative of the two always wins.
        floor = ProposalStatus.SUPPORTED
        if invalid_domains or not basis_traceable:
            floor = ProposalStatus.UNKNOWN
        elif has_unbacked_number:
            adjustments.append("unbacked_numeric_claim")
            floor = ProposalStatus.ASSUMPTION
        final_status = max((item.status, floor), key=_STATUS_RANK.__getitem__)

        final_statuses.append(final_status)
        normalized.append(
            {
                "property_id": item.property_id,
                "name": item.name,
                "status": final_status.value,
                "proposed_status": item.status.value,
                "source_domains": list(item.source_domains),
                "basis": list(item.basis),
                "basis_traceable": basis_traceable,
                "rationale": item.rationale,
                "validation_items": list(item.validation_items),
                "adjustments": adjustments,
            }
        )

    return _aggregate_status(final_statuses), normalized


class PropertyAgentService:
    """Deterministic boundary and execution logic for one Property Agent."""

    def __init__(
        self,
        *,
        executor: AgentExecutor,
        property_id: str = DEFAULT_PROPERTY_ID,
        agent_id: str = PROPERTY_AGENT_ID,
        agent_version: str = PROPERTY_AGENT_VERSION,
    ) -> None:
        self._executor = executor
        self._property_id = property_id
        self._agent_id = agent_id
        self._agent_version = agent_version

    async def derive(self, state: RunState) -> PropertyAgentOutcome:
        task_input, reason = build_property_agent_input(state, self._property_id)
        if task_input is None:
            return PropertyAgentOutcome(
                NodeStatus.BLOCKED,
                {
                    "status": NodeStatus.BLOCKED.value,
                    "support_basis": "frozen_concept_and_layer_directives",
                    "reason": reason,
                    "properties": [],
                },
            )

        context = build_task_context(state)
        result = await self._executor.execute(
            agent_id=self._agent_id,
            agent_version=self._agent_version,
            task_input=task_input.model_dump(mode="json"),
            context=context,
        )
        if result.proposal is None:
            return PropertyAgentOutcome(
                NodeStatus.BLOCKED,
                {
                    "status": NodeStatus.BLOCKED.value,
                    "support_basis": "frozen_concept_and_layer_directives",
                    "reason": result.reason,
                    "properties": [],
                },
            )
        if not isinstance(result.proposal, PropertyAgentProposal):
            raise TypeError("Property AgentDefinition must use PropertyAgentProposal output.")
        if not result.proposal.properties and result.proposal.status is ProposalStatus.BLOCKED:
            return PropertyAgentOutcome(
                NodeStatus.BLOCKED,
                {
                    "status": NodeStatus.BLOCKED.value,
                    "support_basis": "frozen_concept_and_layer_directives",
                    "reason": result.reason or "Property Agent returned blocked without properties.",
                    "properties": [],
                },
            )

        status, properties = _normalize_proposal(result.proposal, task_input)
        returned_property_id = result.proposal.property_id.strip()
        property_id_ok = not returned_property_id or returned_property_id == task_input.property_id
        if not property_id_ok:
            status = max(status, ProposalStatus.UNKNOWN, key=_STATUS_RANK.__getitem__)
        proposed_verdict = result.proposal.rubric_result or result.proposal.overall_verdict
        verdict = _final_verdict(status, proposed_verdict)
        node_status = NodeStatus(status.value)
        trace = {
            "status": node_status.value,
            "property_id": task_input.property_id,
            "returned_property_id": returned_property_id,
            "property_id_ok": property_id_ok,
            "rubric": {
                "property_id": task_input.rubric.property_id,
                "property_name": task_input.rubric.property_name,
                "objective": task_input.rubric.objective,
            },
            "proposed_top_level_status": result.proposal.status.value,
            "proposed_rubric_result": proposed_verdict.value if proposed_verdict else None,
            "rubric_result": verdict.value,
            "confidence": result.proposal.confidence.value,
            "support_basis": "frozen_concept_and_layer_directives",
            "evaluated_layers": list(result.proposal.evaluated_layers),
            "extracted_findings": [
                finding.model_dump(mode="json") for finding in result.proposal.extracted_findings
            ],
            "aggregation_summary": result.proposal.aggregation_summary,
            "target_comparison": result.proposal.target_comparison,
            "cross_domain_tradeoffs": list(result.proposal.cross_domain_tradeoffs),
            "critical_risks": list(result.proposal.critical_risks),
            "mitigation_recommendations": list(result.proposal.mitigation_recommendations),
            "missing_information": list(result.proposal.missing_information),
            "source_domains": sorted(_available_domains(task_input)),
            "directive_count": len(task_input.directives),
            "assessment_count": len(task_input.layer_assessments),
            "properties": properties,
        }
        return PropertyAgentOutcome(node_status, trace)


def create_property_agent_node(
    *,
    adapter: ProviderAdapter | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    property_id: str = DEFAULT_PROPERTY_ID,
    provider_id: str = DEFAULT_PROVIDER_ID,
    model_id: str = DEFAULT_MODEL_ID,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Callable[[RunState], dict[str, Any]]:
    """Build one property_agent node with a registered rubric and provider adapter."""

    if property_id not in PROPERTY_RUBRICS:
        raise ValueError(f"Unregistered property_id: {property_id}.")

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
    agents.register(
        AgentDefinition(
            agent_id=PROPERTY_AGENT_ID,
            version=PROPERTY_AGENT_VERSION,
            input_schema=PropertyAgentInput,
            output_schema=PropertyAgentProposal,
            prompt_template_version="property-prompt-v7",
            system_instruction=SYSTEM_INSTRUCTION,
            provider_id=provider_id,
            model_id=model_id,
            model_version=model_version,
            tool_allowlist=frozenset(),
            context_scope=frozenset(),
            token_budget=4096,
        )
    )
    service = PropertyAgentService(executor=AgentExecutor(agents, providers), property_id=property_id)

    async def property_agent_node(state: RunState) -> dict[str, Any]:
        outcome = await service.derive(state)
        patch: dict[str, Any] = {
            "node_status": {"property_agent": outcome.status},
            "coverage": {"property_agent_trace": outcome.trace},
        }
        if outcome.status in (NodeStatus.BLOCKED, NodeStatus.FAIL):
            patch["coverage"]["errors"] = [
                {
                    "source": "property_agent",
                    "status": outcome.status.value,
                    "reason": outcome.trace.get("reason"),
                }
            ]
        return patch

    return property_agent_node
