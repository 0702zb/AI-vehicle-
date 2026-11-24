"""Typed contracts for LayerAssessment and PropertyVerdict outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from concept_platform.agents import AgentProposal, ProposalStatus
from concept_platform.engineering_knowledge import (
    ConfidenceLevel,
    EvidenceClass,
    LayerAssessmentVerdict,
    PropertyVerdictStatus,
)

_T = TypeVar("_T")


def _coerce_tuple(value: Any) -> Any:
    """Accept a single scalar or list where the model emits one item instead of an array."""

    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


CoercedTuple = Annotated[tuple[_T, ...], BeforeValidator(_coerce_tuple)]


class Findings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1)
    evidence_class: EvidenceClass = EvidenceClass.UNKNOWN
    source_ids: CoercedTuple[str] = Field(default_factory=tuple)
    engineering_implications: str = Field(min_length=1)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW


class PropertyImpact(Findings):
    property_id: str = Field(min_length=1)


class PropertyRecord(BaseModel):
    """Legacy-compatible property record retained for demo and contract tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: ProposalStatus
    source_domains: CoercedTuple[str] = Field(default_factory=tuple)
    basis: CoercedTuple[str] = Field(default_factory=tuple)
    rationale: str = Field(min_length=1)
    validation_items: CoercedTuple[str] = Field(min_length=1)


class LayerAssessment(AgentProposal):
    """One domain architecture report plus a single downstream property handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_id: str = Field(min_length=1)
    property_impacts: CoercedTuple[PropertyImpact] = Field(min_length=1)
    overall_assessment: LayerAssessmentVerdict = Field(
        description="Required explicit Layer Agent verdict; never rely on a default."
    )

    # Backward-compatible field used by the existing Layer demo trace.
    domain: str = ""


class PropertyVerdict(AgentProposal):
    model_config = ConfigDict(extra="forbid", frozen=True)

    property_id: str = ""
    evaluated_layers: CoercedTuple[str] = Field(default_factory=tuple)
    extracted_findings: CoercedTuple[Findings] = Field(default_factory=tuple)
    normalized_records: CoercedTuple[str] = Field(default_factory=tuple)
    aggregation_summary: str = Field(min_length=1)
    target_comparison: str = Field(min_length=1)
    cross_domain_tradeoffs: CoercedTuple[str] = Field(default_factory=tuple)
    critical_risks: CoercedTuple[str] = Field(default_factory=tuple)
    mitigation_recommendations: CoercedTuple[str] = Field(default_factory=tuple)
    missing_information: CoercedTuple[str] = Field(default_factory=tuple)
    evidence_references: CoercedTuple[str] = Field(default_factory=tuple)
    rubric_result: PropertyVerdictStatus | None = None
    overall_verdict: PropertyVerdictStatus | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.LOW

    # Backward-compatible property records used by the first demo version.
    properties: CoercedTuple[PropertyRecord] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def fill_insufficient_information_gaps(cls, value: Any) -> Any:
        """Keep the trace honest when the model omits gaps for an insufficient verdict."""

        if not isinstance(value, Mapping):
            return value
        verdict = value.get("rubric_result") or value.get("overall_verdict")
        if verdict == PropertyVerdictStatus.INSUFFICIENT_INFORMATION and not value.get(
            "missing_information"
        ):
            return {
                **value,
                "missing_information": [
                    "Registered target or deterministic result is not available."
                ],
            }
        return value
