"""Versionable engineering knowledge packs for Layer and Property agents."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceClass(StrEnum):
    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class LayerAssessmentVerdict(StrEnum):
    FEASIBLE = "FEASIBLE"
    FEASIBLE_WITH_CONDITIONS = "FEASIBLE_WITH_CONDITIONS"
    HIGH_RISK = "HIGH_RISK"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class PropertyVerdictStatus(StrEnum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class DomainPack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain_id: str = Field(min_length=1)
    domain_name: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    knowledge_areas: tuple[str, ...] = Field(default_factory=tuple)
    assessment_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    critical_interfaces: tuple[str, ...] = Field(default_factory=tuple)
    domain_constraints: tuple[str, ...] = Field(default_factory=tuple)
    expected_handoff: tuple[str, ...] = Field(default_factory=tuple)


class PropertyRubric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    property_id: str = Field(min_length=1)
    property_name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    evaluation_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    required_evidence: tuple[str, ...] = Field(default_factory=tuple)
    decision_rules: tuple[str, ...] = Field(default_factory=tuple)
    output_requirements: tuple[str, ...] = Field(default_factory=tuple)


DOMAIN_PACKS: dict[str, DomainPack] = {
    "body": DomainPack(
        domain_id="body",
        domain_name="Body & Exterior",
        scope="BIW, closures, structure, exterior surfaces, crash interfaces and cabin package.",
        knowledge_areas=(
            "BIW structure and load paths",
            "Closures and hinges",
            "Materials and joining",
            "Aerodynamics and exterior surfaces",
            "Crash structure and repairability",
        ),
        assessment_dimensions=(
            "Technical feasibility",
            "Mass and structural impact",
            "Package and cabin space",
            "Safety and repairability",
            "Manufacturing and maturity",
            "Cross-domain interfaces",
        ),
        critical_interfaces=(
            "Chassis: body mounting and load paths",
            "Energy: battery enclosure and underbody protection",
            "Thermal: front-end airflow and ducts",
        ),
        domain_constraints=(
            "Do not invent crash results or regulatory compliance.",
            "Do not assume material compatibility without evidence or stated assumptions.",
            "Keep aerodynamics claims within body scope.",
        ),
        expected_handoff=(
            "Affected body structures",
            "Mass and package implications",
            "Crash and repairability dependencies",
            "Open validation items",
        ),
    ),
    "chassis": DomainPack(
        domain_id="chassis",
        domain_name="Chassis",
        scope="Suspension, steering, braking, tires, subframes and vehicle dynamics.",
        knowledge_areas=(
            "Suspension architecture",
            "Steering system",
            "Braking system",
            "Tires and wheels",
            "Subframe and vehicle dynamics",
        ),
        assessment_dimensions=(
            "Technical feasibility",
            "Ride and handling implications",
            "Mass and package",
            "Safety and redundancy",
            "Integration complexity",
            "Cross-domain interfaces",
        ),
        critical_interfaces=(
            "Body: mounting points and load paths",
            "Energy: battery mass distribution",
            "Thermal: brake and underbody thermal interfaces",
        ),
        domain_constraints=(
            "Do not infer numerical dynamics targets without deterministic results.",
            "Do not treat low-speed ride claims as whole-vehicle dynamics validation.",
            "Flag unresolved steering or braking redundancy issues.",
        ),
        expected_handoff=(
            "Affected chassis modules",
            "Ride and steering implications",
            "Mass and package impacts",
            "Open safety and validation items",
        ),
    ),
    "energy": DomainPack(
        domain_id="energy",
        domain_name="Energy & Charging",
        scope="Battery system, HV architecture, charging system and vehicle energy management.",
        knowledge_areas=(
            "Battery system architecture",
            "High-voltage architecture",
            "Charging system",
            "Energy management",
        ),
        assessment_dimensions=(
            "Battery and HV compatibility",
            "Package and mass implications",
            "Charging performance",
            "Energy and range implications",
            "Safety and thermal dependencies",
            "Technology maturity",
        ),
        critical_interfaces=(
            "Body: battery enclosure and underbody protection",
            "Chassis: battery mass distribution and mounting",
            "Thermal: battery cooling and heating",
            "E/E: BMS communication and charging control",
        ),
        domain_constraints=(
            "Distinguish nominal, operating and maximum voltage.",
            "Do not infer charging time or range from capacity alone.",
            "Flag unresolved HV safety, thermal propagation and charging compatibility issues.",
        ),
        expected_handoff=(
            "Battery and HV architecture changes",
            "Charging requirements",
            "Mass and package implications",
            "Thermal and safety interfaces",
        ),
    ),
}

PROPERTY_RUBRICS: dict[str, PropertyRubric] = {
    "mass": PropertyRubric(
        property_id="mass",
        property_name="Mass",
        objective="Evaluate vehicle-level mass changes and target impact across technical domains.",
        evaluation_dimensions=(
            "Mass contributors by layer",
            "Unit and baseline consistency",
            "Double counting and duplication",
            "Target deviation",
            "Mass-related trade-offs",
        ),
        required_evidence=(
            "Concept premises with mass target",
            "Layer mass-impact records",
            "Registered deterministic mass outputs where available",
        ),
        decision_rules=(
            "PASS: validated mass evidence demonstrates compliance with the target.",
            "CONDITIONAL: compliance depends on unresolved assumptions or offset measures.",
            "FAIL: validated evidence demonstrates a target violation.",
            "INSUFFICIENT_INFORMATION: required mass evidence or target definition is unavailable.",
        ),
        output_requirements=(
            "Mass target",
            "Aggregated contributions",
            "Target comparison",
            "Major trade-offs",
            "Missing information",
            "Final verdict",
        ),
    ),
    "package_space": PropertyRubric(
        property_id="package_space",
        property_name="Package & Space",
        objective="Evaluate whether the concept satisfies space, hard-point and clearance requirements.",
        evaluation_dimensions=(
            "Space demand by layer",
            "Hard-point compatibility",
            "Clearance and packaging conflicts",
            "Occupant and cargo implications",
        ),
        required_evidence=(
            "Package premises and hard points",
            "Layer package-impact records",
            "Registered package measurements where available",
        ),
        decision_rules=(
            "PASS: validated package evidence demonstrates compliance.",
            "CONDITIONAL: compliance depends on unresolved packaging actions.",
            "FAIL: validated evidence demonstrates a hard-point violation.",
            "INSUFFICIENT_INFORMATION: package data or hard points are unavailable.",
        ),
        output_requirements=(
            "Space target",
            "Layer package demands",
            "Conflicts",
            "Missing hard-point data",
            "Final verdict",
        ),
    ),
    "efficiency_range": PropertyRubric(
        property_id="efficiency_range",
        property_name="Efficiency & Range",
        objective="Evaluate energy consumption and range implications across technical domains.",
        evaluation_dimensions=(
            "Energy consumption contributors",
            "Range assumptions",
            "Charging and energy-management dependencies",
            "Operating-condition consistency",
        ),
        required_evidence=(
            "Efficiency and range premises",
            "Layer energy-impact records",
            "Registered deterministic energy outputs where available",
        ),
        decision_rules=(
            "PASS: validated evidence demonstrates compliance.",
            "CONDITIONAL: compliance depends on unresolved operating assumptions.",
            "FAIL: validated evidence demonstrates a target violation.",
            "INSUFFICIENT_INFORMATION: required energy evidence is unavailable.",
        ),
        output_requirements=(
            "Efficiency target",
            "Layer contributions",
            "Range assumptions",
            "Conflicts",
            "Final verdict",
        ),
    ),
    "cost_mek": PropertyRubric(
        property_id="cost_mek",
        property_name="Cost vs. MEK",
        objective="Evaluate concept cost implications against the defined vehicle cost target.",
        evaluation_dimensions=(
            "Cost drivers",
            "Target deviation",
            "Cost offsets",
            "Cost uncertainty",
        ),
        required_evidence=(
            "Program MEK target",
            "Layer cost-impact records",
            "Registered deterministic cost outputs where available",
        ),
        decision_rules=(
            "PASS: validated cost evidence demonstrates compliance.",
            "CONDITIONAL: compliance depends on identified offsets or unresolved assumptions.",
            "FAIL: validated evidence demonstrates a target violation.",
            "INSUFFICIENT_INFORMATION: cost target or evidence is unavailable.",
        ),
        output_requirements=(
            "Cost target",
            "Validated cost result",
            "Major cost drivers",
            "Offset opportunities",
            "Final verdict",
        ),
    ),
}
