from concept_platform.engineering_knowledge import DOMAIN_PACKS
from concept_platform.layer_agents import LAYER_ROUTER_SYSTEM_INSTRUCTION, SHARED_RULES
from concept_platform.prompts import (
    LAYER_REQUIRED_OUTPUT_RULE,
    LAYER_SHARED_SYSTEM_PROMPT,
    PROPERTY_REQUIRED_OUTPUT_RULE,
    SCOPE_ENVELOPE_RULE,
    render_domain_pack,
)
from concept_platform.property_agent import SYSTEM_INSTRUCTION as PROPERTY_SYSTEM_INSTRUCTION


def test_layer_prompts_require_verbatim_scope_echo() -> None:
    assert SCOPE_ENVELOPE_RULE in LAYER_SHARED_SYSTEM_PROMPT
    assert SCOPE_ENVELOPE_RULE in LAYER_ROUTER_SYSTEM_INSTRUCTION
    assert SCOPE_ENVELOPE_RULE in PROPERTY_SYSTEM_INSTRUCTION
    for field in ("project_id", "concept_id", "concept_version", "context_manifest_hash"):
        assert field in SCOPE_ENVELOPE_RULE


def test_required_output_rules_are_injected() -> None:
    assert LAYER_REQUIRED_OUTPUT_RULE in LAYER_SHARED_SYSTEM_PROMPT
    assert PROPERTY_REQUIRED_OUTPUT_RULE in PROPERTY_SYSTEM_INSTRUCTION
    for field in (
        "project_id",
        "status",
        "layer_id",
        "summary",
        "property_impacts",
        "overall_assessment",
    ):
        assert field in LAYER_REQUIRED_OUTPUT_RULE
    assert "Do not compare with other domains" in LAYER_REQUIRED_OUTPUT_RULE
    assert "cross-domain dependencies" in LAYER_REQUIRED_OUTPUT_RULE
    assert "architecture report" in LAYER_SHARED_SYSTEM_PROMPT
    for field in ("aggregation_summary", "target_comparison", "rationale", "missing_information"):
        assert field in PROPERTY_REQUIRED_OUTPUT_RULE


def test_domain_prompt_sections_have_no_cross_domain_content() -> None:
    for domain in ("body", "chassis", "energy"):
        text = render_domain_pack(DOMAIN_PACKS[domain])
        for key in ("critical_interfaces", "expected_handoff", "assessment_dimensions"):
            assert key not in text
        for other in ("Chassis:", "Body:", "Energy:", "Thermal:", "E/E:"):
            assert other not in text
    assert "其它领域" in SHARED_RULES
