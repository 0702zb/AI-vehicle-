from __future__ import annotations

import asyncio
from typing import Any

from concept_platform.agents import MockProvider, ProviderResponse
from concept_platform.assessment_schemas import PropertyVerdict
from concept_platform.property_agent import create_property_agent_node
from concept_platform.state import NodeStatus, RunState

CONCEPT = "轻量化骨架与低风阻轮廓之间需要文字性权衡，碰撞传力通道与可维修性不得缺失。"
DIRECTIVE = "用文字说明低风阻轮廓与轻量化骨架之间的取舍，并标注来源。"


def _state(*, with_rag: bool, tool_status: str = "unknown") -> RunState:
    tool_result: dict[str, Any] = {
        "status": tool_status,
        "summary": "车身文字性建议。",
        "directives": [DIRECTIVE],
        "evidence_refs": ["ev-body-01"] if with_rag else [],
        "rag_context": [
            {
                "evidence_id": "ev-body-01",
                "text": "低风阻与轻量化骨架的权衡。",
                "domain": "body",
            }
        ] if with_rag else [],
    }
    return {
        "run_id": "run-property-test",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "node_status": {},
        "coverage": {
            "selected_concept": CONCEPT,
            "layer_agent_trace": {
                "tool_calls": [
                    {
                        "tool_name": "body",
                        "arguments": {"focus": "低风阻轮廓与轻量化骨架的文字性权衡"},
                        "status": tool_status,
                    }
                ],
                "tool_results": {"body": tool_result},
            },
        },
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


def _proposal(*, basis: list[str], name: str = "低风阻与轻量化结构权衡") -> dict[str, Any]:
    return {
        "project_id": "demo-project",
        "concept_id": "demo-concept",
        "concept_version": "brief-v1",
        "context_manifest_hash": "sha256:demo",
        "status": "supported",
        "summary": "从 Layer directives 归纳工程属性。",
        "aggregation_summary": "三个领域均给出定性质量影响，方向存在分歧。",
        "target_comparison": "未提供注册质量目标，不能判定达标。",
        "evidence_refs": [],
        "model_refs": [],
        "assumptions": [],
        "risks": [],
        "questions": [],
        "properties": [
            {
                "property_id": "prop-aero-mass",
                "name": name,
                "status": "supported",
                "source_domains": ["body"],
                "basis": basis,
                "rationale": "根据车身 directive 归纳。",
                "validation_items": ["检查文字性权衡是否完整。"],
            }
        ],
    }


def test_insufficient_information_fills_missing_gaps() -> None:
    payload = _proposal(basis=["用文字说明低风阻轮廓与轻量化骨架之间的取舍"])
    payload["rubric_result"] = "INSUFFICIENT_INFORMATION"

    verdict = PropertyVerdict.model_validate(payload)

    assert verdict.missing_information
    assert "not available" in verdict.missing_information[0]


def test_property_support_ignores_layer_rag_and_tool_status() -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=_proposal(basis=["用文字说明低风阻轮廓与轻量化骨架之间的取舍"]),
                model_version="property-model-v1",
            )
        ]
    )
    patch = asyncio.run(
        create_property_agent_node(adapter=provider)(
            _state(with_rag=True, tool_status="unknown")
        )
    )

    assert patch["node_status"]["property_agent"] is NodeStatus.SUPPORTED
    trace = patch["coverage"]["property_agent_trace"]
    assert trace["support_basis"] == "frozen_concept_and_layer_directives"
    assert trace["properties"][0]["basis_traceable"] is True

    task_input = provider.requests[0].task_input
    assert set(task_input) == {
        "concept",
        "property_id",
        "rubric",
        "layer_assessments",
        "directives",
    }
    assert "evidence_refs" not in task_input
    assert "rag_context" not in task_input
    assert "layer_status" not in task_input


def test_property_support_is_same_when_layer_rag_is_off() -> None:
    def run(with_rag: bool) -> dict[str, Any]:
        node = create_property_agent_node(
            adapter=MockProvider(
                [
                    ProviderResponse(
                        structured_output=_proposal(basis=["用文字说明低风阻轮廓与轻量化骨架之间的取舍"]),
                        model_version="property-model-v1",
                    )
                ]
            )
        )
        return asyncio.run(node(_state(with_rag=with_rag)))

    on = run(True)
    off = run(False)

    assert on["node_status"]["property_agent"] is NodeStatus.SUPPORTED
    assert off["node_status"]["property_agent"] is NodeStatus.SUPPORTED
    assert on["coverage"]["property_agent_trace"] == off["coverage"]["property_agent_trace"]


def test_untraceable_basis_downgrades_supported_to_unknown() -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=_proposal(basis=["完全不在输入中的凭空依据"]),
                model_version="property-model-v1",
            )
        ]
    )
    patch = asyncio.run(
        create_property_agent_node(adapter=provider)(_state(with_rag=True))
    )

    assert patch["node_status"]["property_agent"] is NodeStatus.UNKNOWN
    item = patch["coverage"]["property_agent_trace"]["properties"][0]
    assert item["proposed_status"] == "supported"
    assert item["status"] == "unknown"
    assert item["basis_traceable"] is False
    assert "basis_not_traceable" in item["adjustments"]


def test_unbacked_numeric_claim_downgrades_supported_to_assumption() -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=_proposal(
                    basis=["用文字说明低风阻轮廓与轻量化骨架之间的取舍"],
                    name="续航达到 600 km",
                ),
                model_version="property-model-v1",
            )
        ]
    )
    patch = asyncio.run(
        create_property_agent_node(adapter=provider)(_state(with_rag=True))
    )

    assert patch["node_status"]["property_agent"] is NodeStatus.ASSUMPTION
    item = patch["coverage"]["property_agent_trace"]["properties"][0]
    assert "unbacked_numeric_claim" in item["adjustments"]


def test_property_agent_is_blocked_without_layer_trace() -> None:
    state = _state(with_rag=False)
    state["coverage"].pop("layer_agent_trace")

    patch = asyncio.run(create_property_agent_node(adapter=MockProvider([]))(state))

    assert patch["node_status"]["property_agent"] is NodeStatus.BLOCKED
    assert patch["coverage"]["property_agent_trace"]["properties"] == []
    assert patch["coverage"]["errors"][0]["source"] == "property_agent"


def test_top_level_blocked_does_not_override_supported_properties() -> None:
    payload = _proposal(basis=["用文字说明低风阻轮廓与轻量化骨架之间的取舍"])
    payload["status"] = "blocked"
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=payload,
                model_version="property-model-v1",
            )
        ]
    )
    patch = asyncio.run(
        create_property_agent_node(adapter=provider)(_state(with_rag=True))
    )
    trace = patch["coverage"]["property_agent_trace"]

    assert patch["node_status"]["property_agent"] is NodeStatus.SUPPORTED
    assert trace["proposed_top_level_status"] == "blocked"
    assert trace["status"] == "supported"
    assert trace["properties"][0]["status"] == "supported"


def test_model_blocked_with_traceable_basis_stays_blocked() -> None:
    payload = _proposal(basis=["用文字说明低风阻轮廓与轻量化骨架之间的取舍"])
    payload["status"] = "blocked"
    payload["properties"][0]["status"] = "blocked"
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=payload,
                model_version="property-model-v1",
            )
        ]
    )
    patch = asyncio.run(
        create_property_agent_node(adapter=provider)(_state(with_rag=True))
    )
    item = patch["coverage"]["property_agent_trace"]["properties"][0]

    assert patch["node_status"]["property_agent"] is NodeStatus.BLOCKED
    assert item["proposed_status"] == "blocked"
    assert item["status"] == "blocked"


def test_property_agent_consumes_layer_assessment_impacts() -> None:
    impact_text = "钢铝混合材料选择会影响车身质量与维修性。"
    state = _state(with_rag=False)
    state["coverage"]["layer_agent_trace"]["tool_results"]["body"]["assessment"] = {
        "layer_id": "body",
        "directives": [],
        "property_impacts": [
            {
                "property_id": "mass",
                "description": impact_text,
                "evidence_class": "INFERRED",
                "source_ids": [],
                "engineering_implications": "材料变化会传导到整车质量。",
                "confidence": "MEDIUM",
            }
        ],
    }
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=_proposal(
                    basis=[impact_text],
                    name="车身材料质量影响",
                ),
                model_version="property-model-v1",
            )
        ]
    )
    patch = asyncio.run(
        create_property_agent_node(adapter=provider, property_id="mass")(state)
    )
    trace = patch["coverage"]["property_agent_trace"]
    task_input = provider.requests[0].task_input

    assert patch["node_status"]["property_agent"] is NodeStatus.SUPPORTED
    assert trace["property_id"] == "mass"
    assert trace["assessment_count"] == 1
    assert trace["source_domains"] == ["body"]
    assert len(task_input["layer_assessments"][0]["property_impacts"]) == 1
