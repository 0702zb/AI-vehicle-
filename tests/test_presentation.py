from __future__ import annotations

from typing import Any

from concept_platform.presentation import build_user_summary, render_summary_text
from concept_platform.state import NodeStatus


def _state() -> dict[str, Any]:
    return {
        "node_status": {
            "layer_agent": NodeStatus.SUPPORTED,
            "property_agent": NodeStatus.UNKNOWN,
            "reality_check": NodeStatus.SUPPORTED,
        },
        "coverage": {
            "selected_concept": "轻量城市通勤舱概念",
            "layer_agent_trace": {
                "tool_results": {
                    "body": {
                        "summary": "车身采用钢铝混合骨架并兼顾低风阻造型与可维修性。"
                    }
                }
            },
            "property_agent_trace": {
                "property_id": "mass",
                "rubric_result": "INSUFFICIENT_INFORMATION",
                "aggregation_summary": "尚无确定性质量计算结果。",
            },
            "reality_check_decision": "PASS",
            "reality_check_rule_snapshot": "demo-rules-v1",
            "reality_check_evaluations": [
                {
                    "rule_id": "demo-mass-hard-limit",
                    "rule_version": "rule-v1",
                    "metric_id": "mass_kg",
                    "decision": "PASS",
                    "hard_constraint": True,
                    "actual_value": "1580",
                    "threshold": "1600",
                    "upper_threshold": None,
                    "unit": "kg",
                    "actual_unit": "kg",
                    "delta": "-20",
                    "reason": "typed_rule_passed",
                    "source_ref": "demo-premises-mass",
                    "source_version": "brief-v1",
                }
            ],
        },
    }


def test_summary_extracts_only_user_facing_fields() -> None:
    summary = build_user_summary(_state())

    assert summary["concept"] == "轻量城市通勤舱概念"
    assert [row["domain"] for row in summary["domains"]] == ["body"]
    assert summary["property"]["verdict_label"] == "信息不足"
    assert summary["reality_check"]["decision"] == "PASS"
    assert summary["reality_check"]["rule_snapshot"] == "demo-rules-v1"
    assert len(summary["reality_check"]["evaluations"]) == 1
    assert summary["overall_status"] == "有充分依据"
    assert "reasoning" not in summary
    assert "flow_trace" not in summary


def test_summary_preserves_full_text_for_verification() -> None:
    long_concept = " Concept " + "细节描述" * 60 + " "
    long_conclusion = "车身方案 " + "详细结论" * 60
    long_summary = "属性说明 " + "完整内容" * 60
    state = _state()
    state["coverage"]["selected_concept"] = long_concept
    state["coverage"]["layer_agent_trace"]["tool_results"]["body"]["summary"] = long_conclusion
    state["coverage"]["property_agent_trace"]["aggregation_summary"] = long_summary

    summary = build_user_summary(state)

    assert summary["concept"].strip() == long_concept.strip()
    assert summary["domains"][0]["conclusion"] == long_conclusion
    assert summary["property"]["summary"] == long_summary
    assert "…" not in summary["concept"]
    assert "…" not in summary["domains"][0]["conclusion"]


def test_summary_preserves_rule_evaluation_detail() -> None:
    item = build_user_summary(_state())["reality_check"]["evaluations"][0]

    assert item["rule_id"] == "demo-mass-hard-limit"
    assert item["decision"] == "PASS"
    assert item["actual_value"] == "1580"
    assert item["threshold"] == "1600"
    assert item["delta"] == "-20"
    assert item["reason"] == "typed_rule_passed"
    assert item["source_ref"] == "demo-premises-mass"


def test_render_summary_text_is_short() -> None:
    text = render_summary_text(build_user_summary(_state()))

    assert len(text.splitlines()) <= 16
    assert "概念工程评估摘要" in text
    assert "mass" in text
