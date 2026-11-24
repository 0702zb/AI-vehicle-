"""User-facing projection of a finished graph run.

The graph state keeps full audit traces in ``coverage``. This module projects that
state into the small set of facts a user needs, keeping the ORIGINAL FULL TEXT of
concept, domain conclusions and property summary so the result can be verified.
Truncation is only applied by :func:`render_summary_text` for terminal display.

It is a pure function: it never mutates the state or touches external services.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from concept_platform.state import NodeStatus

_STATUS_LABEL: dict[str, str] = {
    NodeStatus.SUPPORTED.value: "有充分依据",
    NodeStatus.ASSUMPTION.value: "依赖假设",
    NodeStatus.UNKNOWN.value: "信息不足",
    NodeStatus.BLOCKED.value: "未完成",
    NodeStatus.FAIL.value: "未通过",
}

_VERDICT_LABEL: dict[str, str] = {
    "PASS": "通过",
    "CONDITIONAL": "有条件通过",
    "FAIL": "未通过",
    "INSUFFICIENT_INFORMATION": "信息不足",
}


def _status_text(value: Any) -> str:
    raw = value.value if isinstance(value, NodeStatus) else str(value or "")
    return _STATUS_LABEL.get(raw, raw or "未知")


def _full_text(value: Any) -> str:
    return str(value or "")


def _shorten(value: Any, limit: int = 60) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _domain_results(state: Mapping[str, Any]) -> list[dict[str, str]]:
    trace = state.get("coverage", {}).get("layer_agent_trace") or {}
    if not isinstance(trace, Mapping):
        return []
    results = trace.get("tool_results") or {}
    if not isinstance(results, Mapping):
        return []
    rows = []
    for domain, result in results.items():
        if not isinstance(result, Mapping):
            continue
        rows.append({"domain": str(domain), "conclusion": _full_text(result.get("summary"))})
    return rows


def _reality_evaluations(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    coverage = state.get("coverage") or {}
    items = coverage.get("reality_check_evaluations") or []
    rows = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "rule_id": str(item.get("rule_id") or ""),
                "rule_version": str(item.get("rule_version") or ""),
                "metric_id": str(item.get("metric_id") or ""),
                "decision": str(item.get("decision") or ""),
                "hard_constraint": bool(item.get("hard_constraint")),
                "actual_value": str(item.get("actual_value")) if item.get("actual_value") is not None else "",
                "threshold": str(item.get("threshold")) if item.get("threshold") is not None else "",
                "upper_threshold": str(item.get("upper_threshold")) if item.get("upper_threshold") is not None else "",
                "unit": str(item.get("unit") or ""),
                "actual_unit": str(item.get("actual_unit") or ""),
                "delta": str(item.get("delta")) if item.get("delta") is not None else "",
                "reason": str(item.get("reason") or ""),
                "source_ref": str(item.get("source_ref") or ""),
                "source_version": str(item.get("source_version") or ""),
            }
        )
    return rows


def _property_result(state: Mapping[str, Any]) -> dict[str, str]:
    trace = state.get("coverage", {}).get("property_agent_trace") or {}
    if not isinstance(trace, Mapping):
        return {}
    verdict = str(trace.get("rubric_result") or "")
    return {
        "property_id": str(trace.get("property_id") or ""),
        "verdict": verdict,
        "verdict_label": _VERDICT_LABEL.get(verdict, verdict or "未评估"),
        "summary": _full_text(trace.get("aggregation_summary")),
    }


def build_user_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project the full state into a verification-ready dictionary with full text."""
    coverage = state.get("coverage") or {}
    statuses = state.get("node_status") or {}
    overall = statuses.get("reality_check") or statuses.get("property_agent") or ""

    return {
        "concept": _full_text(coverage.get("selected_concept")),
        "domains": _domain_results(state),
        "property": _property_result(state),
        "reality_check": {
            "decision": str(coverage.get("reality_check_decision") or ""),
            "status": _status_text(statuses.get("reality_check")),
            "rule_snapshot": str(coverage.get("reality_check_rule_snapshot") or ""),
            "evaluations": _reality_evaluations(state),
        },
        "overall_status": _status_text(overall),
    }


def render_summary_text(summary: Mapping[str, Any]) -> str:
    """Render a compact text block for the terminal (truncates long fields)."""
    lines = ["概念工程评估摘要", "=" * 40]
    lines.append(f"输入概念：{_shorten(summary.get('concept'), 70)}")

    domains = summary.get("domains") or []
    lines.append("")
    lines.append("领域架构分析：")
    if domains:
        for row in domains:
            lines.append(f"  - {row['domain']}：{_shorten(row['conclusion'])}")
    else:
        lines.append("  （无）")

    prop = summary.get("property") or {}
    lines.append("")
    lines.append("属性评估：")
    if prop:
        lines.append(
            f"  {prop.get('property_id') or 'property'}：{prop.get('verdict_label') or '-'}"
        )
        lines.append(f"  说明：{_shorten(prop.get('summary'))}")
    else:
        lines.append("  （无）")

    check = summary.get("reality_check") or {}
    lines.append("")
    lines.append(f"现实校验：{check.get('decision') or '-'}（{check.get('status') or '-'}）")
    for item in check.get("evaluations") or []:
        actual = item.get("actual_value") or "-"
        threshold = item.get("threshold") or "-"
        unit = item.get("unit") or ""
        lines.append(
            f"  规则 {item.get('rule_id')} [{item.get('decision')}] "
            f"{item.get('metric_id')}={actual}{item.get('actual_unit') or unit} "
            f"阈值 {threshold}{unit} delta={item.get('delta') or '-'}"
        )
        lines.append(f"    原因：{item.get('reason')}")
    lines.append(f"总体状态：{summary.get('overall_status') or '-'}")
    return "\n".join(lines)
