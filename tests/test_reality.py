from datetime import date
from decimal import Decimal

import pytest

from concept_platform.artifacts import LocalArtifactStore
from concept_platform.engineering import EngineeringMetric, EngineeringResultArtifact
from concept_platform.reality import (
    POSTGRES_RULE_SCHEMA,
    InMemoryRuleRepository,
    RealityCheckService,
    RuleDecision,
    RuleDefinition,
    RuleEvaluationContext,
)
from concept_platform.state import NodeStatus, RunState


def context(**overrides):
    values = {
        "run_id": "run-rules",
        "project_id": "project-a",
        "concept_id": "concept-1",
        "concept_version": "concept-v1",
        "brief_version": "brief-v1",
        "evaluation_version": "eval-v1",
        "context_manifest_hash": "sha256:manifest-v1",
        "market": "CN",
        "vehicle_family": "SUV",
        "effective_date": date(2026, 11, 18),
        "snapshot_version": "rules-snapshot-v3",
        "permission_tags": frozenset({"engineering.read", "market.cn"}),
        "allowed_sources": {"sha256:rule-source": "evidence-v1"},
    }
    values.update(overrides)
    return RuleEvaluationContext(**values)


def state(result_ids=()) -> RunState:
    return {
        "run_id": "run-rules",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:manifest-v1",
        "node_status": {},
        "coverage": {},
        "result_ids": list(result_ids),
        "approval_version": None,
        "pending_writes": [],
    }


def rule(**overrides):
    values = {
        "rule_id": "range-minimum",
        "version": "rule-v2",
        "project_id": "project-a",
        "metric_id": "range",
        "operator": "lte",
        "threshold": Decimal(400),
        "unit": "km",
        "hard_constraint": True,
        "market": "CN",
        "vehicle_family": "SUV",
        "effective_from": date(2026, 1, 1),
        "effective_to": None,
        "snapshot_version": "rules-snapshot-v3",
        "source_ref": "sha256:rule-source",
        "source_version": "evidence-v1",
        "required_permissions": frozenset({"engineering.read"}),
        "approved": True,
    }
    values.update(overrides)
    return RuleDefinition(**values)


async def engineering_result(store, *, status=NodeStatus.SUPPORTED, unit="km"):
    result = EngineeringResultArtifact(
        status=status,
        run_id="run-rules",
        project_id="project-a",
        concept_id="concept-1",
        concept_version="concept-v1",
        brief_version="brief-v1",
        evaluation_version="eval-v1",
        context_manifest_hash="sha256:manifest-v1",
        model_id="lds",
        model_version="lds-v1",
        model_purpose="formal",
        solver_config_version="solver-v1",
        input_artifact_ref="sha256:input-ref",
        metrics={"range": EngineeringMetric(value=Decimal(420), unit=unit)},
    )
    return await store.put_json(result.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_typed_reality_check_emits_code_owned_fail_and_does_not_offset_it(tmp_path):
    store = LocalArtifactStore(tmp_path)
    result_ref = await engineering_result(store)
    rules = InMemoryRuleRepository(
        [
            rule(),
            rule(
                rule_id="mass-soft-limit",
                metric_id="mass",
                threshold=Decimal(2000),
                operator="lte",
                unit="kg",
                hard_constraint=False,
                market=None,
                vehicle_family=None,
                required_permissions=frozenset(),
            ),
        ]
    )
    outcome = await RealityCheckService(
        rule_repository=rules, artifact_store=store
    ).evaluate(state([result_ref]), context())
    artifact = await store.get_json(outcome.artifact_ref)

    assert outcome.gate_decision is RuleDecision.FAIL
    assert outcome.status is NodeStatus.FAIL
    assert artifact["evaluations"][0]["decision"] == "FAIL"
    assert artifact["evaluations"][0]["actual_value"] == "420"
    assert artifact["evaluations"][0]["threshold"] == "400"
    assert artifact["evaluations"][0]["delta"] == "20"


@pytest.mark.asyncio
async def test_no_engineering_result_or_no_hard_rules_means_unknown_not_pass(tmp_path):
    store = LocalArtifactStore(tmp_path)
    service = RealityCheckService(
        rule_repository=InMemoryRuleRepository([rule()]), artifact_store=store
    )
    no_result = await service.evaluate(state(), context())
    soft_only = await RealityCheckService(
        rule_repository=InMemoryRuleRepository([rule(hard_constraint=False)]),
        artifact_store=store,
    ).evaluate(state(), context())

    assert no_result.gate_decision is RuleDecision.UNKNOWN
    assert no_result.status is NodeStatus.UNKNOWN
    assert soft_only.gate_decision is RuleDecision.UNKNOWN


@pytest.mark.asyncio
async def test_rule_applicability_filters_market_date_snapshot_and_permissions(tmp_path):
    store = LocalArtifactStore(tmp_path)
    rules = InMemoryRuleRepository(
        [
            rule(rule_id="wrong-market", market="EU"),
            rule(rule_id="not-yet-effective", effective_from=date(2027, 1, 1)),
            rule(rule_id="wrong-snapshot", snapshot_version="rules-v-old"),
            rule(rule_id="no-permission", required_permissions=frozenset({"safety.write"})),
            rule(rule_id="wrong-project", project_id="project-b"),
        ]
    )
    outcome = await RealityCheckService(rule_repository=rules, artifact_store=store).evaluate(
        state(), context()
    )
    artifact = await store.get_json(outcome.artifact_ref)

    assert outcome.gate_decision is RuleDecision.UNKNOWN
    assert artifact["evaluations"] == []


@pytest.mark.asyncio
async def test_incompatible_units_and_non_supported_engineering_results_are_unknown(tmp_path):
    store = LocalArtifactStore(tmp_path)
    unit_ref = await engineering_result(store, unit="mi")
    blocked_ref = await engineering_result(store, status=NodeStatus.BLOCKED)
    unit_check = await RealityCheckService(
        rule_repository=InMemoryRuleRepository([rule()]), artifact_store=store
    ).evaluate(state([unit_ref]), context())
    blocked_check = await RealityCheckService(
        rule_repository=InMemoryRuleRepository([rule()]), artifact_store=store
    ).evaluate(state([blocked_ref]), context())

    unit_artifact = await store.get_json(unit_check.artifact_ref)
    blocked_artifact = await store.get_json(blocked_check.artifact_ref)
    assert unit_check.gate_decision is RuleDecision.UNKNOWN
    assert unit_artifact["evaluations"][0]["reason"] == "metric_rule_unit_mismatch"
    assert blocked_check.gate_decision is RuleDecision.UNKNOWN
    assert blocked_artifact["evaluations"][0]["reason"] == "engineering_result_not_supported"


@pytest.mark.asyncio
async def test_stale_engineering_result_is_not_used_by_reality_check(tmp_path):
    store = LocalArtifactStore(tmp_path)
    result_ref = await engineering_result(store)
    stale = context(context_manifest_hash="sha256:other-manifest")
    stale_state = state([result_ref])
    stale_state["context_manifest_hash"] = stale.context_manifest_hash
    outcome = await RealityCheckService(
        rule_repository=InMemoryRuleRepository([rule()]), artifact_store=store
    ).evaluate(stale_state, stale)

    assert outcome.gate_decision is RuleDecision.UNKNOWN


def test_rule_schema_uses_operator_enum_and_never_executes_rule_text():
    with pytest.raises(ValueError):
        rule(operator="__import__('os').system('bad')")
    assert all("eval(" not in statement for statement in POSTGRES_RULE_SCHEMA)
    assert "CHECK (operator IN" in POSTGRES_RULE_SCHEMA[0]
