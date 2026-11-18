"""Typed, versioned Reality Check rules; no dynamic code execution or eval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from concept_platform.artifacts import ArtifactStore
from concept_platform.engineering import EngineeringMetric, EngineeringResultArtifact
from concept_platform.state import NodeStatus, RunState


class RuleOperator(StrEnum):
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    GTE = "gte"
    GT = "gt"
    BETWEEN = "between"


class RuleDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    operator: RuleOperator
    threshold: Decimal
    upper_threshold: Decimal | None = None
    unit: str = Field(min_length=1)
    hard_constraint: bool
    market: str | None = None
    vehicle_family: str | None = None
    effective_from: date
    effective_to: date | None = None
    snapshot_version: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    required_permissions: frozenset[str] = frozenset()
    approved: bool

    def model_post_init(self, __context: Any, /) -> None:
        if self.operator is RuleOperator.BETWEEN:
            if self.upper_threshold is None or self.threshold > self.upper_threshold:
                raise ValueError("BETWEEN rules require an ordered upper threshold.")
        elif self.upper_threshold is not None:
            raise ValueError("Only BETWEEN rules may define upper_threshold.")


class RuleEvaluationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    project_id: str
    concept_id: str
    concept_version: str
    brief_version: str
    evaluation_version: str
    context_manifest_hash: str
    market: str
    vehicle_family: str
    effective_date: date
    snapshot_version: str
    permission_tags: frozenset[str]
    allowed_sources: dict[str, str]


class RuleDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class RuleEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    rule_version: str
    metric_id: str
    decision: RuleDecision
    hard_constraint: bool
    actual_value: Decimal | None = None
    threshold: Decimal | None = None
    upper_threshold: Decimal | None = None
    unit: str | None = None
    actual_unit: str | None = None
    delta: Decimal | None = None
    source_ref: str | None = None
    source_version: str | None = None
    reason: str


class RealityCheckArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: str = "reality_check"
    run_id: str
    project_id: str
    concept_id: str
    concept_version: str
    brief_version: str
    evaluation_version: str
    context_manifest_hash: str
    rule_snapshot_version: str
    gate_decision: RuleDecision
    evaluations: tuple[RuleEvaluation, ...]
    engineering_result_refs: tuple[str, ...]


class RuleRepository(Protocol):
    async def load_applicable(
        self, context: RuleEvaluationContext
    ) -> Sequence[RuleDefinition]: ...


class EngineeringResultReferences(Protocol):
    async def list_result_refs(
        self, state: RunState, context: RuleEvaluationContext
    ) -> Sequence[str]: ...


class StaticEngineeringResultReferences:
    """Uses only result IDs already pinned in RunState."""

    async def list_result_refs(
        self, state: RunState, context: RuleEvaluationContext
    ) -> Sequence[str]:
        return state["result_ids"]


class InMemoryRuleRepository:
    def __init__(self, rules: Sequence[RuleDefinition]) -> None:
        self._rules = tuple(rules)

    async def load_applicable(
        self, context: RuleEvaluationContext
    ) -> Sequence[RuleDefinition]:
        return tuple(
            rule
            for rule in self._rules
            if rule.approved
            and rule.project_id == context.project_id
            and (rule.market is None or rule.market == context.market)
            and (rule.vehicle_family is None or rule.vehicle_family == context.vehicle_family)
            and rule.effective_from <= context.effective_date
            and (rule.effective_to is None or context.effective_date <= rule.effective_to)
            and rule.snapshot_version == context.snapshot_version
            and rule.required_permissions.issubset(context.permission_tags)
        )


POSTGRES_RULE_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS rule_definitions (
        project_id TEXT NOT NULL,
        rule_id TEXT NOT NULL,
        version TEXT NOT NULL,
        metric_id TEXT NOT NULL,
        operator TEXT NOT NULL CHECK (operator IN ('lt','lte','eq','gte','gt','between')),
        threshold NUMERIC NOT NULL,
        upper_threshold NUMERIC,
        unit TEXT NOT NULL,
        hard_constraint BOOLEAN NOT NULL,
        market TEXT,
        vehicle_family TEXT,
        effective_from DATE NOT NULL,
        effective_to DATE,
        snapshot_version TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        source_version TEXT NOT NULL,
        required_permissions TEXT[] NOT NULL DEFAULT '{}',
        approved BOOLEAN NOT NULL,
        approval_actor TEXT NOT NULL,
        PRIMARY KEY (project_id, rule_id, version),
        CHECK ((operator = 'between' AND upper_threshold IS NOT NULL)
            OR (operator <> 'between' AND upper_threshold IS NULL))
    )""",
    """CREATE INDEX IF NOT EXISTS rule_applicability_idx
        ON rule_definitions (project_id, market, vehicle_family, snapshot_version,
                             effective_from, effective_to, approved)""",
    """CREATE OR REPLACE FUNCTION reject_rule_version_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'rule_definitions is versioned and append-only';
        END;
        $$ LANGUAGE plpgsql""",
    """DROP TRIGGER IF EXISTS rule_definitions_immutable ON rule_definitions""",
    """CREATE TRIGGER rule_definitions_immutable
        BEFORE UPDATE OR DELETE ON rule_definitions
        FOR EACH ROW EXECUTE FUNCTION reject_rule_version_mutation()""",
)


class PostgresRuleRepository:
    def __init__(self, connection_pool: Any) -> None:
        self._connection_pool = connection_pool

    async def setup(self) -> None:
        async with self._connection_pool.connection() as connection, connection.transaction():
            for statement in POSTGRES_RULE_SCHEMA:
                await connection.execute(statement)

    async def load_applicable(
        self, context: RuleEvaluationContext
    ) -> Sequence[RuleDefinition]:
        async with self._connection_pool.connection() as connection:
            return await self._load_applicable(connection, context)

    async def _load_applicable(
        self, connection: Any, context: RuleEvaluationContext
    ) -> Sequence[RuleDefinition]:
        cursor = await connection.execute(
            """
            SELECT rule_id, version, project_id, metric_id, operator, threshold,
                   upper_threshold, unit, hard_constraint, market, vehicle_family,
                   effective_from, effective_to, snapshot_version, source_ref,
                   source_version, required_permissions, approved
            FROM rule_definitions
            WHERE project_id = %s
              AND approved = TRUE
              AND (market IS NULL OR market = %s)
              AND (vehicle_family IS NULL OR vehicle_family = %s)
              AND effective_from <= %s
              AND (effective_to IS NULL OR effective_to >= %s)
              AND snapshot_version = %s
              AND required_permissions <@ %s
            ORDER BY rule_id, version
            """,
            (
                context.project_id,
                context.market,
                context.vehicle_family,
                context.effective_date,
                context.effective_date,
                context.snapshot_version,
                list(context.permission_tags),
            ),
        )
        rows = await cursor.fetchall()
        return tuple(
            RuleDefinition(
                rule_id=row[0],
                version=row[1],
                project_id=row[2],
                metric_id=row[3],
                operator=row[4],
                threshold=row[5],
                upper_threshold=row[6],
                unit=row[7],
                hard_constraint=row[8],
                market=row[9],
                vehicle_family=row[10],
                effective_from=row[11],
                effective_to=row[12],
                snapshot_version=row[13],
                source_ref=row[14],
                source_version=row[15],
                required_permissions=frozenset(row[16]),
                approved=row[17],
            )
            for row in rows
        )


@dataclass(frozen=True)
class RealityCheckOutcome:
    status: NodeStatus
    artifact_ref: str
    gate_decision: RuleDecision


class RealityCheckService:
    def __init__(
        self,
        *,
        rule_repository: RuleRepository,
        artifact_store: ArtifactStore,
        result_references: EngineeringResultReferences | None = None,
    ) -> None:
        self._rules = rule_repository
        self._artifacts = artifact_store
        self._result_references = result_references or StaticEngineeringResultReferences()

    async def evaluate(
        self, state: RunState, context: RuleEvaluationContext
    ) -> RealityCheckOutcome:
        self._validate_run_binding(state, context)
        rules = await self._rules.load_applicable(context)
        for rule in rules:
            self._validate_rule_scope(rule, context)

        result_refs = tuple(await self._result_references.list_result_refs(state, context))
        engineering_results: list[tuple[str, EngineeringResultArtifact]] = []
        for ref in result_refs:
            try:
                payload = await self._artifacts.get_json(ref)
            except (FileNotFoundError, ValueError):
                continue
            if payload.get("artifact_kind") != "engineering_result":
                continue
            result = EngineeringResultArtifact.model_validate(payload)
            if not self._result_matches_context(result, context):
                continue
            engineering_results.append((ref, result))

        evaluations = tuple(self._evaluate_rule(rule, engineering_results) for rule in rules)
        hard_results = [item.decision for item in evaluations if item.hard_constraint]
        if not hard_results:
            gate_decision = RuleDecision.UNKNOWN
        elif RuleDecision.FAIL in hard_results:
            gate_decision = RuleDecision.FAIL
        elif RuleDecision.UNKNOWN in hard_results:
            gate_decision = RuleDecision.UNKNOWN
        else:
            gate_decision = RuleDecision.PASS

        node_status = {
            RuleDecision.PASS: NodeStatus.SUPPORTED,
            RuleDecision.FAIL: NodeStatus.FAIL,
            RuleDecision.UNKNOWN: NodeStatus.UNKNOWN,
        }[gate_decision]
        artifact = RealityCheckArtifact(
            run_id=context.run_id,
            project_id=context.project_id,
            concept_id=context.concept_id,
            concept_version=context.concept_version,
            brief_version=context.brief_version,
            evaluation_version=context.evaluation_version,
            context_manifest_hash=context.context_manifest_hash,
            rule_snapshot_version=context.snapshot_version,
            gate_decision=gate_decision,
            evaluations=evaluations,
            engineering_result_refs=tuple(ref for ref, _result in engineering_results),
        )
        artifact_ref = await self._artifacts.put_json(artifact.model_dump(mode="json"))
        return RealityCheckOutcome(node_status, artifact_ref, gate_decision)

    @staticmethod
    def _validate_run_binding(state: RunState, context: RuleEvaluationContext) -> None:
        if state["run_id"] != context.run_id:
            raise ValueError("Reality Check context run_id mismatch.")
        if state["brief_version"] != context.brief_version:
            raise ValueError("Reality Check context Brief version mismatch.")
        if state["context_manifest_hash"] != context.context_manifest_hash:
            raise ValueError("Reality Check manifest mismatch.")

    @staticmethod
    def _validate_rule_scope(rule: RuleDefinition, context: RuleEvaluationContext) -> None:
        if not rule.approved:
            raise ValueError("Unapproved rule passed the repository boundary.")
        if rule.project_id != context.project_id:
            raise ValueError("Cross-project rule reference rejected.")
        if rule.market is not None and rule.market != context.market:
            raise ValueError("Rule market does not match the frozen evaluation.")
        if rule.vehicle_family is not None and rule.vehicle_family != context.vehicle_family:
            raise ValueError("Rule vehicle family does not match the frozen evaluation.")
        if not rule.effective_from <= context.effective_date:
            raise ValueError("Rule is not yet effective for this evaluation.")
        if rule.effective_to and context.effective_date > rule.effective_to:
            raise ValueError("Rule is no longer effective for this evaluation.")
        if rule.snapshot_version != context.snapshot_version:
            raise ValueError("Rule snapshot version does not match the evaluation.")
        if not rule.required_permissions.issubset(context.permission_tags):
            raise ValueError("Rule is outside the evaluator permission scope.")
        if context.allowed_sources.get(rule.source_ref) != rule.source_version:
            raise ValueError("Rule source is absent from the authorized evidence snapshot.")

    @staticmethod
    def _result_matches_context(
        result: EngineeringResultArtifact, context: RuleEvaluationContext
    ) -> bool:
        return all(
            (
                result.run_id == context.run_id,
                result.project_id == context.project_id,
                result.concept_id == context.concept_id,
                result.concept_version == context.concept_version,
                result.brief_version == context.brief_version,
                result.evaluation_version == context.evaluation_version,
                result.context_manifest_hash == context.context_manifest_hash,
            )
        )

    @staticmethod
    def _evaluate_rule(
        rule: RuleDefinition,
        results: Sequence[tuple[str, EngineeringResultArtifact]],
    ) -> RuleEvaluation:
        metrics: list[EngineeringMetric] = []
        dependency_reasons: list[str] = []
        for _ref, result in results:
            metric = result.metrics.get(rule.metric_id)
            if metric is not None:
                if result.status != NodeStatus.SUPPORTED:
                    dependency_reasons.append("engineering_result_not_supported")
                else:
                    metrics.append(metric)

        base = {
            "rule_id": rule.rule_id,
            "rule_version": rule.version,
            "metric_id": rule.metric_id,
            "hard_constraint": rule.hard_constraint,
            "threshold": rule.threshold,
            "upper_threshold": rule.upper_threshold,
            "unit": rule.unit,
            "source_ref": rule.source_ref,
            "source_version": rule.source_version,
        }
        if dependency_reasons:
            return RuleEvaluation(
                **base,
                decision=RuleDecision.UNKNOWN,
                reason=dependency_reasons[0],
            )
        if not metrics:
            return RuleEvaluation(**base, decision=RuleDecision.UNKNOWN, reason="metric_result_missing")
        if len(metrics) != 1:
            return RuleEvaluation(
                **base,
                decision=RuleDecision.UNKNOWN,
                reason="ambiguous_multiple_metric_results",
            )
        metric = metrics[0]
        if metric.unit != rule.unit:
            return RuleEvaluation(
                **base,
                decision=RuleDecision.UNKNOWN,
                actual_value=metric.value,
                actual_unit=metric.unit,
                reason="metric_rule_unit_mismatch",
            )

        actual = metric.value
        if rule.operator is RuleOperator.LT:
            passed = actual < rule.threshold
        elif rule.operator is RuleOperator.LTE:
            passed = actual <= rule.threshold
        elif rule.operator is RuleOperator.EQ:
            passed = actual == rule.threshold
        elif rule.operator is RuleOperator.GTE:
            passed = actual >= rule.threshold
        elif rule.operator is RuleOperator.GT:
            passed = actual > rule.threshold
        else:
            passed = rule.threshold <= actual <= rule.upper_threshold
        return RuleEvaluation(
            **base,
            decision=RuleDecision.PASS if passed else RuleDecision.FAIL,
            actual_value=actual,
            actual_unit=metric.unit,
            delta=actual - rule.threshold,
            reason="typed_rule_passed" if passed else "typed_rule_violated",
        )


class RealityContextResolver(Protocol):
    async def resolve(self, run_id: str, state: RunState) -> RuleEvaluationContext: ...


def make_reality_check_node(
    service: RealityCheckService, context_resolver: RealityContextResolver
):
    async def reality_check_node(state: RunState) -> dict[str, Any]:
        context = await context_resolver.resolve(state["run_id"], state)
        outcome = await service.evaluate(state, context)
        return {
            "node_status": {"reality_check": outcome.status},
            "coverage": {"reality_check_decision": outcome.gate_decision.value},
            "result_ids": [outcome.artifact_ref],
        }

    return reality_check_node
