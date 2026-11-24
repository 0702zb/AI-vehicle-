"""Synthetic deterministic Reality Check fixture for demo and Studio runs.

The fixture is explicitly marked as demo data. It exists so the graph can show
a PASS -> END path without pretending that an LLM calculated a certified metric.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any

from langgraph.graph import END

from concept_platform.artifacts import ArtifactStore
from concept_platform.engineering import EngineeringMetric, EngineeringResultArtifact, ModelPurpose
from concept_platform.reality import (
    EngineeringResultReferences,
    InMemoryRuleRepository,
    RealityCheckService,
    RuleDefinition,
    RuleEvaluationContext,
    RealityContextResolver,
    make_reality_check_node,
)
from concept_platform.state import NodeStatus, RunState

PROJECT_ID = "demo-project"
CONCEPT_ID = "demo-concept"
CONCEPT_VERSION = "brief-v1"
EVALUATION_VERSION = "eval-v1"
SNAPSHOT_VERSION = "demo-rules-v1"
SOURCE_REF = "demo-premises-mass"
SOURCE_VERSION = "brief-v1"


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    async def put_json(self, payload: Mapping[str, Any]) -> str:
        data = json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        artifact_id = "sha256:" + hashlib.sha256(data).hexdigest()
        self._items[artifact_id] = copy.deepcopy(dict(payload))
        return artifact_id

    async def get_json(self, artifact_id: str) -> dict[str, Any]:
        if artifact_id not in self._items:
            raise FileNotFoundError(artifact_id)
        return copy.deepcopy(self._items[artifact_id])


class DemoResultReferences:
    """Create a synthetic, context-bound engineering result for any demo run."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def list_result_refs(
        self, state: RunState, context: RuleEvaluationContext
    ) -> list[str]:
        result = EngineeringResultArtifact(
            status=NodeStatus.SUPPORTED,
            run_id=context.run_id,
            project_id=context.project_id,
            concept_id=context.concept_id,
            concept_version=context.concept_version,
            brief_version=context.brief_version,
            evaluation_version=context.evaluation_version,
            context_manifest_hash=context.context_manifest_hash,
            model_id="demo-mass-model",
            model_version="demo-mass-model-v1",
            model_purpose=ModelPurpose.DEMO,
            solver_config_version="demo-solver-v1",
            input_artifact_ref="sha256:demo-input",
            metrics={
                "mass_kg": EngineeringMetric(
                    value=Decimal("1580"),
                    unit="kg",
                    source_component="synthetic-demo-fixture",
                )
            },
            warnings=("SYNTHETIC DEMO RESULT - NOT CERTIFIED ENGINEERING OUTPUT",),
        )
        return [await self._store.put_json(result.model_dump(mode="json"))]


class DemoContextResolver:
    async def resolve(self, run_id: str, state: RunState) -> RuleEvaluationContext:
        return RuleEvaluationContext(
            run_id=state["run_id"],
            project_id=PROJECT_ID,
            concept_id=CONCEPT_ID,
            concept_version=state["brief_version"],
            brief_version=state["brief_version"],
            evaluation_version=EVALUATION_VERSION,
            context_manifest_hash=state["context_manifest_hash"],
            market="CN",
            vehicle_family="demo",
            effective_date=date(2026, 11, 20),
            snapshot_version=SNAPSHOT_VERSION,
            permission_tags=frozenset(),
            allowed_sources={SOURCE_REF: SOURCE_VERSION},
        )


def build_demo_reality_check_node():
    store = InMemoryArtifactStore()
    rules = InMemoryRuleRepository(
        [
            RuleDefinition(
                rule_id="demo-mass-hard-limit",
                version="rule-v1",
                project_id=PROJECT_ID,
                metric_id="mass_kg",
                operator="lte",
                threshold=Decimal("1600"),
                unit="kg",
                hard_constraint=True,
                market=None,
                vehicle_family=None,
                effective_from=date(2026, 1, 1),
                effective_to=None,
                snapshot_version=SNAPSHOT_VERSION,
                source_ref=SOURCE_REF,
                source_version=SOURCE_VERSION,
                required_permissions=frozenset(),
                approved=True,
            )
        ]
    )
    service = RealityCheckService(
        rule_repository=rules,
        artifact_store=store,
        result_references=DemoResultReferences(store),
    )
    return make_reality_check_node(service, DemoContextResolver())


def route_after_demo_reality_check(state: RunState) -> str:
    """PASS/UNKNOWN end the demo; FAIL returns to orchestrator."""

    decision = state["coverage"].get("reality_check_decision")
    if decision == "FAIL":
        return "orchestrator"
    return END
