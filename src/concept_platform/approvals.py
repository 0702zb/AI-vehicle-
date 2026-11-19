"""Human approval with immutable version binding and deterministic invalidation."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from concept_platform.artifacts import ArtifactStore
from concept_platform.state import NodeStatus, RunState


class EvaluationVersionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    concept_version: str = Field(min_length=1)
    evaluation_version: str = Field(min_length=1)
    evidence_manifest_hash: str = Field(min_length=1)
    model_rule_versions: Mapping[str, str]


@dataclass(frozen=True)
class RecalculationPlan:
    changed_dimensions: frozenset[str]
    affected_nodes: tuple[str, ...]
    approval_invalidated: bool


# Inputs are dependency roots; only affected downstream nodes are recomputed.
DEPENDENCY_NODES: dict[str, tuple[str, ...]] = {
    "concept_version": ("layer_agent", "property_agent", "reality_check", "expert_review", "concept_paper", "report_issuance"),
    "evaluation_version": ("reality_check", "expert_review", "concept_paper", "report_issuance"),
    "evidence_manifest_hash": ("layer_agent", "property_agent", "reality_check", "expert_review", "concept_paper", "report_issuance"),
    "model_rule_versions": ("layer_agent", "property_agent", "reality_check", "expert_review", "concept_paper", "report_issuance"),
}


def thread_id_for(bundle: EvaluationVersionBundle) -> str:
    """Stable opaque thread ID covering all approval-bound evaluation inputs."""

    material = {
        "run_id": bundle.run_id,
        "concept_version": bundle.concept_version,
        "evaluation_version": bundle.evaluation_version,
        "evidence_manifest_hash": bundle.evidence_manifest_hash,
        "model_rule_versions": dict(sorted(bundle.model_rule_versions.items())),
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return "run:" + digest


def plan_recalculation(
    previous: EvaluationVersionBundle, current: EvaluationVersionBundle
) -> RecalculationPlan:
    if previous.run_id != current.run_id:
        raise ValueError("Version invalidation cannot cross run IDs.")
    changed = frozenset(
        name
        for name in DEPENDENCY_NODES
        if getattr(previous, name) != getattr(current, name)
    )
    affected = {node for dimension in changed for node in DEPENDENCY_NODES[dimension]}
    ordered = tuple(node for node in _TOPOLOGICAL_ORDER if node in affected)
    return RecalculationPlan(changed, ordered, bool(changed))


_TOPOLOGICAL_ORDER = (
    "layer_agent", "property_agent", "reality_check",
    "expert_review", "concept_paper", "report_issuance",
)


class ApprovalContextResolver(Protocol):
    async def resolve(self, state: RunState) -> tuple[EvaluationVersionBundle, str, str, str]: ...


class ApprovalRepository(Protocol):
    async def store_approval(self, approval_ref: str, bundle: EvaluationVersionBundle) -> None: ...

    async def store_objection(self, objection_ref: str, bundle: EvaluationVersionBundle) -> None: ...

    async def load_approval(self, thread_id: str) -> str | None: ...

    async def has_objection(self, thread_id: str) -> bool: ...


class PostgresApprovalRepository:
    """Stores immutable approval decisions in a separate version-keyed table."""

    def __init__(self, connection_pool: Any) -> None:
        self._connection_pool = connection_pool

    async def setup(self) -> None:
        from concept_platform.outbox import APPROVAL_SCHEMA_STATEMENTS

        async with self._connection_pool.connection() as connection, connection.transaction():
            for statement in APPROVAL_SCHEMA_STATEMENTS:
                await connection.execute(statement)

    async def store_approval(self, approval_ref: str, bundle: EvaluationVersionBundle) -> None:
        async with self._connection_pool.connection() as connection, connection.transaction():
            cursor = await connection.execute(
                """INSERT INTO expert_approvals (
                    thread_id, run_id, approval_ref, concept_version, evaluation_version,
                    evidence_manifest_hash, model_rule_versions
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (thread_id) DO NOTHING RETURNING approval_ref""",
                (thread_id_for(bundle), bundle.run_id, approval_ref, bundle.concept_version,
                 bundle.evaluation_version, bundle.evidence_manifest_hash,
                 json.dumps(dict(bundle.model_rule_versions), sort_keys=True)),
            )
            row = await cursor.fetchone()
            if row is None:
                cursor = await connection.execute(
                    "SELECT approval_ref FROM expert_approvals WHERE thread_id = %s",
                    (thread_id_for(bundle),),
                )
                row = await cursor.fetchone()
                if row is None or row[0] != approval_ref:
                    raise ValueError("A different approval is already bound to this version thread.")
            await connection.execute(
                """INSERT INTO audit_events (
                    event_id, run_id, event_type, entity_id, entity_version, payload
                ) VALUES (%s, %s, 'expert_approval_recorded', %s, %s, %s::jsonb)
                ON CONFLICT (event_id) DO NOTHING""",
                (uuid.uuid5(uuid.NAMESPACE_URL, "approval:" + thread_id_for(bundle)), bundle.run_id, approval_ref, thread_id_for(bundle),
                 json.dumps({"thread_id": thread_id_for(bundle), "approval_ref": approval_ref})),
            )

    async def store_objection(self, objection_ref: str, bundle: EvaluationVersionBundle) -> None:
        async with self._connection_pool.connection() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO approval_objections (objection_ref, thread_id, run_id, version_bundle)
                VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT (objection_ref) DO NOTHING""",
                (objection_ref, thread_id_for(bundle), bundle.run_id,
                 json.dumps(bundle.model_dump(mode="json"), sort_keys=True)),
            )
            await connection.execute(
                """INSERT INTO audit_events (
                    event_id, run_id, event_type, entity_id, entity_version, payload
                ) VALUES (%s, %s, 'expert_objection_recorded', %s, %s, %s::jsonb)""",
                (uuid.uuid5(uuid.NAMESPACE_URL, "objection:" + objection_ref), bundle.run_id, objection_ref, thread_id_for(bundle),
                 json.dumps({"thread_id": thread_id_for(bundle), "objection_ref": objection_ref})),
            )

    async def load_approval(self, thread_id: str) -> str | None:
        async with self._connection_pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT approval_ref FROM expert_approvals WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return str(row[0]) if row else None

    async def has_objection(self, thread_id: str) -> bool:
        async with self._connection_pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT 1 FROM approval_objections WHERE thread_id = %s LIMIT 1",
                (thread_id,),
            )
            return await cursor.fetchone() is not None


class ExpertApprovalService:
    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        context_resolver: ApprovalContextResolver,
        approval_repository: ApprovalRepository,
    ) -> None:
        self._artifact_store = artifact_store
        self._context_resolver = context_resolver
        self._approval_repository = approval_repository

    async def approve(self, state: RunState, decision: Mapping[str, Any]) -> tuple[str, str]:
        if state["node_status"].get("reality_check") is not NodeStatus.SUPPORTED:
            raise ValueError("Expert approval requires code-evaluated hard-rule PASS.")
        if state["coverage"].get("reality_check_decision") != "PASS":
            raise ValueError("Expert approval requires code-evaluated hard-rule PASS.")
        bundle, evidence_manifest_ref, approved_package_ref, approved_package_digest = await self._context_resolver.resolve(state)
        if bundle.run_id != state["run_id"]:
            raise ValueError("Approval version bundle belongs to another run.")
        expected_thread_id = thread_id_for(bundle)
        action = decision.get("action")
        if action == "reject":
            payload = {
                "artifact_kind": "expert_objection", "run_id": bundle.run_id,
                "thread_id": expected_thread_id, "version_bundle": bundle.model_dump(mode="json"),
                "approved_package_ref": approved_package_ref,
                "approved_package_digest": approved_package_digest,
                "expert_id": decision.get("expert_id"), "reason": decision.get("reason"),
            }
            objection_ref = await self._artifact_store.put_json(payload)
            await self._approval_repository.store_objection(objection_ref, bundle)
            return objection_ref, expected_thread_id
        if action != "approve":
            raise ValueError("Expert decision must be approve or reject.")
        if await self._approval_repository.has_objection(expected_thread_id):
            raise ValueError("An expert objection already exists for this exact version thread.")
        expert_id = decision.get("expert_id")
        if not isinstance(expert_id, str) or not expert_id.strip():
            raise ValueError("Authenticated expert identity is required.")
        payload = {
            "artifact_kind": "expert_approval", "run_id": bundle.run_id,
            "thread_id": expected_thread_id, "version_bundle": bundle.model_dump(mode="json"),
            "evidence_manifest_ref": evidence_manifest_ref, "expert_id": expert_id,
            "approved_package_ref": approved_package_ref,
            "approved_package_digest": approved_package_digest,
            "decision": "approved",
        }
        approval_ref = await self._artifact_store.put_json(payload)
        await self._approval_repository.store_approval(approval_ref, bundle)
        return approval_ref, expected_thread_id


def make_expert_approval_node(service: ExpertApprovalService):
    from langgraph.types import interrupt

    async def expert_approval_node(state: RunState) -> dict[str, Any]:
        bundle, _, _, _ = await service._context_resolver.resolve(state)
        thread_id = thread_id_for(bundle)
        decision = interrupt({
            "kind": "expert_approval", "thread_id": thread_id,
            "approval_version": state["approval_version"],
            "version_bundle": bundle.model_dump(mode="json"),
        })
        if not isinstance(decision, Mapping):
            raise TypeError("Approval resume payload must be a mapping.")
        approval_ref, approval_thread_id = await service.approve(state, decision)
        return {
            "approval_version": approval_thread_id,
            "result_ids": [approval_ref],
            "node_status": {"expert_review": NodeStatus.SUPPORTED if decision.get("action") == "approve" else NodeStatus.FAIL},
        }

    return expert_approval_node
