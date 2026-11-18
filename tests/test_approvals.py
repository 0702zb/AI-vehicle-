from pathlib import Path

import pytest

from concept_platform.approvals import (
    EvaluationVersionBundle,
    ExpertApprovalService,
    thread_id_for,
)
from concept_platform.artifacts import LocalArtifactStore
from concept_platform.graph import route_after_expert_review
from concept_platform.state import NodeStatus, RunState


class Resolver:
    def __init__(self, bundle):
        self.bundle = bundle

    async def resolve(self, state):
        return self.bundle, "sha256:evidence-manifest", "sha256:approved-package", "digest-package"


class Repository:
    def __init__(self):
        self.approvals = {}
        self.objections = []

    async def store_approval(self, ref, bundle):
        self.approvals[thread_id_for(bundle)] = ref

    async def store_objection(self, ref, bundle):
        self.objections.append((ref, bundle))

    async def load_approval(self, thread_id):
        return self.approvals.get(thread_id)

    async def has_objection(self, thread_id):
        return any(thread_id_for(bundle) == thread_id for _, bundle in self.objections)


def bundle(**changes):
    values = {
        "run_id": "run-a",
        "concept_version": "concept-v4",
        "evaluation_version": "evaluation-v3",
        "evidence_manifest_hash": "sha256:evidence-v2",
        "model_rule_versions": {"solver": "solver-v3", "rules": "rules-v2"},
    }
    values.update(changes)
    return EvaluationVersionBundle(**values)


def state(**changes) -> RunState:
    value = {
        "run_id": "run-a", "brief_version": "brief-v2",
        "context_manifest_hash": "sha256:manifest", "node_status": {"reality_check": NodeStatus.SUPPORTED},
        "coverage": {"reality_check_decision": "PASS"}, "result_ids": [],
        "approval_version": None, "pending_writes": [],
    }
    value.update(changes)
    return value


@pytest.mark.asyncio
async def test_expert_approval_is_immutable_and_bound_to_exact_versions(tmp_path: Path):
    artifact_store = LocalArtifactStore(tmp_path)
    repository = Repository()
    service = ExpertApprovalService(
        artifact_store=artifact_store,
        context_resolver=Resolver(bundle()),
        approval_repository=repository,
    )
    approval_ref, thread_id = await service.approve(
        state(), {"action": "approve", "expert_id": "expert-17"}
    )
    artifact = await artifact_store.get_json(approval_ref)
    assert artifact["thread_id"] == thread_id == thread_id_for(bundle())
    assert artifact["approved_package_ref"] == "sha256:approved-package"
    assert await repository.load_approval(thread_id) == approval_ref
    assert route_after_expert_review(state(node_status={"expert_review": NodeStatus.SUPPORTED}, approval_version=thread_id)) == "concept_paper"


@pytest.mark.asyncio
async def test_human_rejection_is_recorded_and_never_advances(tmp_path: Path):
    service = ExpertApprovalService(
        artifact_store=LocalArtifactStore(tmp_path), context_resolver=Resolver(bundle()),
        approval_repository=Repository(),
    )
    objection_ref, thread_id = await service.approve(
        state(), {"action": "reject", "expert_id": "expert-17", "reason": "needs review"}
    )
    assert objection_ref.startswith("sha256:")
    assert thread_id == thread_id_for(bundle())
    assert route_after_expert_review(state(node_status={"expert_review": NodeStatus.FAIL}, approval_version=thread_id)) == "stop"


@pytest.mark.asyncio
async def test_approval_cannot_override_unknown_or_failed_reality_check(tmp_path: Path):
    service = ExpertApprovalService(
        artifact_store=LocalArtifactStore(tmp_path), context_resolver=Resolver(bundle()),
        approval_repository=Repository(),
    )
    for status, decision in ((NodeStatus.UNKNOWN, "UNKNOWN"), (NodeStatus.FAIL, "FAIL")):
        with pytest.raises(ValueError, match="code-evaluated"):
            await service.approve(
                state(node_status={"reality_check": status}, coverage={"reality_check_decision": decision}),
                {"action": "approve", "expert_id": "expert-17"},
            )
