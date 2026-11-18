import hashlib
import json
from pathlib import Path

import pytest

from concept_platform.agents import AgentExecutionResult, AgentTaskContext
from concept_platform.approvals import EvaluationVersionBundle, thread_id_for
from concept_platform.artifacts import LocalArtifactStore
from concept_platform.documents import (
    ApprovedPackage,
    ConceptPaperProposal,
    ConceptPaperService,
    PaperClaim,
    ReportIssuanceService,
)
from concept_platform.state import NodeStatus, RunState


def bundle(**changes):
    data = {
        "run_id": "run-doc", "concept_version": "concept-v2", "evaluation_version": "eval-v4",
        "evidence_manifest_hash": "sha256:manifest-evidence", "model_rule_versions": {"model": "m2", "rules": "r3"},
    }
    data.update(changes)
    return EvaluationVersionBundle(**data)


class PaperResolver:
    async def resolve(self, state):
        return bundle(), AgentTaskContext(
            project_id="project-a", concept_id="concept-a", concept_version="concept-v2",
            context_manifest_hash="sha256:ctx", allowed_evidence={"ev-1": "e1"},
            allowed_models={},
        )


class PackageResolver:
    def __init__(self, package):
        self.package = package
        self.calls = 0

    async def load_approved_package(self, state, version_bundle):
        self.calls += 1
        return "sha256:package-ref", self.package


class IssuanceResolver:
    def __init__(self, digest):
        self.digest = digest

    async def resolve(self, state):
        return bundle(), "sha256:package-ref", self.digest


class ApprovalRepo:
    def __init__(self, ref):
        self.ref = ref

    async def load_approval(self, thread_id):
        return self.ref if thread_id == thread_id_for(bundle()) else None

    async def has_objection(self, thread_id):
        return False


class FakeExecutor:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    async def execute(self, **kwargs):
        self.calls += 1
        return AgentExecutionResult(
            status=NodeStatus.SUPPORTED, proposal=self.proposal, reason=None,
            agent_id="concept_paper", agent_version="v1", prompt_template_version="p1",
            provider_id="mock", model_id="mock", model_version="mock-v1",
        )


def package():
    return ApprovedPackage(
        artifact_kind="approved_package", run_id="run-doc", project_id="project-a",
        concept_id="concept-a", concept_version="concept-v2", evaluation_version="eval-v4",
        evidence_manifest_hash="sha256:manifest-evidence", model_rule_versions={"model": "m2", "rules": "r3"},
        content={"decision": "approved"}, evidence_refs=("sha256:evidence",),
    )


def valid_proposal() -> ConceptPaperProposal:
    return ConceptPaperProposal(
        project_id="project-a", concept_id="concept-a", concept_version="concept-v2",
        context_manifest_hash="sha256:ctx", status="supported", summary="Draft",
        title="Concept study", executive_summary="A claim awaiting evidence", claims=(),
    )


def state() -> RunState:
    return {
        "run_id": "run-doc", "brief_version": "brief-v2", "context_manifest_hash": "sha256:ctx",
        "node_status": {"expert_review": NodeStatus.SUPPORTED}, "coverage": {}, "result_ids": [],
        "approval_version": thread_id_for(bundle()), "pending_writes": [],
    }


@pytest.mark.asyncio
async def test_package_mismatch_blocks_model_call_before_generation(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    approval = {"artifact_kind": "expert_approval", "thread_id": thread_id_for(bundle()),
                "approved_package_ref": "some-other-package", "approved_package_digest": "wrong"}
    approval_ref = await store.put_json(approval)
    repo = ApprovalRepo(approval_ref)
    resolver = PackageResolver(package())
    executor = FakeExecutor(valid_proposal())
    service = ConceptPaperService(
        executor=executor, context_resolver=PaperResolver(), package_resolver=resolver,
        approval_repository=repo, artifact_store=store, agent_version="v1",
    )
    with pytest.raises(ValueError, match="differs from the package"):
        await service.create(state())
    assert executor.calls == 0


@pytest.mark.asyncio
async def test_uncited_summary_and_claims_become_review_items_and_issue_is_rechecked(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    approved_package = package()
    package_digest = hashlib.sha256(
        json.dumps(approved_package.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    approval_ref = await store.put_json({
        "artifact_kind": "expert_approval", "thread_id": thread_id_for(bundle()),
        "approved_package_ref": "sha256:package-ref", "approved_package_digest": package_digest,
    })
    proposal = valid_proposal().model_copy(update={
        "claims": (PaperClaim(text="Uncited quantitative claim"),),
    })
    executor = FakeExecutor(proposal)
    service = ConceptPaperService(
        executor=executor, context_resolver=PaperResolver(), package_resolver=PackageResolver(approved_package),
        approval_repository=ApprovalRepo(approval_ref), artifact_store=store, agent_version="v1",
    )
    outcome = await service.create(state())
    paper = await store.get_json(outcome.artifact_ref)
    assert paper["executive_summary"] == ""
    assert paper["formal_claims"] == []
    assert paper["review_items"] == ["A claim awaiting evidence", "Uncited quantitative claim"]

    issue_service = ReportIssuanceService(
        artifact_store=store, context_resolver=IssuanceResolver(package_digest), approval_repository=ApprovalRepo(approval_ref)
    )
    issued_ref = await issue_service.issue({**state(), "result_ids": [outcome.artifact_ref]})
    report = await store.get_json(issued_ref)
    assert report["issued"] is True
    assert report["review_items"] == paper["review_items"]

    stale_service = ReportIssuanceService(
        artifact_store=store, context_resolver=IssuanceResolver("stale-digest"), approval_repository=ApprovalRepo(approval_ref)
    )
    with pytest.raises(ValueError, match="recheck failed"):
        await stale_service.issue({**state(), "result_ids": [outcome.artifact_ref]})
