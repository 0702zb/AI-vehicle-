"""Approval-bound Concept Paper generation and deterministic issuance checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from concept_platform.agents import (
    AgentExecutor,
    AgentInput,
    AgentProposal,
    AgentTaskContext,
    EvidenceReference,
)
from concept_platform.approvals import (
    ApprovalRepository,
    EvaluationVersionBundle,
    thread_id_for,
)
from concept_platform.artifacts import ArtifactStore
from concept_platform.state import NodeStatus, RunState


class PaperClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    evidence_refs: tuple[EvidenceReference, ...] = ()


class ConceptPaperInput(AgentInput):
    approved_package_ref: str
    approved_package_digest: str
    evaluation_version: str
    evidence_manifest_hash: str


class ConceptPaperProposal(AgentProposal):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    executive_summary_evidence_refs: tuple[EvidenceReference, ...] = ()
    claims: tuple[PaperClaim, ...]


class ApprovedPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: str
    run_id: str
    project_id: str
    concept_id: str
    concept_version: str
    evaluation_version: str
    evidence_manifest_hash: str
    model_rule_versions: Mapping[str, str]
    content: Mapping[str, Any]
    evidence_refs: tuple[str, ...]


class PaperContextResolver(Protocol):
    async def resolve(self, state: RunState) -> tuple[EvaluationVersionBundle, AgentTaskContext]: ...


class ApprovedPackageResolver(Protocol):
    async def load_approved_package(
        self, state: RunState, bundle: EvaluationVersionBundle
    ) -> tuple[str, ApprovedPackage]: ...


@dataclass(frozen=True)
class PaperOutcome:
    status: NodeStatus
    artifact_ref: str | None
    reason: str | None = None


class ConceptPaperService:
    def __init__(
        self,
        *,
        executor: AgentExecutor,
        context_resolver: PaperContextResolver,
        package_resolver: ApprovedPackageResolver,
        approval_repository: ApprovalRepository,
        artifact_store: ArtifactStore,
        agent_id: str = "concept_paper",
        agent_version: str,
    ) -> None:
        self._executor = executor
        self._context_resolver = context_resolver
        self._package_resolver = package_resolver
        self._approval_repository = approval_repository
        self._artifact_store = artifact_store
        self._agent_id = agent_id
        self._agent_version = agent_version

    async def create(self, state: RunState) -> PaperOutcome:
        bundle, context = await self._context_resolver.resolve(state)
        thread_id = thread_id_for(bundle)
        if state["approval_version"] != thread_id:
            raise ValueError("Approval version does not match current concept/evaluation inputs.")
        approval_ref = await self._approval_repository.load_approval(thread_id)
        if not approval_ref or await self._approval_repository.has_objection(thread_id):
            raise ValueError("No expert approval exists for the current version thread.")
        stored_approval = await self._artifact_store.get_json(approval_ref)
        if stored_approval.get("artifact_kind") != "expert_approval" or stored_approval.get("thread_id") != thread_id:
            raise ValueError("Approval artifact is invalid or belongs to another version thread.")

        package_ref, package = await self._package_resolver.load_approved_package(state, bundle)
        package_digest = hashlib.sha256(
            json.dumps(package.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            package.run_id != bundle.run_id
            or package.concept_version != bundle.concept_version
            or package.evaluation_version != bundle.evaluation_version
            or package.evidence_manifest_hash != bundle.evidence_manifest_hash
            or dict(package.model_rule_versions) != dict(bundle.model_rule_versions)
        ):
            raise ValueError("Approved package summary does not match the approval version bundle.")
        if (
            stored_approval.get("approved_package_ref") != package_ref
            or stored_approval.get("approved_package_digest") != package_digest
        ):
            raise ValueError("Approved package differs from the package reviewed by the expert.")

        result = await self._executor.execute(
            agent_id=self._agent_id,
            agent_version=self._agent_version,
            task_input={
                "approved_package_ref": package_ref,
                "approved_package_digest": package_digest,
                "evaluation_version": bundle.evaluation_version,
                "evidence_manifest_hash": bundle.evidence_manifest_hash,
            },
            context=context,
        )
        if result.status is NodeStatus.BLOCKED or result.proposal is None:
            return PaperOutcome(NodeStatus.BLOCKED, None, result.reason)
        if not isinstance(result.proposal, ConceptPaperProposal):
            raise TypeError("Concept Paper AgentDefinition has an unexpected output schema.")

        formal_claims = []
        review_items = []
        if result.proposal.executive_summary_evidence_refs:
            self._validate_citations(result.proposal.executive_summary_evidence_refs, context)
            formal_summary = result.proposal.executive_summary
            summary_refs = [ref.model_dump(mode="json") for ref in result.proposal.executive_summary_evidence_refs]
        else:
            formal_summary = ""
            summary_refs = []
            review_items.append(result.proposal.executive_summary)
        for claim in result.proposal.claims:
            if not claim.evidence_refs:
                review_items.append(claim.text)
            else:
                self._validate_citations(claim.evidence_refs, context)
                formal_claims.append(claim.model_dump(mode="json"))
        artifact = {
            "artifact_kind": "concept_paper",
            "run_id": bundle.run_id,
            "thread_id": thread_id,
            "approval_ref": approval_ref,
            "approved_package_ref": package_ref,
            "approved_package_digest": package_digest,
            "concept_version": bundle.concept_version,
            "evaluation_version": bundle.evaluation_version,
            "evidence_manifest_hash": bundle.evidence_manifest_hash,
            "model_rule_versions": dict(bundle.model_rule_versions),
            "title": result.proposal.title,
            "executive_summary": formal_summary,
            "executive_summary_evidence_refs": summary_refs,
            "formal_claims": formal_claims,
            "review_items": review_items,
            "agent_version": result.agent_version,
            "prompt_template_version": result.prompt_template_version,
            "provider_id": result.provider_id,
            "model_id": result.model_id,
            "model_version": result.model_version,
        }
        return PaperOutcome(result.status, await self._artifact_store.put_json(artifact))

    @staticmethod
    def _validate_citations(references: Sequence[EvidenceReference], context: AgentTaskContext) -> None:
        for ref in references:
            if (
                ref.project_id != context.project_id
                or ref.context_manifest_hash != context.context_manifest_hash
                or context.allowed_evidence.get(ref.evidence_id) != ref.version
            ):
                raise ValueError("Concept Paper citation is outside the authorized evidence snapshot.")


class IssuanceContextResolver(Protocol):
    async def resolve(self, state: RunState) -> tuple[EvaluationVersionBundle, str, str]: ...


class ReportIssuanceService:
    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        context_resolver: IssuanceContextResolver,
        approval_repository: ApprovalRepository,
    ) -> None:
        self._artifact_store = artifact_store
        self._context_resolver = context_resolver
        self._approval_repository = approval_repository

    async def issue(self, state: RunState) -> str:
        if not state["result_ids"]:
            raise ValueError("Concept Paper artifact reference is missing.")
        paper_ref = state["result_ids"][-1]
        paper = await self._artifact_store.get_json(paper_ref)
        bundle, expected_package_ref, expected_package_digest = await self._context_resolver.resolve(state)
        thread_id = thread_id_for(bundle)
        approval_ref = await self._approval_repository.load_approval(thread_id)
        expected_digest = paper.get("approved_package_digest")
        package_identity = paper.get("approved_package_ref")
        if (
            not approval_ref
            or await self._approval_repository.has_objection(thread_id)
            or state["approval_version"] != thread_id
            or paper.get("thread_id") != thread_id
            or paper.get("approval_ref") != approval_ref
            or package_identity != expected_package_ref
            or expected_digest != expected_package_digest
            or paper.get("concept_version") != bundle.concept_version
            or paper.get("evaluation_version") != bundle.evaluation_version
            or paper.get("evidence_manifest_hash") != bundle.evidence_manifest_hash
            or paper.get("model_rule_versions") != dict(bundle.model_rule_versions)
            or not expected_digest
        ):
            raise ValueError("Pre-issuance approval/package/version recheck failed.")
        # Verify the paper's package digest against the immutable approved package snapshot.
        if paper.get("artifact_kind") != "concept_paper":
            raise ValueError("Issuance requires a Concept Paper artifact.")
        report = {
            "artifact_kind": "issued_report",
            "run_id": bundle.run_id,
            "thread_id": thread_id,
            "approval_ref": approval_ref,
            "concept_paper_ref": paper_ref,
            "approved_package_ref": package_identity,
            "approved_package_digest": expected_digest,
            "version_bundle": bundle.model_dump(mode="json"),
            "title": paper["title"],
            "executive_summary": paper["executive_summary"],
            "executive_summary_evidence_refs": paper["executive_summary_evidence_refs"],
            "formal_claims": paper["formal_claims"],
            "review_items": paper["review_items"],
            "issued": True,
        }
        return await self._artifact_store.put_json(report)


def make_concept_paper_node(service: ConceptPaperService):
    async def concept_paper_node(state: RunState) -> dict[str, Any]:
        outcome = await service.create(state)
        patch: dict[str, Any] = {"node_status": {"concept_paper": outcome.status}}
        if outcome.artifact_ref:
            patch["result_ids"] = [outcome.artifact_ref]
        return patch

    return concept_paper_node


def make_report_issuance_node(service: ReportIssuanceService):
    async def report_issuance_node(state: RunState) -> dict[str, Any]:
        report_ref = await service.issue(state)
        return {"node_status": {"report_issuance": NodeStatus.SUPPORTED}, "result_ids": [report_ref]}

    return report_issuance_node
