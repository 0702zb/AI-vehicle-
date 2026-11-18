"""M.I.R.A. intake, interrupt-driven confirmation, and immutable Brief freeze."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from concept_platform.agents import (
    AgentExecutionResult,
    AgentExecutor,
    AgentInput,
    AgentProposal,
    AgentTaskContext,
    ProposalStatus,
)
from concept_platform.artifacts import ArtifactStore
from concept_platform.state import NodeStatus, RunState


class BriefValueType(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"


class BriefFieldDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value_type: BriefValueType
    required: bool = False
    unit: str | None = None
    allowed_values: tuple[str, ...] = ()
    definition_version: str = Field(min_length=1)


class BriefQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(min_length=1)
    field_id: str | None = None
    text: str = Field(min_length=1)


class MiraFieldCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str = Field(min_length=1)
    value_text: str | None = None
    source_quote: str | None = None
    status: ProposalStatus
    missing_reason: str | None = None


class MiraInput(AgentInput):
    source_text: str
    field_definitions: list[BriefFieldDefinition]


class MiraProposal(AgentProposal):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_candidates: list[MiraFieldCandidate] = Field(default_factory=list)
    clarification_questions: list[BriefQuestion] = Field(default_factory=list)


class BriefSubmission(BaseModel):
    """Current user input plus trusted server-resolved task and field scope."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    source_text: str
    field_definitions: tuple[BriefFieldDefinition, ...]
    task_context: AgentTaskContext


class BriefDraftArtifact(BaseModel):
    """Full draft lives in object storage, never in LangGraph state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: str = "brief_draft"
    run_id: str
    project_id: str
    concept_id: str
    context_manifest_hash: str
    source_text: str
    field_definitions: tuple[BriefFieldDefinition, ...]
    field_candidates: tuple[MiraFieldCandidate, ...]
    clarification_questions: tuple[BriefQuestion, ...]
    clarification_answers: dict[str, str] = Field(default_factory=dict)
    confirmed_values: dict[str, str] = Field(default_factory=dict)


class SubmissionRepository(Protocol):
    async def load_submission(self, run_id: str) -> BriefSubmission: ...


@dataclass(frozen=True)
class MiraIntakeResult:
    status: NodeStatus
    draft_ref: str | None
    reason: str | None = None


class MiraIntakeService:
    def __init__(
        self,
        *,
        executor: AgentExecutor,
        submission_repository: SubmissionRepository,
        artifact_store: ArtifactStore,
        agent_id: str = "mira",
        agent_version: str,
    ) -> None:
        self._executor = executor
        self._submission_repository = submission_repository
        self._artifact_store = artifact_store
        self._agent_id = agent_id
        self._agent_version = agent_version

    async def create_draft(
        self, *, run_id: str, expected_manifest_hash: str
    ) -> MiraIntakeResult:
        submission = await self._submission_repository.load_submission(run_id)
        if submission.task_context.context_manifest_hash != expected_manifest_hash:
            raise ValueError("M.I.R.A. submission manifest does not match the active run.")

        response: AgentExecutionResult = await self._executor.execute(
            agent_id=self._agent_id,
            agent_version=self._agent_version,
            task_input={
                "source_text": submission.source_text,
                "field_definitions": [item.model_dump(mode="json") for item in submission.field_definitions],
            },
            context=submission.task_context,
        )
        if response.status is NodeStatus.BLOCKED or response.proposal is None:
            return MiraIntakeResult(NodeStatus.BLOCKED, None, response.reason)
        if not isinstance(response.proposal, MiraProposal):
            raise TypeError("M.I.R.A. AgentDefinition must use MiraProposal as its output schema.")

        self._validate_proposal(response.proposal, submission)
        draft = BriefDraftArtifact(
            run_id=run_id,
            project_id=submission.task_context.project_id,
            concept_id=submission.task_context.concept_id,
            context_manifest_hash=expected_manifest_hash,
            source_text=submission.source_text,
            field_definitions=submission.field_definitions,
            field_candidates=tuple(response.proposal.field_candidates),
            clarification_questions=tuple(response.proposal.clarification_questions),
        )
        artifact_id = await self._artifact_store.put_json(draft.model_dump(mode="json"))
        return MiraIntakeResult(response.status, artifact_id)

    @staticmethod
    def _validate_proposal(proposal: MiraProposal, submission: BriefSubmission) -> None:
        field_definitions = {field.field_id: field for field in submission.field_definitions}
        if len(field_definitions) != len(submission.field_definitions):
            raise ValueError("Brief field registry contains duplicate field IDs.")
        seen_fields: set[str] = set()
        for candidate in proposal.field_candidates:
            if candidate.field_id not in field_definitions:
                raise ValueError(f"M.I.R.A. proposed unregistered Brief field: {candidate.field_id}")
            if candidate.field_id in seen_fields:
                raise ValueError(f"M.I.R.A. returned duplicate Brief field: {candidate.field_id}")
            seen_fields.add(candidate.field_id)
            if candidate.source_quote is not None and candidate.source_quote not in submission.source_text:
                raise ValueError("M.I.R.A. source quote does not occur verbatim in submitted text.")
            if candidate.value_text is not None:
                if not candidate.source_quote or candidate.value_text not in candidate.source_quote:
                    raise ValueError("M.I.R.A. extracted value must be a verbatim source substring.")
            elif candidate.status in (ProposalStatus.SUPPORTED, ProposalStatus.ASSUMPTION):
                raise ValueError("A supported/assumption candidate must cite an extracted source value.")

        for question in proposal.clarification_questions:
            if question.field_id is not None and question.field_id not in field_definitions:
                raise ValueError(f"M.I.R.A. asked about unregistered Brief field: {question.field_id}")


@dataclass(frozen=True)
class FrozenBrief:
    artifact_id: str
    brief_version: str
    context_manifest_hash: str


class BriefConfirmationService:
    """Applies user responses and freezes only explicitly confirmed Brief values."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    async def load_draft(self, artifact_id: str) -> BriefDraftArtifact:
        payload = await self._artifact_store.get_json(artifact_id)
        return BriefDraftArtifact.model_validate(payload)

    async def user_questions(self, draft: BriefDraftArtifact) -> tuple[BriefQuestion, ...]:
        questions = list(draft.clarification_questions)
        definitions = {field.field_id: field for field in draft.field_definitions}
        candidate_values = {
            candidate.field_id
            for candidate in draft.field_candidates
            if candidate.value_text is not None
            and self._valid_value(definitions[candidate.field_id], candidate.value_text)
        }
        resolved_fields = candidate_values | {
            field_id
            for field_id, value in draft.confirmed_values.items()
            if field_id in definitions and self._valid_value(definitions[field_id], value)
        }
        existing_ids = {question.question_id for question in questions}
        existing_fields = {question.field_id for question in questions if question.field_id}
        for field in draft.field_definitions:
            if (
                field.required
                and field.field_id not in resolved_fields
                and field.field_id not in existing_fields
            ):
                question_id = f"required:{field.field_id}"
                if question_id not in existing_ids:
                    questions.append(
                        BriefQuestion(
                            question_id=question_id,
                            field_id=field.field_id,
                            text=f"Please provide the required value for {field.label}.",
                        )
                    )
        return tuple(questions)

    async def apply_user_answers(
        self,
        draft_id: str,
        answers: Mapping[str, str],
        expected_questions: Sequence[BriefQuestion],
    ) -> str:
        draft = await self.load_draft(draft_id)
        fields = {field.field_id: field for field in draft.field_definitions}
        questions = {question.question_id: question for question in expected_questions}
        accepted_keys = set(fields) | set(questions)
        unknown = set(answers) - accepted_keys
        if unknown:
            raise ValueError("User response contains unregistered Brief fields/questions: " + ", ".join(sorted(unknown)))
        if set(answers) & set(questions) != set(questions):
            raise ValueError("Every outstanding clarification question requires a user response.")
        if any(not isinstance(value, str) or not value.strip() for value in answers.values()):
            raise ValueError("Brief answers must be non-empty user-provided strings.")

        confirmed = dict(draft.confirmed_values)
        clarification_answers = dict(draft.clarification_answers)
        for key, value in answers.items():
            if key in fields:
                if not self._valid_value(fields[key], value):
                    raise ValueError(f"User value does not match the registered type/unit for {key}.")
                confirmed[key] = value.strip()
            elif questions[key].field_id:
                field_id = questions[key].field_id
                if not self._valid_value(fields[field_id], value):
                    raise ValueError(
                        f"User value does not match the registered type/unit for {field_id}."
                    )
                confirmed[field_id] = value.strip()
            else:
                clarification_answers[key] = value.strip()

        updated = draft.model_copy(
            update={
                "clarification_questions": tuple(
                    question for question in draft.clarification_questions if question.question_id not in answers
                ),
                "clarification_answers": clarification_answers,
                "confirmed_values": confirmed,
            }
        )
        return await self._artifact_store.put_json(updated.model_dump(mode="json"))

    async def freeze(
        self,
        *,
        draft_id: str,
        parent_manifest_hash: str,
        actor_id: str,
        user_edits: Mapping[str, str] | None = None,
    ) -> FrozenBrief:
        if not actor_id:
            raise ValueError("Authenticated actor identity is required to freeze a Brief.")
        draft = await self.load_draft(draft_id)
        outstanding = await self.user_questions(draft)
        if outstanding:
            raise ValueError("Brief cannot freeze while required fields or clarification questions remain.")

        definitions = {field.field_id: field for field in draft.field_definitions}
        edits = dict(user_edits or {})
        if set(edits) - set(definitions):
            raise ValueError("Brief edits contain unregistered field IDs.")

        confirmed = dict(draft.confirmed_values)
        for candidate in draft.field_candidates:
            if candidate.value_text is not None:
                definition = definitions[candidate.field_id]
                if self._valid_value(definition, candidate.value_text):
                    confirmed.setdefault(candidate.field_id, candidate.value_text)
        for key, value in edits.items():
            if not isinstance(value, str) or not self._valid_value(definitions[key], value):
                raise ValueError(f"Confirmed Brief value does not match the type/unit for {key}.")
            confirmed[key] = value.strip()

        for key, value in confirmed.items():
            if key not in definitions or not self._valid_value(definitions[key], value):
                raise ValueError(f"Confirmed Brief value does not match the type/unit for {key}.")

        missing = [
            field.field_id
            for field in draft.field_definitions
            if field.required and not confirmed.get(field.field_id)
        ]
        if missing:
            raise ValueError("Brief cannot freeze with missing required fields: " + ", ".join(missing))

        frozen_content = {
            "artifact_kind": "frozen_brief",
            "run_id": draft.run_id,
            "project_id": draft.project_id,
            "concept_id": draft.concept_id,
            "confirmed_values": confirmed,
            "field_definitions": [field.model_dump(mode="json") for field in draft.field_definitions],
            "source_draft_ref": draft_id,
            "confirmed_by": actor_id,
        }
        artifact_id = await self._artifact_store.put_json(frozen_content)
        digest = artifact_id.partition(":")[2]
        manifest_payload = {
            "parent_context_manifest_hash": parent_manifest_hash,
            "brief_artifact_id": artifact_id,
            "field_definition_versions": sorted(
                (field.field_id, field.definition_version) for field in draft.field_definitions
            ),
        }
        manifest_hash = "sha256:" + hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return FrozenBrief(
            artifact_id=artifact_id,
            brief_version="brief:" + digest,
            context_manifest_hash=manifest_hash,
        )

    @staticmethod
    def _valid_value(field: BriefFieldDefinition, value: str) -> bool:
        candidate = value.strip()
        if not candidate:
            return False
        if field.value_type is BriefValueType.TEXT:
            return True
        if field.value_type is BriefValueType.ENUM:
            return candidate in field.allowed_values
        if field.value_type is BriefValueType.BOOLEAN:
            allowed = field.allowed_values or ("true", "false")
            return candidate.casefold() in {item.casefold() for item in allowed}
        if field.value_type is BriefValueType.NUMBER:
            unit = re.escape(field.unit) if field.unit else None
            suffix = rf"\s*{unit}" if unit else ""
            if not re.fullmatch(rf"[+-]?(?:\d+(?:\.\d*)?|\.\d+){suffix}", candidate):
                return False
            numeric = re.match(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", candidate)
            try:
                return bool(numeric and Decimal(numeric.group()).is_finite())
            except InvalidOperation:
                return False
        return False


class ActorResolver(Protocol):
    async def current_actor_id(self, run_id: str) -> str: ...


def make_mira_node(service: MiraIntakeService):
    async def mira_node(state: RunState) -> dict[str, Any]:
        outcome = await service.create_draft(
            run_id=state["run_id"], expected_manifest_hash=state["context_manifest_hash"]
        )
        patch: dict[str, Any] = {"node_status": {"mira": outcome.status}}
        if outcome.draft_ref:
            patch["result_ids"] = [outcome.draft_ref]
        return patch

    return mira_node


def make_brief_confirmation_node(
    service: BriefConfirmationService, actor_resolver: ActorResolver
):
    from langgraph.types import interrupt

    async def brief_confirmation_node(state: RunState) -> dict[str, Any]:
        if not state["result_ids"]:
            raise ValueError("M.I.R.A. draft reference is missing from RunState.result_ids.")
        draft_id = state["result_ids"][-1]

        while True:
            draft = await service.load_draft(draft_id)
            questions = await service.user_questions(draft)
            if questions:
                response = interrupt(
                    {
                        "kind": "brief_clarification",
                        "draft_ref": draft_id,
                        "questions": [question.model_dump(mode="json") for question in questions],
                    }
                )
                if not isinstance(response, Mapping) or not isinstance(response.get("answers"), Mapping):
                    raise ValueError("Clarification resume must contain an answers mapping.")
                draft_id = await service.apply_user_answers(
                    draft_id, response["answers"], questions
                )
                continue

            response = interrupt(
                {
                    "kind": "brief_confirmation",
                    "draft_ref": draft_id,
                    "candidate_fields": [
                        candidate.model_dump(mode="json") for candidate in draft.field_candidates
                    ],
                    "confirmed_values": draft.confirmed_values,
                    "instruction": "Confirm or edit this Brief before it is frozen.",
                }
            )
            if not isinstance(response, Mapping):
                raise TypeError("Brief confirmation resume must be a mapping.")
            action = response.get("action")
            if action == "edit":
                edits = response.get("values")
                if not isinstance(edits, Mapping):
                    raise ValueError("Brief edit response must contain a values mapping.")
                draft_id = await service.apply_user_answers(draft_id, edits, ())
                continue
            if action != "confirm":
                raise ValueError("Brief confirmation action must be 'confirm' or 'edit'.")

            actor_id = await actor_resolver.current_actor_id(state["run_id"])
            frozen = await service.freeze(
                draft_id=draft_id,
                parent_manifest_hash=state["context_manifest_hash"],
                actor_id=actor_id,
                user_edits=response.get("values") if isinstance(response.get("values"), Mapping) else None,
            )
            return {
                "brief_version": frozen.brief_version,
                "context_manifest_hash": frozen.context_manifest_hash,
                "approval_version": None,
                "result_ids": [frozen.artifact_id],
                "node_status": {"brief_confirmation": NodeStatus.SUPPORTED},
            }

    return brief_confirmation_node
