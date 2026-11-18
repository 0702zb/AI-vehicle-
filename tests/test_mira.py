from dataclasses import dataclass

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from concept_platform.agents import (
    AgentDefinition,
    AgentExecutor,
    AgentRegistry,
    AgentTaskContext,
    MockProvider,
    ProviderRegistry,
    ProviderResponse,
)
from concept_platform.artifacts import LocalArtifactStore
from concept_platform.graph import build_graph
from concept_platform.mira import (
    BriefConfirmationService,
    BriefDraftArtifact,
    BriefFieldDefinition,
    BriefQuestion,
    BriefSubmission,
    BriefValueType,
    MiraInput,
    MiraIntakeService,
    MiraProposal,
    make_brief_confirmation_node,
)
from concept_platform.state import NodeStatus, RunState


def task_context() -> AgentTaskContext:
    return AgentTaskContext(
        project_id="project-a",
        concept_id="concept-seed",
        concept_version="concept-v0",
        context_manifest_hash="sha256:initial-context",
        allowed_evidence={},
        allowed_models={},
        context_scope=frozenset({"brief"}),
    )


def mira_executor(provider: MockProvider) -> AgentExecutor:
    agents = AgentRegistry()
    agents.register(
        AgentDefinition(
            agent_id="mira",
            version="1.0",
            input_schema=MiraInput,
            output_schema=MiraProposal,
            prompt_template_version="mira-brief-v1",
            system_instruction="Extract source-grounded candidates and clarification questions.",
            provider_id="mock",
            model_id="mira-mock",
            model_version="mock-v1",
            context_scope=frozenset({"brief"}),
        )
    )
    providers = ProviderRegistry()
    providers.register("mock", "mira-mock", "mock-v1", provider)
    return AgentExecutor(agents, providers)


def mira_output(**overrides) -> dict:
    values = {
        "project_id": "project-a",
        "concept_id": "concept-seed",
        "concept_version": "concept-v0",
        "context_manifest_hash": "sha256:initial-context",
        "status": "supported",
        "summary": "The submitted brief identifies a target market.",
        "evidence_refs": [],
        "model_refs": [],
        "assumptions": [],
        "risks": [],
        "questions": [],
        "field_candidates": [
            {
                "field_id": "market",
                "value_text": "China",
                "source_quote": "for China market",
                "status": "supported",
                "missing_reason": None,
            }
        ],
        "clarification_questions": [],
    }
    values.update(overrides)
    return values


def submission(source_text: str = "We need an urban EV for China market.") -> BriefSubmission:
    return BriefSubmission(
        source_text=source_text,
        field_definitions=(
            BriefFieldDefinition(
                field_id="market",
                label="Target market",
                value_type=BriefValueType.TEXT,
                required=True,
                definition_version="market-field-v1",
            ),
        ),
        task_context=task_context(),
    )


class FakeSubmissionRepository:
    def __init__(self, value: BriefSubmission) -> None:
        self.value = value

    async def load_submission(self, run_id: str) -> BriefSubmission:
        return self.value


@pytest.mark.asyncio
async def test_mira_persists_draft_by_reference_and_checks_source_quotes(tmp_path) -> None:
    provider = MockProvider(
        [ProviderResponse(structured_output=mira_output(), model_version="mock-v1")]
    )
    store = LocalArtifactStore(tmp_path)
    service = MiraIntakeService(
        executor=mira_executor(provider),
        submission_repository=FakeSubmissionRepository(submission()),
        artifact_store=store,
        agent_version="1.0",
    )

    result = await service.create_draft(
        run_id="run-1", expected_manifest_hash="sha256:initial-context"
    )
    draft = BriefDraftArtifact.model_validate(await store.get_json(result.draft_ref))

    assert result.status is NodeStatus.SUPPORTED
    assert draft.field_candidates[0].value_text == "China"
    assert result.draft_ref.startswith("sha256:")


@pytest.mark.asyncio
async def test_mira_rejects_fabricated_or_unregistered_field_quotes(tmp_path) -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=mira_output(
                    field_candidates=[
                        {
                            "field_id": "market",
                            "value_text": "Atlantis",
                            "source_quote": "for Atlantis market",
                            "status": "supported",
                        }
                    ]
                ),
                model_version="mock-v1",
            )
        ]
    )
    store = LocalArtifactStore(tmp_path)
    service = MiraIntakeService(
        executor=mira_executor(provider),
        submission_repository=FakeSubmissionRepository(submission()),
        artifact_store=store,
        agent_version="1.0",
    )

    with pytest.raises(ValueError, match="does not occur verbatim"):
        await service.create_draft(
            run_id="run-1", expected_manifest_hash="sha256:initial-context"
        )


@dataclass
class FakeActorResolver:
    async def current_actor_id(self, run_id: str) -> str:
        return "engineer-7"


def initial_state() -> RunState:
    return {
        "run_id": "run-freeze",
        "brief_version": "brief-draft-v0",
        "context_manifest_hash": "sha256:initial-context",
        "node_status": {},
        "coverage": {},
        "result_ids": [],
        "approval_version": "approval-old",
        "pending_writes": [],
    }


@pytest.mark.asyncio
async def test_brief_confirmation_interrupts_then_freezes_only_after_user_resume(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    draft = BriefDraftArtifact(
        run_id="run-freeze",
        project_id="project-a",
        concept_id="concept-seed",
        context_manifest_hash="sha256:initial-context",
        source_text="",
        field_definitions=(
            BriefFieldDefinition(
                field_id="market",
                label="Target market",
                value_type=BriefValueType.TEXT,
                required=True,
                definition_version="market-field-v1",
            ),
        ),
        field_candidates=(),
        clarification_questions=(
            BriefQuestion(question_id="market-question", field_id="market", text="Which market?"),
        ),
    )
    draft_ref = await store.put_json(draft.model_dump(mode="json"))
    confirm_service = BriefConfirmationService(store)

    async def mira_node(state: RunState):
        return {"node_status": {"mira": NodeStatus.UNKNOWN}, "result_ids": [draft_ref]}

    graph = build_graph(
        checkpointer=InMemorySaver(),
        node_overrides={
            "mira": mira_node,
            "brief_confirmation": make_brief_confirmation_node(
                confirm_service, FakeActorResolver()
            ),
        },
    )
    config = {"configurable": {"thread_id": "mira-brief-interrupt-test"}}

    first = await graph.ainvoke(initial_state(), config)
    assert first["__interrupt__"][0].value["kind"] == "brief_clarification"

    second = await graph.ainvoke(
        Command(resume={"answers": {"market-question": "China"}}), config
    )
    assert second["__interrupt__"][0].value["kind"] == "brief_confirmation"

    final = await graph.ainvoke(Command(resume={"action": "confirm"}), config)

    assert final["brief_version"].startswith("brief:")
    assert final["context_manifest_hash"] != "sha256:initial-context"
    assert final["approval_version"] is None
    assert final["node_status"]["brief_confirmation"] is NodeStatus.SUPPORTED
    assert len(final["result_ids"]) == 2
    frozen = await store.get_json(final["result_ids"][-1])
    assert frozen["confirmed_values"] == {"market": "China"}
    assert frozen["confirmed_by"] == "engineer-7"
    assert "source_text" not in final
