
import pytest
from pydantic import ConfigDict, ValidationError

from concept_platform.agents import (
    AgentBackendValidationError,
    AgentDefinition,
    AgentExecutor,
    AgentInput,
    AgentProposal,
    AgentRegistry,
    AgentTaskContext,
    MockProvider,
    ProviderRegistry,
    ProviderResponse,
    make_agent_node,
)
from concept_platform.artifacts import LocalArtifactStore
from concept_platform.config import Environment
from concept_platform.state import NodeStatus


class BriefInput(AgentInput):
    brief_text: str


class IntakeProposal(AgentProposal):
    model_config = ConfigDict(extra="forbid", frozen=True)


def task_context(**overrides) -> AgentTaskContext:
    values = {
        "project_id": "project-a",
        "concept_id": "concept-1",
        "concept_version": "concept-v1",
        "context_manifest_hash": "sha256:manifest-v1",
        "allowed_evidence": {"ev-1": "ev-v3"},
        "allowed_models": {"model-1": "model-v2"},
        "context_scope": frozenset({"brief", "evidence"}),
        "requested_tools": frozenset(),
    }
    values.update(overrides)
    return AgentTaskContext(**values)


def definition(**overrides) -> AgentDefinition:
    values = {
        "agent_id": "mira",
        "version": "1.2.0",
        "input_schema": BriefInput,
        "output_schema": IntakeProposal,
        "prompt_template_version": "mira-intake-v4",
        "system_instruction": "Extract candidate fields and clarification questions.",
        "provider_id": "mock",
        "model_id": "mock-structured",
        "model_version": "mock-v1",
        "tool_allowlist": frozenset(),
        "context_scope": frozenset({"brief", "evidence"}),
        "token_budget": 1024,
        "timeout_seconds": 5,
        "stop_conditions": ("schema_valid",),
        "max_schema_repairs": 1,
    }
    values.update(overrides)
    return AgentDefinition(**values)


def valid_proposal(**overrides) -> dict:
    values = {
        "project_id": "project-a",
        "concept_id": "concept-1",
        "concept_version": "concept-v1",
        "context_manifest_hash": "sha256:manifest-v1",
        "status": "unknown",
        "summary": "Needs user clarification.",
        "evidence_refs": [],
        "model_refs": [],
        "assumptions": [],
        "risks": [],
        "questions": ["Which market is in scope?"],
    }
    values.update(overrides)
    return values


def executor_for(provider: MockProvider | None = None, *, register_agent: bool = True):
    agents = AgentRegistry()
    if register_agent:
        agents.register(definition())
    providers = ProviderRegistry()
    if provider is not None:
        providers.register("mock", "mock-structured", "mock-v1", provider)
    return AgentExecutor(agents, providers)


def test_mock_provider_cannot_be_registered_in_production() -> None:
    providers = ProviderRegistry(Environment.PROD)
    with pytest.raises(ValueError, match="restricted to the development"):
        providers.register("mock", "mock-model", "mock-v1", MockProvider([]))


class AgentNodeResolver:
    async def resolve(self, state):
        return {"brief_text": "Short urban vehicle"}, task_context()


@pytest.mark.asyncio
async def test_shared_agent_node_stores_proposal_body_and_returns_only_reference(tmp_path):
    provider = MockProvider([ProviderResponse(valid_proposal(), "mock-v1")])
    node = make_agent_node(
        executor=executor_for(provider), input_resolver=AgentNodeResolver(),
        artifact_store=LocalArtifactStore(tmp_path), agent_id="mira", agent_version="1.2.0",
    )
    state = {
        "run_id": "run-agent", "brief_version": "brief-v1", "context_manifest_hash": "sha256:manifest-v1",
        "node_status": {}, "coverage": {}, "result_ids": [], "approval_version": None, "pending_writes": [],
    }
    patch = await node(state)
    assert patch["node_status"]["mira"] == NodeStatus.UNKNOWN
    assert len(patch["result_ids"]) == 1
    artifact = await LocalArtifactStore(tmp_path).get_json(patch["result_ids"][0])
    assert artifact["artifact_kind"] == "agent_proposal"
    assert "summary" in artifact["proposal"]


@pytest.mark.asyncio
async def test_unregistered_model_returns_blocked_without_fabricated_output() -> None:
    result = await executor_for(register_agent=True).execute(
        agent_id="mira",
        agent_version="1.2.0",
        task_input={"brief_text": "A compact urban vehicle"},
        context=task_context(),
    )

    assert result.status is NodeStatus.BLOCKED
    assert result.proposal is None
    assert "not registered" in result.reason


@pytest.mark.asyncio
async def test_cross_project_evidence_reference_is_rejected_by_backend() -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=valid_proposal(
                    evidence_refs=[
                        {
                            "evidence_id": "ev-1",
                            "project_id": "project-b",
                            "version": "ev-v3",
                            "context_manifest_hash": "sha256:manifest-v1",
                        }
                    ]
                ),
                model_version="mock-v1",
            )
        ]
    )

    with pytest.raises(AgentBackendValidationError, match="Cross-project"):
        await executor_for(provider).execute(
            agent_id="mira",
            agent_version="1.2.0",
            task_input={"brief_text": "A compact urban vehicle"},
            context=task_context(),
        )

    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_schema_repair_is_limited_to_one_and_pass_text_is_not_a_decision() -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output={**valid_proposal(), "next_node": "report_issuance"},
                model_version="mock-v1",
            ),
            ProviderResponse(
                structured_output=valid_proposal(summary="Pass: ready for technical review."),
                model_version="mock-v1",
            ),
        ]
    )

    result = await executor_for(provider).execute(
        agent_id="mira",
        agent_version="1.2.0",
        task_input={"brief_text": "A compact urban vehicle"},
        context=task_context(),
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].repair_errors
    assert provider.requests[1].repair_candidate["next_node"] == "report_issuance"
    assert result.status is NodeStatus.UNKNOWN
    assert result.proposal.status == "unknown"
    assert "Pass" in result.proposal.summary


@pytest.mark.asyncio
async def test_second_invalid_response_is_rejected_without_a_second_repair() -> None:
    invalid = {**valid_proposal(), "next_node": "expert_review"}
    provider = MockProvider(
        [
            ProviderResponse(structured_output=invalid, model_version="mock-v1"),
            ProviderResponse(structured_output=invalid, model_version="mock-v1"),
            ProviderResponse(structured_output=valid_proposal(), model_version="mock-v1"),
        ]
    )

    with pytest.raises(AgentBackendValidationError, match="after the allowed repair"):
        await executor_for(provider).execute(
            agent_id="mira",
            agent_version="1.2.0",
            task_input={"brief_text": "A compact urban vehicle"},
            context=task_context(),
        )

    assert len(provider.requests) == 2


def test_agent_definition_cannot_configure_multiple_schema_repairs() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        definition(max_schema_repairs=2)


@pytest.mark.asyncio
async def test_backend_rejects_evidence_reference_outside_snapshot_manifest() -> None:
    provider = MockProvider(
        [
            ProviderResponse(
                structured_output=valid_proposal(
                    evidence_refs=[
                        {
                            "evidence_id": "ev-1",
                            "project_id": "project-a",
                            "version": "ev-v2",
                            "context_manifest_hash": "sha256:manifest-v1",
                        }
                    ]
                ),
                model_version="mock-v1",
            )
        ]
    )

    with pytest.raises(AgentBackendValidationError, match="version-mismatched"):
        await executor_for(provider).execute(
            agent_id="mira",
            agent_version="1.2.0",
            task_input={"brief_text": "A compact urban vehicle"},
            context=task_context(),
        )


@pytest.mark.asyncio
async def test_backend_rejects_task_scope_and_tools_outside_agent_definition() -> None:
    provider = MockProvider([])
    with pytest.raises(AgentBackendValidationError, match="scope"):
        await executor_for(provider).execute(
            agent_id="mira",
            agent_version="1.2.0",
            task_input={"brief_text": "A compact urban vehicle"},
            context=task_context(context_scope=frozenset({"brief", "finance"})),
        )

    with pytest.raises(AgentBackendValidationError, match="allowlist"):
        await executor_for(provider).execute(
            agent_id="mira",
            agent_version="1.2.0",
            task_input={"brief_text": "A compact urban vehicle"},
            context=task_context(requested_tools=frozenset({"shell"})),
        )


@pytest.mark.asyncio
async def test_unknown_agent_definition_is_blocked() -> None:
    result = await executor_for(register_agent=False).execute(
        agent_id="not-registered",
        agent_version="0",
        task_input={"brief_text": "A compact urban vehicle"},
        context=task_context(),
    )

    assert result.status is NodeStatus.BLOCKED
    assert result.proposal is None


@pytest.mark.asyncio
async def test_model_pass_status_cannot_become_formal_or_agent_status() -> None:
    pass_proposal = valid_proposal(status="PASS")
    provider = MockProvider(
        [
            ProviderResponse(structured_output=pass_proposal, model_version="mock-v1"),
            ProviderResponse(structured_output=pass_proposal, model_version="mock-v1"),
        ]
    )

    with pytest.raises(AgentBackendValidationError, match="after the allowed repair"):
        await executor_for(provider).execute(
            agent_id="mira",
            agent_version="1.2.0",
            task_input={"brief_text": "A compact urban vehicle"},
            context=task_context(),
        )

    assert len(provider.requests) == 2

