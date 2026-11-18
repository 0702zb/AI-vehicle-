from typing import Any

import pytest

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
from concept_platform.innovator import (
    InnovatorInput,
    InnovatorProposal,
    InnovatorService,
    VariableDefinition,
    make_candidate_concept_node,
)
from concept_platform.state import NodeStatus, RunState


class InputResolver:
    def __init__(self, payload: dict[str, Any], context: AgentTaskContext) -> None:
        self.payload = payload
        self.context = context

    async def resolve(self, run_id: str, state: RunState):
        return self.payload, self.context


def sample_state() -> RunState:
    return {
        "run_id": "run-innovator",
        "brief_version": "brief:v1",
        "context_manifest_hash": "sha256:manifest-v1",
        "node_status": {},
        "coverage": {},
        "result_ids": ["sha256:frozen-brief"],
        "approval_version": None,
        "pending_writes": [],
    }


def variable_catalog() -> tuple[VariableDefinition, ...]:
    return (
        VariableDefinition(
            variable_id="drive_mode",
            value_type="enum",
            allowed_values=("BEV", "PHEV"),
            definition_version="drive-mode-v1",
        ),
        VariableDefinition(
            variable_id="wheelbase_mm",
            value_type="number",
            minimum="2500",
            maximum="3000",
            unit="mm",
            definition_version="wheelbase-v1",
        ),
    )


def candidate_payload(candidate_id="c1", wheelbase=2700):
    return {
        "candidate_id": candidate_id,
        "title": "Urban electric package",
        "design_principle": "Prioritize cabin room within approved vehicle bounds.",
        "variables": [
            {"variable_id": "drive_mode", "value": "BEV"},
            {"variable_id": "wheelbase_mm", "value": wheelbase},
        ],
        "assumptions": [],
        "benefits": ["Proposes a packaging direction."],
        "tradeoffs": [],
        "risks": [],
        "validation_items": ["Run registered package model."],
        "evidence_refs": [],
        "model_refs": [],
    }


def proposal(candidates):
    return {
        "project_id": "project-a",
        "concept_id": "concept-seed",
        "concept_version": "concept-v0",
        "context_manifest_hash": "sha256:manifest-v1",
        "status": "assumption",
        "summary": "Two proposed directions need engineering evaluation.",
        "evidence_refs": [],
        "model_refs": [],
        "assumptions": [],
        "risks": [],
        "questions": [],
        "candidates": candidates,
    }


def build_service(tmp_path, candidates, *, budget=3):
    context = AgentTaskContext(
        project_id="project-a",
        concept_id="concept-seed",
        concept_version="concept-v0",
        context_manifest_hash="sha256:manifest-v1",
        allowed_evidence={},
        allowed_models={},
        context_scope=frozenset({"brief", "variables"}),
    )
    provider = MockProvider(
        [ProviderResponse(structured_output=proposal(candidates), model_version="mock-v1")]
    )
    agents = AgentRegistry()
    agents.register(
        AgentDefinition(
            agent_id="innovator",
            version="1.0",
            input_schema=InnovatorInput,
            output_schema=InnovatorProposal,
            prompt_template_version="innovator-v1",
            system_instruction="Generate bounded candidate design proposals.",
            provider_id="mock",
            model_id="innovator-mock",
            model_version="mock-v1",
            context_scope=frozenset({"brief", "variables"}),
        )
    )
    providers = ProviderRegistry()
    providers.register("mock", "innovator-mock", "mock-v1", provider)
    definitions = variable_catalog()
    service = InnovatorService(
        executor=AgentExecutor(agents, providers),
        input_resolver=InputResolver(
            {
                "frozen_brief_ref": "sha256:frozen-brief",
                "brief_version": "brief:v1",
                "context_manifest_hash": "sha256:manifest-v1",
                "candidate_budget": budget,
                "variable_definitions": [item.model_dump(mode="json") for item in definitions],
            },
            context,
        ),
        artifact_store=LocalArtifactStore(tmp_path),
        variable_definitions=definitions,
        agent_version="1.0",
        candidate_budget=budget,
    )
    return service


@pytest.mark.asyncio
async def test_innovator_stores_candidate_versions_by_reference(tmp_path) -> None:
    service = build_service(tmp_path, [candidate_payload()])

    result = await service.generate(sample_state())

    assert result.status is NodeStatus.ASSUMPTION
    assert result.artifact_ref.startswith("sha256:")


@pytest.mark.asyncio
async def test_innovator_rejects_variables_outside_registered_bounds(tmp_path) -> None:
    service = build_service(tmp_path, [candidate_payload(wheelbase=3200)])

    with pytest.raises(ValueError, match="above its registered maximum"):
        await service.generate(sample_state())


@pytest.mark.asyncio
async def test_innovator_rejects_duplicate_normalized_design_combinations(tmp_path) -> None:
    service = build_service(
        tmp_path,
        [candidate_payload("c1"), candidate_payload("c2")],
    )

    with pytest.raises(ValueError, match="duplicate normalized"):
        await service.generate(sample_state())


@pytest.mark.asyncio
async def test_candidate_concept_node_marks_missing_innovator_as_blocked() -> None:
    node = make_candidate_concept_node()
    state = sample_state()
    state["node_status"] = {"innovator": NodeStatus.BLOCKED}

    result = await node(state)

    assert result["node_status"]["candidate_concept"] is NodeStatus.BLOCKED
