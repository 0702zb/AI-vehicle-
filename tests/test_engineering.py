import asyncio
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from concept_platform.artifacts import LocalArtifactStore
from concept_platform.config import Environment
from concept_platform.engineering import (
    EngineeringContext,
    EngineeringCore,
    EngineeringInputArtifact,
    EngineeringMetric,
    EngineeringModelDefinition,
    EngineeringModelOutput,
    EngineeringModelRegistry,
    EngineeringOutboxWorker,
    EngineeringOutcome,
    EngineeringRequestReceipt,
    EngineeringRequestService,
    EngineeringRunRequest,
    ModelPurpose,
    make_engineering_core_node,
)
from concept_platform.graph import build_graph
from concept_platform.outbox import InMemoryOutbox
from concept_platform.state import NodeStatus, RunState


class InputParameters(BaseModel):
    mass_kg: Decimal


class ModelOutput(EngineeringModelOutput):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FakeEngineeringModel:
    async def validate_inputs(self, inputs, context) -> None:
        if inputs.mass_kg <= 0:
            raise ValueError("mass must be positive")

    async def run(self, inputs, context, cancellation=None):
        return ModelOutput(
            metrics={
                "mass": EngineeringMetric(value=inputs.mass_kg, unit="kg"),
                "range": EngineeringMetric(value=Decimal(420), unit="km"),
            },
            warnings=(),
            iteration_count=12,
            convergence_status="converged",
        )

    async def validate_outputs(self, output, context) -> None:
        assert all(metric.value.is_finite() for metric in output.metrics.values())


class SlowEngineeringModel(FakeEngineeringModel):
    async def run(self, inputs, context, cancellation=None):
        await asyncio.sleep(0.05)
        return await super().run(inputs, context, cancellation)


class FakeEngineeringContextResolver:
    async def resolve(self, request):
        return context()


async def applies(inputs, context) -> bool:
    return True


def definition(*, purpose=ModelPurpose.FORMAL, timeout=1):
    return EngineeringModelDefinition(
        model_id="lds",
        version="lds-v1",
        input_schema=InputParameters,
        output_schema=ModelOutput,
        applicability_rule_id="approved-vehicle-domain-v1",
        solver_config_version="solver-v1",
        purpose=purpose,
        runtime="python-worker",
        license_ref="license-registry-entry",
        timeout_seconds=timeout,
    )


def request(**overrides):
    values = {
        "run_id": "run-eng",
        "project_id": "project-a",
        "concept_id": "concept-1",
        "concept_version": "concept-v2",
        "brief_version": "brief-v2",
        "evaluation_version": "evaluation-v1",
        "context_manifest_hash": "sha256:manifest-v2",
        "model_id": "lds",
        "model_version": "lds-v1",
        "input_artifact_ref": "sha256:placeholder",
        "solver_config_version": "solver-v1",
        "requested_by_node": "drive_energy",
    }
    values.update(overrides)
    return EngineeringRunRequest(**values)


def context(**overrides):
    values = {
        "project_id": "project-a",
        "concept_id": "concept-1",
        "concept_version": "concept-v2",
        "brief_version": "brief-v2",
        "evaluation_version": "evaluation-v1",
        "context_manifest_hash": "sha256:manifest-v2",
        "solver_config_version": "solver-v1",
        "environment": Environment.DEV,
    }
    values.update(overrides)
    return EngineeringContext(**values)


async def input_ref(store, **overrides):
    values = {
        "run_id": "run-eng",
        "project_id": "project-a",
        "concept_id": "concept-1",
        "concept_version": "concept-v2",
        "brief_version": "brief-v2",
        "evaluation_version": "evaluation-v1",
        "context_manifest_hash": "sha256:manifest-v2",
        "model_id": "lds",
        "model_version": "lds-v1",
        "solver_config_version": "solver-v1",
        "parameters": {"mass_kg": "1800"},
    }
    values.update(overrides)
    return await store.put_json(EngineeringInputArtifact(**values).model_dump(mode="json"))


def registry(model=None, *, purpose=ModelPurpose.FORMAL, timeout=1):
    result = EngineeringModelRegistry()
    if model is not None:
        result.register(definition(purpose=purpose, timeout=timeout), model, applies)
    return result


@pytest.mark.asyncio
async def test_unregistered_engineering_model_returns_blocked_without_numeric_defaults(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store)
    result = await EngineeringCore(
        registry=registry(), artifact_store=store, environment=Environment.DEV
    ).run(request(input_artifact_ref=ref), context())
    artifact = await store.get_json(result.result_ref)

    assert result.status is NodeStatus.BLOCKED
    assert artifact["blocked_reason"] == "model_not_registered"
    assert artifact["metrics"] == {}


@pytest.mark.asyncio
async def test_registered_model_output_is_stored_outside_state_with_provenance(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store)
    result = await EngineeringCore(
        registry=registry(FakeEngineeringModel()), artifact_store=store, environment=Environment.DEV
    ).run(request(input_artifact_ref=ref), context())
    artifact = await store.get_json(result.result_ref)

    assert result.status is NodeStatus.SUPPORTED
    assert artifact["model_id"] == "lds"
    assert artifact["model_version"] == "lds-v1"
    assert artifact["metrics"]["range"] == {"value": "420", "unit": "km", "uncertainty": None, "source_component": None}
    assert "metrics" not in {"run_id": "run-eng", "result_ids": [result.result_ref]}


@pytest.mark.asyncio
async def test_input_snapshot_mismatch_blocks_run(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store, concept_version="concept-v1")
    result = await EngineeringCore(
        registry=registry(FakeEngineeringModel()), artifact_store=store, environment=Environment.DEV
    ).run(request(input_artifact_ref=ref), context())
    artifact = await store.get_json(result.result_ref)

    assert result.status is NodeStatus.BLOCKED
    assert artifact["blocked_reason"] == "input_snapshot_mismatch"


@pytest.mark.asyncio
async def test_demo_model_result_is_unknown_and_production_blocks_demo_model(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store)
    dev_result = await EngineeringCore(
        registry=registry(FakeEngineeringModel(), purpose=ModelPurpose.DEMO),
        artifact_store=store,
        environment=Environment.DEV,
    ).run(request(input_artifact_ref=ref), context())
    prod_result = await EngineeringCore(
        registry=registry(FakeEngineeringModel(), purpose=ModelPurpose.DEMO),
        artifact_store=store,
        environment=Environment.PROD,
    ).run(request(input_artifact_ref=ref), context(environment=Environment.PROD))

    assert dev_result.status is NodeStatus.UNKNOWN
    assert prod_result.status is NodeStatus.BLOCKED


@pytest.mark.asyncio
async def test_timeout_produces_blocked_artifact_not_zero_or_success(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store)
    result = await EngineeringCore(
        registry=registry(SlowEngineeringModel(), timeout=0.001),
        artifact_store=store,
        environment=Environment.DEV,
    ).run(request(input_artifact_ref=ref), context())
    artifact = await store.get_json(result.result_ref)

    assert result.status is NodeStatus.BLOCKED
    assert artifact["blocked_reason"] == "model_timeout"
    assert artifact["metrics"] == {}


@pytest.mark.asyncio
async def test_request_engineering_run_enqueues_only_registered_model(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store)
    outbox = InMemoryOutbox()
    missing = EngineeringRequestService(
        registry=registry(), artifact_store=store, outbox=outbox
    )
    blocked = await missing.request_engineering_run(
        None, request(input_artifact_ref=ref)
    )
    assert blocked.status is NodeStatus.BLOCKED
    assert blocked.event_id is None
    assert blocked.result_ref is not None
    assert (await store.get_json(blocked.result_ref))["metrics"] == {}

    registered = EngineeringRequestService(
        registry=registry(FakeEngineeringModel()), artifact_store=store, outbox=outbox
    )
    queued = await registered.request_engineering_run(
        None, request(input_artifact_ref=ref)
    )
    repeated = await registered.request_engineering_run(
        None, request(input_artifact_ref=ref)
    )

    assert queued.status is NodeStatus.UNKNOWN
    assert queued.event_id == repeated.event_id
    assert queued.request_ref == repeated.request_ref


@pytest.mark.asyncio
async def test_engineering_worker_requires_live_fencing_claim_and_persists_result(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = await input_ref(store)
    outbox = InMemoryOutbox()
    request_service = EngineeringRequestService(
        registry=registry(FakeEngineeringModel()), artifact_store=store, outbox=outbox
    )
    receipt = await request_service.request_engineering_run(
        None, request(input_artifact_ref=ref)
    )
    claim = await outbox.claim(receipt.event_id)
    worker = EngineeringOutboxWorker(
        outbox=outbox,
        core=EngineeringCore(
            registry=registry(FakeEngineeringModel()),
            artifact_store=store,
            environment=Environment.DEV,
        ),
        context_resolver=FakeEngineeringContextResolver(),
        artifact_store=store,
    )

    result = await worker.process_claimed(
        None, event_id=claim.event_id, fencing_token=claim.fencing_token
    )

    completion = await outbox.get_completion(None, receipt.event_id)
    assert result.status is NodeStatus.SUPPORTED
    assert completion.result_ref == result.result_ref
    assert completion.result_status == "supported"


class FakeEngineeringAwaiter:
    def __init__(self) -> None:
        self.requests = 0

    async def request_engineering_run(self, state):
        self.requests += 1
        return EngineeringRequestReceipt(
            NodeStatus.UNKNOWN, "event-1", "sha256:request-ref"
        )

    async def resolve_result(self, event_id, resume_value, state):
        assert event_id == "event-1"
        return EngineeringOutcome(
            NodeStatus.SUPPORTED, "sha256:engineering-result", "lds", "lds-v1"
        )


def graph_state() -> RunState:
    return {
        "run_id": "run-1",
        "brief_version": "brief-v1",
        "context_manifest_hash": "sha256:manifest",
        "node_status": {},
        "coverage": {},
        "result_ids": [],
        "approval_version": None,
        "pending_writes": [],
    }


@pytest.mark.asyncio
async def test_engineering_external_task_interrupt_resumes_with_result_id_only():
    awaiter = FakeEngineeringAwaiter()
    graph = build_graph(
        checkpointer=InMemorySaver(),
        node_overrides={"engineering_core": make_engineering_core_node(awaiter)},
    )
    config = {"configurable": {"thread_id": "engineering-external-task-test"}}

    paused = await graph.ainvoke(graph_state(), config)
    assert paused["__interrupt__"][0].value["kind"] == "external_task_wait"

    resumed = await graph.ainvoke(Command(resume={"event_id": "event-1"}), config)

    assert awaiter.requests == 2  # Replayed node re-enqueues the same idempotent Outbox task.
    assert resumed["node_status"]["engineering_core"] is NodeStatus.SUPPORTED
    assert "sha256:engineering-result" in resumed["result_ids"]
    assert "metrics" not in resumed
