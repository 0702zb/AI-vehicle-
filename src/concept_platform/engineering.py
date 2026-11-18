"""Registered Engineering Core models and independent Outbox queue requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from concept_platform.artifacts import ArtifactStore
from concept_platform.config import Environment
from concept_platform.outbox import (
    ClaimedTask,
    EnqueuedTask,
    OutboxRepository,
    OutboxTask,
    TaskQueue,
)
from concept_platform.state import NodeStatus, RunState


class ModelPurpose(StrEnum):
    FORMAL = "formal"
    DEMO = "demo"


class EngineeringMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Decimal
    unit: str = Field(min_length=1)
    uncertainty: Decimal | None = None
    source_component: str | None = None

    @field_validator("value", "uncertainty")
    @classmethod
    def finite_decimal(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("Engineering metric values must be finite.")
        return value


class EngineeringModelOutput(BaseModel):
    """Base schema for model-produced values; status is not PASS/FAIL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics: dict[str, EngineeringMetric]
    warnings: tuple[str, ...] = ()
    iteration_count: int = Field(default=1, ge=0)
    convergence_status: str | None = None


class EngineeringRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    concept_version: str = Field(min_length=1)
    brief_version: str = Field(min_length=1)
    evaluation_version: str = Field(min_length=1)
    context_manifest_hash: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    input_artifact_ref: str = Field(min_length=1)
    solver_config_version: str = Field(min_length=1)
    requested_by_node: str = Field(min_length=1)


class EngineeringInputArtifact(BaseModel):
    """Immutable, version-bound input envelope stored outside RunState."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: str = "engineering_input"
    run_id: str
    project_id: str
    concept_id: str
    concept_version: str
    brief_version: str
    evaluation_version: str
    context_manifest_hash: str
    model_id: str
    model_version: str
    solver_config_version: str
    parameters: dict[str, Any]


class EngineeringModelDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema: type[BaseModel]
    output_schema: type[EngineeringModelOutput]
    applicability_rule_id: str = Field(min_length=1)
    solver_config_version: str = Field(min_length=1)
    purpose: ModelPurpose = ModelPurpose.FORMAL
    runtime: str = Field(min_length=1)
    license_ref: str = Field(min_length=1)
    timeout_seconds: float = Field(default=300, gt=0)


class EngineeringContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    concept_id: str
    concept_version: str
    brief_version: str
    evaluation_version: str
    context_manifest_hash: str
    solver_config_version: str
    environment: Environment


class EngineeringModelAdapter(Protocol):
    async def validate_inputs(
        self, inputs: BaseModel, context: EngineeringContext
    ) -> None: ...

    async def run(
        self,
        inputs: BaseModel,
        context: EngineeringContext,
        cancellation: asyncio.Event | None = None,
    ) -> BaseModel: ...

    async def validate_outputs(
        self, output: EngineeringModelOutput, context: EngineeringContext
    ) -> None: ...


class ApplicabilityCheck(Protocol):
    async def __call__(
        self, inputs: BaseModel, context: EngineeringContext
    ) -> bool: ...


@dataclass(frozen=True)
class RegisteredEngineeringModel:
    definition: EngineeringModelDefinition
    adapter: EngineeringModelAdapter
    applicability_check: ApplicabilityCheck


class EngineeringModelRegistry:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], RegisteredEngineeringModel] = {}

    def register(
        self,
        definition: EngineeringModelDefinition,
        adapter: EngineeringModelAdapter,
        applicability_check: ApplicabilityCheck,
    ) -> None:
        key = (definition.model_id, definition.version)
        if key in self._models:
            raise ValueError(f"Engineering model already registered: {key}")
        self._models[key] = RegisteredEngineeringModel(
            definition, adapter, applicability_check
        )

    def resolve(self, model_id: str, version: str) -> RegisteredEngineeringModel | None:
        return self._models.get((model_id, version))


class EngineeringRunStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class EngineeringResultArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: str = "engineering_result"
    status: NodeStatus
    run_id: str
    project_id: str
    concept_id: str
    concept_version: str
    brief_version: str
    evaluation_version: str
    context_manifest_hash: str
    model_id: str
    model_version: str
    model_purpose: ModelPurpose
    solver_config_version: str
    input_artifact_ref: str
    metrics: dict[str, EngineeringMetric] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    iteration_count: int = 0
    convergence_status: str | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class EngineeringOutcome:
    status: NodeStatus
    result_ref: str
    model_id: str
    model_version: str


class EngineeringCore:
    def __init__(
        self,
        *,
        registry: EngineeringModelRegistry,
        artifact_store: ArtifactStore,
        environment: Environment,
    ) -> None:
        self._registry = registry
        self._artifact_store = artifact_store
        self._environment = environment

    async def run(
        self,
        request: EngineeringRunRequest,
        context: EngineeringContext,
        *,
        cancellation: asyncio.Event | None = None,
    ) -> EngineeringOutcome:
        registered = self._registry.resolve(request.model_id, request.model_version)
        if registered is None:
            return await self._blocked(request, "model_not_registered")

        definition = registered.definition
        if context.environment is not self._environment:
            return await self._blocked(request, "worker_environment_mismatch")
        if definition.purpose is ModelPurpose.DEMO and self._environment is Environment.PROD:
            return await self._blocked(request, "demo_model_prohibited_in_production")
        if request.solver_config_version != definition.solver_config_version:
            return await self._blocked(request, "solver_config_version_mismatch")

        try:
            self._validate_context(request, context)
            input_payload = await self._artifact_store.get_json(request.input_artifact_ref)
            envelope = EngineeringInputArtifact.model_validate(input_payload)
            self._validate_input_binding(request, envelope)
            inputs = definition.input_schema.model_validate(envelope.parameters)
            await registered.adapter.validate_inputs(inputs, context)
            if not await registered.applicability_check(inputs, context):
                return await self._blocked(request, "model_applicability_not_satisfied")
            async with asyncio.timeout(definition.timeout_seconds):
                raw_output = await registered.adapter.run(inputs, context, cancellation)
            if not isinstance(raw_output, BaseModel):
                raise TypeError("Engineering adapter must return a Pydantic output model.")
            output = definition.output_schema.model_validate(raw_output.model_dump())
            await registered.adapter.validate_outputs(output, context)
        except TimeoutError:
            return await self._blocked(request, "model_timeout")
        except FileNotFoundError:
            return await self._blocked(request, "engineering_input_artifact_not_found")
        except ValueError as exc:
            message = str(exc)
            if "input artifact binding mismatch" in message:
                return await self._blocked(request, "input_snapshot_mismatch")
            if "request context mismatch" in message:
                return await self._blocked(request, "request_context_mismatch")
            return await self._blocked(request, "model_validation_or_execution:ValueError")
        except (TypeError, KeyError) as exc:
            return await self._blocked(request, f"model_validation_or_execution:{type(exc).__name__}")

        artifact = EngineeringResultArtifact(
            status=(
                NodeStatus.UNKNOWN
                if definition.purpose is ModelPurpose.DEMO
                else NodeStatus.SUPPORTED
            ),
            run_id=request.run_id,
            project_id=request.project_id,
            concept_id=request.concept_id,
            concept_version=request.concept_version,
            brief_version=request.brief_version,
            evaluation_version=request.evaluation_version,
            context_manifest_hash=request.context_manifest_hash,
            model_id=definition.model_id,
            model_version=definition.version,
            model_purpose=definition.purpose,
            solver_config_version=definition.solver_config_version,
            input_artifact_ref=request.input_artifact_ref,
            metrics=output.metrics,
            warnings=output.warnings,
            iteration_count=output.iteration_count,
            convergence_status=output.convergence_status,
        )
        artifact_ref = await self._artifact_store.put_json(artifact.model_dump(mode="json"))
        return EngineeringOutcome(artifact.status, artifact_ref, definition.model_id, definition.version)

    @staticmethod
    def _validate_context(request: EngineeringRunRequest, context: EngineeringContext) -> None:
        for name in (
            "project_id",
            "concept_id",
            "concept_version",
            "brief_version",
            "evaluation_version",
            "context_manifest_hash",
            "solver_config_version",
        ):
            if getattr(request, name) != getattr(context, name):
                raise ValueError(f"Engineering request context mismatch for {name}.")

    @staticmethod
    def _validate_input_binding(
        request: EngineeringRunRequest, envelope: EngineeringInputArtifact
    ) -> None:
        for name in (
            "run_id",
            "project_id",
            "concept_id",
            "concept_version",
            "brief_version",
            "evaluation_version",
            "context_manifest_hash",
            "model_id",
            "model_version",
            "solver_config_version",
        ):
            if getattr(request, name) != getattr(envelope, name):
                raise ValueError(f"Engineering input artifact binding mismatch for {name}.")

    async def _blocked(
        self, request: EngineeringRunRequest, reason: str
    ) -> EngineeringOutcome:
        artifact = EngineeringResultArtifact(
            status=NodeStatus.BLOCKED,
            run_id=request.run_id,
            project_id=request.project_id,
            concept_id=request.concept_id,
            concept_version=request.concept_version,
            brief_version=request.brief_version,
            evaluation_version=request.evaluation_version,
            context_manifest_hash=request.context_manifest_hash,
            model_id=request.model_id,
            model_version=request.model_version,
            model_purpose=(
                self._registry.resolve(request.model_id, request.model_version).definition.purpose
                if self._registry.resolve(request.model_id, request.model_version)
                else ModelPurpose.FORMAL
            ),
            solver_config_version=request.solver_config_version,
            input_artifact_ref=request.input_artifact_ref,
            blocked_reason=reason,
        )
        artifact_ref = await self._artifact_store.put_json(artifact.model_dump(mode="json"))
        return EngineeringOutcome(NodeStatus.BLOCKED, artifact_ref, request.model_id, request.model_version)


@dataclass(frozen=True)
class EngineeringRequestReceipt:
    status: NodeStatus
    event_id: str | None
    request_ref: str | None
    reason: str | None = None
    result_ref: str | None = None


class EngineeringRequestService:
    """Sole `request_engineering_run` interface available to Agent/domain nodes."""

    def __init__(
        self,
        *,
        registry: EngineeringModelRegistry,
        artifact_store: ArtifactStore,
        outbox: OutboxRepository,
    ) -> None:
        self._registry = registry
        self._artifact_store = artifact_store
        self._outbox = outbox

    async def request_engineering_run(
        self, connection: Any, request: EngineeringRunRequest
    ) -> EngineeringRequestReceipt:
        registered = self._registry.resolve(request.model_id, request.model_version)
        if registered is None:
            result_ref = await self._blocked_result(request, "model_not_registered")
            return EngineeringRequestReceipt(
                NodeStatus.BLOCKED, None, None, "model_not_registered", result_ref
            )
        if request.solver_config_version != registered.definition.solver_config_version:
            result_ref = await self._blocked_result(request, "solver_config_version_mismatch")
            return EngineeringRequestReceipt(
                NodeStatus.BLOCKED,
                None,
                None,
                "solver_config_version_mismatch",
                result_ref,
            )

        try:
            input_payload = await self._artifact_store.get_json(request.input_artifact_ref)
            # Validate the immutable input artifact before it is placed on the queue.
            envelope = EngineeringInputArtifact.model_validate(input_payload)
            EngineeringCore._validate_input_binding(request, envelope)
            registered.definition.input_schema.model_validate(envelope.parameters)
        except (ValueError, TypeError) as exc:
            reason = (
                "input_snapshot_mismatch"
                if "binding mismatch" in str(exc)
                else "model_input_schema_invalid"
            )
            result_ref = await self._blocked_result(request, reason, registered.definition.purpose)
            return EngineeringRequestReceipt(NodeStatus.BLOCKED, None, None, reason, result_ref)
        except FileNotFoundError:
            reason = "engineering_input_artifact_not_found"
            result_ref = await self._blocked_result(
                request, reason, registered.definition.purpose
            )
            return EngineeringRequestReceipt(NodeStatus.BLOCKED, None, None, reason, result_ref)
        request_ref = await self._artifact_store.put_json(request.model_dump(mode="json"))
        digest_payload = {
            "request_ref": request_ref,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "solver_config_version": request.solver_config_version,
            "context_manifest_hash": request.context_manifest_hash,
        }
        input_digest = "sha256:" + hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        task = OutboxTask(
            run_id=request.run_id,
            node_key=request.requested_by_node + ":" + request.concept_version,
            input_digest=input_digest,
            queue=TaskQueue.ENGINEERING,
            artifact_refs=(request_ref, request.input_artifact_ref),
            payload_version="engineering-request-v1",
        )
        outbox_result: EnqueuedTask = await self._outbox.enqueue(connection, task)
        return EngineeringRequestReceipt(NodeStatus.UNKNOWN, outbox_result.event_id, request_ref)

    async def _blocked_result(
        self,
        request: EngineeringRunRequest,
        reason: str,
        purpose: ModelPurpose = ModelPurpose.FORMAL,
    ) -> str:
        artifact = EngineeringResultArtifact(
            status=NodeStatus.BLOCKED,
            run_id=request.run_id,
            project_id=request.project_id,
            concept_id=request.concept_id,
            concept_version=request.concept_version,
            brief_version=request.brief_version,
            evaluation_version=request.evaluation_version,
            context_manifest_hash=request.context_manifest_hash,
            model_id=request.model_id,
            model_version=request.model_version,
            model_purpose=purpose,
            solver_config_version=request.solver_config_version,
            input_artifact_ref=request.input_artifact_ref,
            blocked_reason=reason,
        )
        return await self._artifact_store.put_json(artifact.model_dump(mode="json"))


class EngineeringInputResolver(Protocol):
    async def resolve(
        self, run_id: str, state: RunState
    ) -> tuple[EngineeringRunRequest, EngineeringContext]: ...


class EngineeringContextResolver(Protocol):
    async def resolve(self, request: EngineeringRunRequest) -> EngineeringContext: ...


class EngineeringOutboxWorker:
    """Consumes only a live fenced Outbox claim and persists validated result artifacts."""

    def __init__(
        self,
        *,
        outbox: OutboxRepository,
        core: EngineeringCore,
        context_resolver: EngineeringContextResolver,
        artifact_store: ArtifactStore,
    ) -> None:
        self._outbox = outbox
        self._core = core
        self._context_resolver = context_resolver
        self._artifact_store = artifact_store

    async def process_claimed(
        self,
        connection: Any,
        *,
        event_id: str,
        fencing_token: int,
        cancellation: asyncio.Event | None = None,
    ) -> EngineeringOutcome | None:
        claim: ClaimedTask | None = await self._outbox.load_claimed(
            connection, event_id=event_id, fencing_token=fencing_token
        )
        if claim is None or claim.queue is not TaskQueue.ENGINEERING:
            return None
        if len(claim.artifact_refs) != 2:
            await self._outbox.fail(
                connection,
                event_id=event_id,
                fencing_token=fencing_token,
                failure_code="invalid_engineering_outbox_refs",
                retryable=False,
            )
            return None

        request_payload = await self._artifact_store.get_json(claim.artifact_refs[0])
        request = EngineeringRunRequest.model_validate(request_payload)
        if request.run_id != claim.run_id or request.input_artifact_ref != claim.artifact_refs[1]:
            await self._outbox.fail(
                connection,
                event_id=event_id,
                fencing_token=fencing_token,
                failure_code="engineering_outbox_binding_mismatch",
                retryable=False,
            )
            return None

        context = await self._context_resolver.resolve(request)
        outcome = await self._core.run(request, context, cancellation=cancellation)
        completed = await self._outbox.complete(
            connection,
            event_id=event_id,
            fencing_token=fencing_token,
            result_ref=outcome.result_ref,
            result_version=outcome.model_version,
            result_status=outcome.status.value,
        )
        return outcome if completed else None


class EngineeringAwaiter(Protocol):
    async def request_engineering_run(
        self, state: RunState
    ) -> EngineeringRequestReceipt: ...

    async def resolve_result(
        self, event_id: str, resume_value: Mapping[str, Any], state: RunState
    ) -> EngineeringOutcome: ...


class AsyncConnectionPool(Protocol):
    def connection(self): ...


class PostgresEngineeringAwaiter:
    """Idempotently enqueue a run and resolve only a persisted fenced completion."""

    def __init__(
        self,
        *,
        connection_pool: AsyncConnectionPool,
        input_resolver: EngineeringInputResolver,
        request_service: EngineeringRequestService,
        outbox: OutboxRepository,
        artifact_store: ArtifactStore,
    ) -> None:
        self._connection_pool = connection_pool
        self._input_resolver = input_resolver
        self._request_service = request_service
        self._outbox = outbox
        self._artifact_store = artifact_store

    async def request_engineering_run(self, state: RunState) -> EngineeringRequestReceipt:
        request, _context = await self._input_resolver.resolve(state["run_id"], state)
        async with self._connection_pool.connection() as connection:
            return await self._request_service.request_engineering_run(connection, request)

    async def resolve_result(
        self, event_id: str, resume_value: Mapping[str, Any], state: RunState
    ) -> EngineeringOutcome:
        if resume_value.get("event_id") != event_id:
            raise ValueError("External task resume event ID does not match the checkpointed task.")
        async with self._connection_pool.connection() as connection:
            completion = await self._outbox.get_completion(connection, event_id)
        if completion is None or not completion.result_ref:
            raise ValueError("Engineering Outbox event has no completed result reference.")

        request, _context = await self._input_resolver.resolve(state["run_id"], state)
        result = EngineeringResultArtifact.model_validate(
            await self._artifact_store.get_json(completion.result_ref)
        )
        expected = {
            "run_id": request.run_id,
            "project_id": request.project_id,
            "concept_id": request.concept_id,
            "concept_version": request.concept_version,
            "brief_version": request.brief_version,
            "evaluation_version": request.evaluation_version,
            "context_manifest_hash": request.context_manifest_hash,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "solver_config_version": request.solver_config_version,
            "input_artifact_ref": request.input_artifact_ref,
        }
        if any(getattr(result, key) != value for key, value in expected.items()):
            raise ValueError("Engineering result is not bound to the active run snapshot.")
        if completion.result_status != result.status.value:
            raise ValueError("Engineering result status differs from the Outbox completion record.")
        return EngineeringOutcome(
            result.status,
            completion.result_ref,
            result.model_id,
            result.model_version,
        )


def make_engineering_core_node(awaiter: EngineeringAwaiter):
    from langgraph.types import interrupt

    async def engineering_core_node(state: RunState) -> dict[str, Any]:
        receipt = await awaiter.request_engineering_run(state)
        if receipt.status is NodeStatus.BLOCKED or receipt.event_id is None:
            result_ids = [receipt.result_ref] if receipt.result_ref else []
            return {
                "node_status": {"engineering_core": NodeStatus.BLOCKED},
                **({"result_ids": result_ids} if result_ids else {}),
            }
        resumed = interrupt(
            {
                "kind": "external_task_wait",
                "task_kind": "engineering",
                "event_id": receipt.event_id,
                "request_ref": receipt.request_ref,
            }
        )
        if not isinstance(resumed, Mapping):
            raise TypeError("Engineering task resume value must be a mapping.")
        outcome = await awaiter.resolve_result(receipt.event_id, resumed, state)
        return {
            "node_status": {"engineering_core": outcome.status},
            "result_ids": [outcome.result_ref],
        }

    return engineering_core_node
