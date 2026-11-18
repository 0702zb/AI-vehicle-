"""PostgreSQL Outbox, fencing-token delivery, and append-only audit events."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from concept_platform.config import Settings
from concept_platform.queues import AGENTS_QUEUE, DOCUMENTS_QUEUE, ENGINEERING_QUEUE


class TaskQueue(StrEnum):
    AGENTS = AGENTS_QUEUE
    ENGINEERING = ENGINEERING_QUEUE
    DOCUMENTS = DOCUMENTS_QUEUE


class OutboxTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    node_key: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    queue: TaskQueue
    artifact_refs: tuple[str, ...] = ()
    payload_version: str = Field(min_length=1)

    @property
    def idempotency_key(self) -> str:
        material = (
            f"{self.run_id}\x1f{self.node_key}\x1f{self.input_digest}\x1f{self.payload_version}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @property
    def event_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.idempotency_key))


@dataclass(frozen=True)
class EnqueuedTask:
    event_id: str
    idempotency_key: str
    created: bool


@dataclass(frozen=True)
class ClaimedTask:
    event_id: str
    run_id: str
    node_key: str
    input_digest: str
    queue: TaskQueue
    artifact_refs: tuple[str, ...]
    fencing_token: int
    attempt: int


@dataclass(frozen=True)
class OutboxCompletion:
    event_id: str
    status: str
    fencing_token: int
    result_ref: str
    result_version: str
    result_status: str


TASK_NAME_BY_QUEUE = {
    TaskQueue.AGENTS: "concept_platform.tasks.run_agent",
    TaskQueue.ENGINEERING: "concept_platform.tasks.run_engineering",
    TaskQueue.DOCUMENTS: "concept_platform.tasks.run_document",
}


APPROVAL_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS expert_approvals (
        thread_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        approval_ref TEXT NOT NULL,
        concept_version TEXT NOT NULL,
        evaluation_version TEXT NOT NULL,
        evidence_manifest_hash TEXT NOT NULL,
        model_rule_versions JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );""",
    """CREATE TABLE IF NOT EXISTS approval_objections (
        objection_ref TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        version_bundle JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );""",
    """CREATE OR REPLACE FUNCTION reject_approval_mutation() RETURNS trigger AS $$
    BEGIN
        RAISE EXCEPTION 'approval records are append-only';
    END;
    $$ LANGUAGE plpgsql;""",
    """DROP TRIGGER IF EXISTS expert_approvals_immutable ON expert_approvals;""",
    """CREATE TRIGGER expert_approvals_immutable BEFORE UPDATE OR DELETE ON expert_approvals
    FOR EACH ROW EXECUTE FUNCTION reject_approval_mutation();""",
    """DROP TRIGGER IF EXISTS expert_approvals_no_truncate ON expert_approvals;""",
    """CREATE TRIGGER expert_approvals_no_truncate BEFORE TRUNCATE ON expert_approvals
    FOR EACH STATEMENT EXECUTE FUNCTION reject_approval_mutation();""",
    """DROP TRIGGER IF EXISTS approval_objections_immutable ON approval_objections;""",
    """CREATE TRIGGER approval_objections_immutable BEFORE UPDATE OR DELETE ON approval_objections
    FOR EACH ROW EXECUTE FUNCTION reject_approval_mutation();""",
    """DROP TRIGGER IF EXISTS approval_objections_no_truncate ON approval_objections;""",
    """CREATE TRIGGER approval_objections_no_truncate BEFORE TRUNCATE ON approval_objections
    FOR EACH STATEMENT EXECUTE FUNCTION reject_approval_mutation();""",
)


POSTGRES_SCHEMA_STATEMENTS = (
"""CREATE TABLE IF NOT EXISTS outbox (
    event_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    node_key TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    queue_name TEXT NOT NULL CHECK (queue_name IN ('agents', 'engineering', 'documents')),
    artifact_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    payload_version TEXT NOT NULL,
    delivery_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (delivery_status IN ('pending', 'leased', 'completed', 'failed')),
    fencing_token BIGINT NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    lease_until TIMESTAMPTZ,
    result_ref TEXT,
    result_version TEXT,
    result_status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, node_key, input_digest, payload_version)
);""",
"""CREATE INDEX IF NOT EXISTS outbox_ready_idx
    ON outbox (queue_name, delivery_status, created_at);""",
"""ALTER TABLE outbox ADD COLUMN IF NOT EXISTS result_ref TEXT;""",
"""ALTER TABLE outbox ADD COLUMN IF NOT EXISTS result_version TEXT;""",
"""ALTER TABLE outbox ADD COLUMN IF NOT EXISTS result_status TEXT;""",

"""CREATE TABLE IF NOT EXISTS audit_events (
    event_id UUID PRIMARY KEY,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_version TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);""",
"""CREATE INDEX IF NOT EXISTS audit_events_run_idx ON audit_events (run_id, created_at);""",

"""CREATE OR REPLACE FUNCTION reject_audit_event_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only';
END;
$$ LANGUAGE plpgsql;""",
"""DROP TRIGGER IF EXISTS audit_events_immutable ON audit_events;""",
"""CREATE TRIGGER audit_events_immutable
    BEFORE UPDATE OR DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION reject_audit_event_mutation();""",
"""DROP TRIGGER IF EXISTS audit_events_no_truncate ON audit_events;""",
"""CREATE TRIGGER audit_events_no_truncate
    BEFORE TRUNCATE ON audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION reject_audit_event_mutation();""",
)


class AsyncConnection(Protocol):
    def transaction(self): ...

    async def execute(self, query: str, params: tuple[Any, ...] = ()): ...


class OutboxRepository:
    """Persists delivery intent independently; retries are fenced at the Outbox."""

    async def setup(self, connection: AsyncConnection) -> None:
        async with connection.transaction():
            for statement in POSTGRES_SCHEMA_STATEMENTS:
                await connection.execute(statement)

    async def enqueue(self, connection: AsyncConnection, task: OutboxTask) -> EnqueuedTask:
        artifact_refs = json.dumps(task.artifact_refs, separators=(",", ":"))
        async with connection.transaction():
            cursor = await connection.execute(
                """
                INSERT INTO outbox (
                    event_id, idempotency_key, run_id, node_key, input_digest,
                    queue_name, artifact_refs, payload_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING event_id
                """,
                (
                    task.event_id,
                    task.idempotency_key,
                    task.run_id,
                    task.node_key,
                    task.input_digest,
                    task.queue.value,
                    artifact_refs,
                    task.payload_version,
                ),
            )
            row = await cursor.fetchone()
            created = row is not None
            if not created:
                cursor = await connection.execute(
                    "SELECT event_id FROM outbox WHERE idempotency_key = %s",
                    (task.idempotency_key,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("Idempotent Outbox row disappeared during conflict recovery.")
            if created:
                await connection.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, run_id, event_type, entity_id, entity_version, payload
                    ) VALUES (%s, %s, 'outbox_enqueued', %s, %s, %s::jsonb)
                    """,
                    (
                        uuid.uuid4(),
                        task.run_id,
                        task.event_id,
                        task.payload_version,
                        json.dumps(
                            {
                                "node_key": task.node_key,
                                "queue": task.queue.value,
                                "artifact_refs": task.artifact_refs,
                            },
                            separators=(",", ":"),
                        ),
                    ),
                )
            return EnqueuedTask(str(row[0]), task.idempotency_key, created)

    async def claim(
        self,
        connection: AsyncConnection,
        *,
        queue: TaskQueue,
        lease_seconds: int,
    ) -> ClaimedTask | None:
        if lease_seconds < 1:
            raise ValueError("Outbox lease must be positive.")
        async with connection.transaction():
            cursor = await connection.execute(
                """
                UPDATE outbox
                SET delivery_status = 'leased',
                    fencing_token = fencing_token + 1,
                    attempt = attempt + 1,
                    lease_until = now() + (%s * interval '1 second'),
                    updated_at = now()
                WHERE event_id = (
                    SELECT event_id FROM outbox
                    WHERE queue_name = %s
                      AND (delivery_status = 'pending'
                           OR (delivery_status = 'leased' AND lease_until < now()))
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING event_id, run_id, node_key, input_digest, queue_name,
                          artifact_refs, fencing_token, attempt
                """,
                (lease_seconds, queue.value),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            refs = row[5] if isinstance(row[5], list) else json.loads(row[5])
            return ClaimedTask(
                event_id=row[0],
                run_id=row[1],
                node_key=row[2],
                input_digest=row[3],
                queue=TaskQueue(row[4]),
                artifact_refs=tuple(refs),
                fencing_token=row[6],
                attempt=row[7],
            )

    async def complete(
        self,
        connection: AsyncConnection,
        *,
        event_id: str,
        fencing_token: int,
        result_ref: str,
        result_version: str,
        result_status: str,
    ) -> bool:
        async with connection.transaction():
            cursor = await connection.execute(
                """
                UPDATE outbox
                SET delivery_status = 'completed', lease_until = NULL,
                    result_ref = %s, result_version = %s, result_status = %s,
                    updated_at = now()
                WHERE event_id = %s AND fencing_token = %s AND delivery_status = 'leased'
                RETURNING run_id, node_key
                """,
                (result_ref, result_version, result_status, event_id, fencing_token),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            await connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, run_id, event_type, entity_id, entity_version, payload
                ) VALUES (%s, %s, 'outbox_completed', %s, %s, %s::jsonb)
                """,
                (
                    uuid.uuid4(),
                    row[0],
                    event_id,
                    result_version,
                    json.dumps({"node_key": row[1], "result_ref": result_ref}),
                ),
            )
            return True

    async def load_claimed(
        self,
        connection: AsyncConnection,
        *,
        event_id: str,
        fencing_token: int,
    ) -> ClaimedTask | None:
        cursor = await connection.execute(
            """
            SELECT event_id, run_id, node_key, input_digest, queue_name,
                   artifact_refs, fencing_token, attempt
            FROM outbox
            WHERE event_id = %s AND fencing_token = %s AND delivery_status = 'leased'
            """,
            (event_id, fencing_token),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        refs = row[5] if isinstance(row[5], list) else json.loads(row[5])
        return ClaimedTask(
            event_id=row[0],
            run_id=row[1],
            node_key=row[2],
            input_digest=row[3],
            queue=TaskQueue(row[4]),
            artifact_refs=tuple(refs),
            fencing_token=row[6],
            attempt=row[7],
        )

    async def get_completion(
        self, connection: AsyncConnection, event_id: str
    ) -> OutboxCompletion | None:
        cursor = await connection.execute(
            """
            SELECT event_id, delivery_status, fencing_token, result_ref,
                   result_version, result_status
            FROM outbox WHERE event_id = %s
            """,
            (event_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[1] != "completed":
            return None
        return OutboxCompletion(
            event_id=row[0],
            status=row[1],
            fencing_token=row[2],
            result_ref=row[3],
            result_version=row[4],
            result_status=row[5],
        )

    async def fail(
        self,
        connection: AsyncConnection,
        *,
        event_id: str,
        fencing_token: int,
        failure_code: str,
        retryable: bool,
    ) -> bool:
        status = "pending" if retryable else "failed"
        async with connection.transaction():
            cursor = await connection.execute(
                """
                UPDATE outbox
                SET delivery_status = %s, lease_until = NULL, updated_at = now()
                WHERE event_id = %s AND fencing_token = %s AND delivery_status = 'leased'
                RETURNING run_id, node_key
                """,
                (status, event_id, fencing_token),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            await connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, run_id, event_type, entity_id, entity_version, payload
                ) VALUES (%s, %s, 'outbox_failed', %s, %s, %s::jsonb)
                """,
                (
                    uuid.uuid4(),
                    row[0],
                    event_id,
                    f"attempt-{fencing_token}",
                    json.dumps(
                        {"node_key": row[1], "failure_code": failure_code, "retryable": retryable}
                    ),
                ),
            )
            return True


class InMemoryOutbox:
    """Deterministic unit-test ledger with the same idempotency/fencing contract."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    async def enqueue(
        self, connection_or_task: Any, task: OutboxTask | None = None
    ) -> EnqueuedTask:
        # Accept the repository signature for service tests; the connection is unused.
        task = task or connection_or_task
        created = task.idempotency_key not in self._tasks
        if created:
            self._tasks[task.idempotency_key] = {
                "task": task,
                "fencing_token": 0,
                "status": "pending",
                "attempt": 0,
            }
        return EnqueuedTask(task.event_id, task.idempotency_key, created)

    async def claim(self, event_id: str) -> ClaimedTask | None:
        for record in self._tasks.values():
            task = record["task"]
            if task.event_id == event_id and record["status"] == "pending":
                record["fencing_token"] += 1
                record["attempt"] += 1
                record["status"] = "leased"
                return ClaimedTask(
                    event_id=task.event_id,
                    run_id=task.run_id,
                    node_key=task.node_key,
                    input_digest=task.input_digest,
                    queue=task.queue,
                    artifact_refs=task.artifact_refs,
                    fencing_token=record["fencing_token"],
                    attempt=record["attempt"],
                )
        return None

    async def load_claimed(
        self, connection: Any, *, event_id: str, fencing_token: int
    ) -> ClaimedTask | None:
        for record in self._tasks.values():
            task = record["task"]
            if (
                task.event_id == event_id
                and record["status"] == "leased"
                and record["fencing_token"] == fencing_token
            ):
                return ClaimedTask(
                    event_id=task.event_id,
                    run_id=task.run_id,
                    node_key=task.node_key,
                    input_digest=task.input_digest,
                    queue=task.queue,
                    artifact_refs=task.artifact_refs,
                    fencing_token=fencing_token,
                    attempt=record["attempt"],
                )
        return None

    async def complete(
        self,
        connection_or_event_id: Any = None,
        fencing_token: int = 0,
        result_ref: str = "",
        result_version: str = "",
        result_status: str = "completed",
        *,
        event_id: str | None = None,
    ) -> bool:
        event_id = event_id or str(connection_or_event_id)
        for record in self._tasks.values():
            if record["task"].event_id == event_id:
                if record["status"] != "leased" or record["fencing_token"] != fencing_token:
                    return False
                record["status"] = "completed"
                record["result_ref"] = result_ref
                record["result_version"] = result_version
                record["result_status"] = result_status
                return True
        return False

    async def fail(
        self,
        connection: Any = None,
        *,
        event_id: str,
        fencing_token: int,
        failure_code: str,
        retryable: bool,
    ) -> bool:
        for record in self._tasks.values():
            if record["task"].event_id == event_id:
                if record["status"] != "leased" or record["fencing_token"] != fencing_token:
                    return False
                record["status"] = "pending" if retryable else "failed"
                record["failure_code"] = failure_code
                return True
        return False

    async def get_completion(self, connection: Any, event_id: str) -> OutboxCompletion | None:
        for record in self._tasks.values():
            task = record["task"]
            if task.event_id == event_id and record["status"] == "completed":
                return OutboxCompletion(
                    event_id=event_id,
                    status="completed",
                    fencing_token=record["fencing_token"],
                    result_ref=record["result_ref"],
                    result_version=record["result_version"],
                    result_status=record["result_status"],
                )
        return None

    async def expire_and_requeue(self, event_id: str) -> None:
        for record in self._tasks.values():
            if record["task"].event_id == event_id and record["status"] == "leased":
                record["status"] = "pending"
                return


def create_celery_app(settings: Settings):
    """Configure the required three queues without assigning retries to Celery/LangGraph."""

    from celery import Celery
    from kombu import Queue

    app = Celery(
        "concept_platform",
        broker=settings.redis_url.get_secret_value(),
    )
    app.conf.update(
        task_default_queue=AGENTS_QUEUE,
        task_queues=(
            Queue(AGENTS_QUEUE),
            Queue(ENGINEERING_QUEUE),
            Queue(DOCUMENTS_QUEUE),
        ),
        task_routes={
            "concept_platform.tasks.run_agent": {"queue": AGENTS_QUEUE},
            "concept_platform.tasks.run_engineering": {"queue": ENGINEERING_QUEUE},
            "concept_platform.tasks.run_document": {"queue": DOCUMENTS_QUEUE},
        },
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_annotations={"*": {"max_retries": 0}},
    )
    return app


def dispatch_claimed_task(celery_app: Any, task: ClaimedTask) -> None:
    """Publish only an Outbox event reference and fencing token to Redis."""

    celery_app.send_task(
        TASK_NAME_BY_QUEUE[task.queue],
        args=[task.event_id, task.fencing_token],
        task_id=f"{task.event_id}:{task.fencing_token}",
        queue=task.queue.value,
    )


def register_celery_handlers(
    celery_app: Any,
    *,
    agent_handler: Any,
    engineering_handler: Any,
    document_handler: Any,
) -> None:
    """Register worker entry points that receive only Outbox IDs and fencing tokens."""

    handlers = (
        ("concept_platform.tasks.run_agent", agent_handler, AGENTS_QUEUE),
        ("concept_platform.tasks.run_engineering", engineering_handler, ENGINEERING_QUEUE),
        ("concept_platform.tasks.run_document", document_handler, DOCUMENTS_QUEUE),
    )
    for task_name, handler, queue in handlers:
        if not callable(handler):
            raise TypeError(f"Handler for {task_name} must be callable.")

        def register(name: str, task_handler: Any, task_queue: str) -> None:
            @celery_app.task(name=name, bind=True, acks_late=True, max_retries=0)
            def run_outbox_handler(task: Any, event_id: str, fencing_token: int) -> Any:
                if not event_id or fencing_token < 1:
                    raise ValueError("Worker tasks require a valid Outbox event ID and fencing token.")
                result = task_handler(event_id, fencing_token)
                if inspect.isawaitable(result):
                    return asyncio.run(result)
                return result

            run_outbox_handler.queue = task_queue

        register(task_name, handler, queue)
