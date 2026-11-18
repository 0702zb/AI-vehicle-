import pytest

from concept_platform.outbox import (
    POSTGRES_SCHEMA_STATEMENTS,
    InMemoryOutbox,
    OutboxTask,
    TaskQueue,
)


def task(input_digest: str = "sha256:input-v1") -> OutboxTask:
    return OutboxTask(
        run_id="run-1",
        node_key="package:concept-v1",
        input_digest=input_digest,
        queue=TaskQueue.AGENTS,
        artifact_refs=("sha256:concept",),
        payload_version="task-v1",
    )


@pytest.mark.asyncio
async def test_repeated_outbox_delivery_uses_same_idempotency_key_and_event() -> None:
    outbox = InMemoryOutbox()

    first = await outbox.enqueue(task())
    repeated = await outbox.enqueue(task())
    changed_input = await outbox.enqueue(task("sha256:input-v2"))

    assert first.created is True
    assert repeated.created is False
    assert repeated.event_id == first.event_id
    assert changed_input.event_id != first.event_id


@pytest.mark.asyncio
async def test_fencing_token_rejects_late_worker_completion() -> None:
    outbox = InMemoryOutbox()
    created = await outbox.enqueue(task())
    first_claim = await outbox.claim(created.event_id)
    await outbox.expire_and_requeue(created.event_id)
    second_claim = await outbox.claim(created.event_id)

    assert first_claim.fencing_token == 1
    assert second_claim.fencing_token == 2
    assert await outbox.complete(created.event_id, first_claim.fencing_token) is False
    assert await outbox.complete(created.event_id, second_claim.fencing_token) is True
    assert await outbox.claim(created.event_id) is None


def test_postgres_audit_event_schema_is_append_only_and_outbox_is_unique() -> None:
    ddl = "\n".join(POSTGRES_SCHEMA_STATEMENTS)

    assert "UNIQUE (run_id, node_key, input_digest, payload_version)" in ddl
    assert "audit_events is append-only" in ddl
    assert "BEFORE UPDATE OR DELETE" in ddl
    assert "BEFORE TRUNCATE" in ddl
