import ast
import inspect

from concept_platform.config import Environment, Settings
from concept_platform.graph import (
    route_after_concept_paper,
    route_after_expert_review,
    route_after_mira,
    route_after_orchestration,
)
from concept_platform.outbox import (
    InMemoryOutbox,
    OutboxTask,
    TaskQueue,
    create_celery_app,
    register_celery_handlers,
)
from concept_platform.state import NodeStatus


def test_all_graph_condition_functions_are_read_only_state_routes():
    routes = (
        route_after_concept_paper,
        route_after_expert_review,
        route_after_mira,
        route_after_orchestration,
    )
    forbidden_calls = {"execute", "query", "connect", "request", "complete", "invoke", "ainvoke"}
    for route in routes:
        tree = ast.parse(inspect.getsource(route))
        assert len(route.__annotations__) <= 2
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls


def test_production_configuration_has_no_switch_for_bypassing_validations():
    settings = Settings(
        _env_file=None,
        app_env=Environment.PROD,
        database_url="postgresql://platform:secret@db.internal:5432/platform",
        redis_url="rediss://:secret@cache.internal:6379/0",
        object_store_backend="s3",
        object_store_bucket="approved-bucket",
        disable_checks=True,
    )
    assert not hasattr(settings, "disable_checks")
    assert settings.app_env is Environment.PROD


async def test_duplicate_delivery_event_cannot_create_second_formal_result():
    outbox = InMemoryOutbox()
    task = OutboxTask(
        run_id="run-idempotent", node_key="engineering_core", input_digest="input-v1",
        queue=TaskQueue.ENGINEERING, artifact_refs=("sha256:input",), payload_version="v1",
    )
    first = await outbox.enqueue(task)
    replay = await outbox.enqueue(task)
    assert first.created
    assert not replay.created
    assert first.event_id == replay.event_id
    claim = await outbox.claim(first.event_id)
    assert claim is not None
    assert await outbox.complete(
        event_id=claim.event_id, fencing_token=claim.fencing_token,
        result_ref="sha256:formal-result", result_version="result-v1", result_status="supported",
    )
    assert await outbox.claim(first.event_id) is None
    completion = await outbox.get_completion(None, first.event_id)
    assert completion is not None
    assert completion.result_ref == "sha256:formal-result"


def test_approval_routes_do_not_promote_non_supported_statuses():
    state = {
        "run_id": "run", "brief_version": "brief", "context_manifest_hash": "manifest",
        "node_status": {"expert_review": NodeStatus.FAIL}, "coverage": {}, "result_ids": [],
        "approval_version": "run:some-hash", "pending_writes": [],
    }
    assert route_after_expert_review(state) == "stop"
    state["node_status"] = {"concept_paper": NodeStatus.UNKNOWN}
    assert route_after_concept_paper(state) == "stop"


def test_celery_uses_only_outbox_reference_and_fencing_token_for_all_three_queues():
    app = create_celery_app(Settings(_env_file=None))
    seen = []

    def handler(queue_name):
        return lambda event_id, token: seen.append((queue_name, event_id, token))

    register_celery_handlers(
        app,
        agent_handler=handler("agents"),
        engineering_handler=handler("engineering"),
        document_handler=handler("documents"),
    )
    tasks = (
        ("concept_platform.tasks.run_agent", "agents"),
        ("concept_platform.tasks.run_engineering", "engineering"),
        ("concept_platform.tasks.run_document", "documents"),
    )
    for index, (task_name, queue_name) in enumerate(tasks, start=1):
        app.tasks[task_name].run(f"event-{index}", index)
        assert app.tasks[task_name].max_retries == 0
    assert seen == [
        ("agents", "event-1", 1),
        ("engineering", "event-2", 2),
        ("documents", "event-3", 3),
    ]
