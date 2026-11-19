# VW China AI Concept Platform — Control Plane

LangGraph control-plane framework for the auditable flow from confirmed brief through candidates, specialist analysis, engineering computation, deterministic rules, expert approval, and report issuance.

This repository contains orchestration software only. Engineering models, validated datasets, business rules, and approval roles must be supplied by their owners.

## Stage 1: skeleton and configuration

- Python 3.11+, Pydantic v2, LangGraph, PostgreSQL checkpointer, Celery/Redis, and object storage dependencies are pinned in `pyproject.toml`.
- Runtime settings are validated in `src/concept_platform/config.py`. Production requires S3-compatible storage and does not permit disabling checks.
- The three worker queues are `agents`, `engineering`, and `documents`.
- Retry ownership is fixed to Outbox `fencing_token`; configuration rejects nonzero LangGraph retries so the responsibility cannot move accidentally.
- Observability defaults to the self-hosted event-table path. Optional Langfuse configuration is self-hosted only; LangSmith SaaS is not used.
- `RunState` is restricted to the eight contracted control-plane fields. The graph snapshot follows MD §5; `request_engineering_run` remains an in-node tool interface, not a graph node.
- All model-backed work uses the shared `AgentExecutor`, versioned Pydantic `AgentDefinition`, and provider-neutral async adapter contract. Missing model registrations return `BLOCKED`; proposal schemas cannot produce `PASS` or `FAIL` statuses.
- M.I.R.A. draft and frozen Brief bodies are immutable content-addressed artifacts. Clarification and confirmation resume through LangGraph interrupts; state stores only the artifact references and version hashes.
- Innovator candidates are individually versioned artifacts; server-owned variable ranges/options and evidence references are checked before persistence. The Orchestrator starts the `layer_agent → property_agent → reality_check` chain; `FAIL`/`BLOCKED` returns to the Orchestrator.
- Outbox events have stable idempotency keys, and worker claims are fenced by monotonic lease tokens. Celery is configured with only the three selected queues; automatic task retries are disabled so Outbox fencing owns retry semantics. Audit events reject update/delete/truncate.
- Engineering Core accepts only registered typed model adapters. Its numerical results are immutable object-store artifacts; the graph and queues carry identifiers. Engineering requests are Outbox events and resumed only after the fenced completion is persisted.
- Reality Check loads approved, snapshot-bound rules filtered by project, market, vehicle family, effective date, and permissions. Its typed operators have no dynamic expression evaluation; hard failures dominate and missing inputs remain `UNKNOWN`.
- Approval thread IDs hash the run ID, concept version, evaluation version, evidence manifest hash, and sorted model/rule versions. Expert approval and objection records are separate immutable artifacts and append-only database rows.
- Concept Paper inputs must match the exact approval package digest before the shared AgentExecutor is called. Uncited claims and summaries are kept in review items. Report issuance rechecks the approval, package digest, and version bundle.
- `build_graph` accepts concrete service-backed node overrides; the repository does not invent production engineering models, datasets, rule catalogs, evidence, or approval roles.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
pytest -q
```

Development defaults target local object storage and the mock ProviderAdapter. Production should start from `.env.prod.example`; database and Redis URLs must be supplied by the deployment secret manager, and object storage must use a private S3-compatible bucket. Production rejects local storage and default local database/Redis credentials.

Before serving graph runs, call `setup_control_plane_schema(connection_pool)` once from `concept_platform.database`. It applies the Outbox, audit, approval, and versioned rule tables in a single PostgreSQL transaction.

The sole retry owner is Outbox fencing: Celery automatic retries are disabled and `LANGGRAPH_NODE_RETRIES=0`. When a graph node replays after a crash, its stable idempotency key returns the existing Outbox event; a persisted completion is resumed by reference. The append-only event table is the audit trail; LangGraph checkpoints are for workflow recovery.

## Acceptance gates

Run the contract suite and static checks:

```powershell
python -m pytest -q
ruff check .
```

The suite covers graph node/edge snapshots, fixed state keys and statuses, M.I.R.A. interrupt/freeze, provider and backend scope checks, repeated Outbox delivery, engineering BLOCKED/UNKNOWN behavior, rule ACL and snapshot filtering, four version invalidation dimensions, approval rejection, and issuance rechecks. PostgreSQL, Redis, S3/MinIO, Langfuse, business model adapters, evidence catalogs, and human approval identities require deployment-provided integrations and are not replaced by local success simulations.

## Implemented control-plane sequence

1. Skeleton and configuration
2. RunState, checkpointer, and graph skeleton
3. Shared agent runtime and backend validation
4. M.I.R.A. clarification and Brief freeze
5. Innovator fan-out and idempotent Outbox
6. Independent Engineering Core queue
7. Typed Reality Check rules
8. Version-bound expert approval
9. Concept Paper and issuance recheck
10. Acceptance matrix and dev/prod separation
