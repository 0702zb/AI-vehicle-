# Collaboration map

This repository provides control-plane code. It does not own validated business data, engineering model approval, policy decisions, or expert identities.

## Work areas and interface owners

| Work area | Code boundary | Inputs required from collaborating owner | Acceptance evidence |
| --- | --- | --- | --- |
| Workflow/runtime | `graph.py`, `state.py`, `config.py` | Approved node/edge changes and deployment settings | Graph snapshot, state contract, pure-router checks |
| Agent integration | `agents.py`, `mira.py`, `innovator.py`, `documents.py` | Provider credentials/endpoints, prompt versions, AgentDefinitions, evidence access policy | Provider contract tests, schema/reference validation, no PASS/FAIL model status |
| Engineering integration | `engineering.py` | Registered model adapter, input/output schemas, applicability rule, solver and license versions | Adapter validation, version binding, no-model `BLOCKED`, integration test against approved model fixture |
| Rules and evidence | `reality.py`, `artifacts.py` | Approved rule revisions, source references, market scope, permission tags, evidence manifest | ACL/date/snapshot tests, immutable artifact references, deterministic rule results |
| Approval and issuance | `approvals.py`, `documents.py` | Authenticated expert identity source, approval permissions, document template/signature process | Version-bound decision, objection precedence, issuance recheck |
| Platform operations | `outbox.py`, `database.py`, `config.py` | PostgreSQL/Redis/S3 or MinIO endpoints, secret management, backup/retention and observability endpoints | Schema bootstrap, queue replay/idempotency, deployment smoke checks |

These are interface boundaries, not assigned people. The repository maintainers should map actual GitHub users to ownership after the organization confirms the team roster.

## Integration sequence

1. Supply approved interface schemas and version identifiers without adding business conclusions to orchestration code.
2. Implement each adapter behind its existing protocol and add contract tests using sanitized fixtures.
3. Validate authorization and version snapshots before enabling an integration in a deployment environment.
4. Run the full CI suite, then a staging integration against PostgreSQL, Redis, and the selected object store.
5. Record operational owners and escalation paths in the deployment repository; do not put secrets or private datasets in this repository.

## Review rules

- Changes to graph topology require an updated MD §5 snapshot assertion.
- Changes to approval-bound inputs require tests proving a new `thread_id` and no stale approval reuse.
- Changes to queue semantics must preserve the single Outbox fencing-token retry owner and append-only audit records.
- Do not add named CODEOWNERS until valid GitHub handles are supplied by repository maintainers.
