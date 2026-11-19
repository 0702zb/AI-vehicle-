# Contributing

Thanks for helping improve the VW China AI Concept Platform control plane.

## Development setup

Use Python 3.11 or newer and install the development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```powershell
python -m pytest -q
ruff check .
git diff --check
```

## Change guidelines

- Keep the `RunState` field set and node status vocabulary unchanged unless the control-plane contract is updated with tests.
- Put evidence and report bodies in the artifact store; graph state should carry references and version identifiers only.
- Keep conditional graph routers pure functions of state.
- Add or modify a single shared `AgentDefinition` rather than adding an Agent-specific executor stack.
- Engineering values must come from registered Engineering Core adapters. Missing registrations remain `BLOCKED`.
- Add regression tests for changed graph edges, version invalidation, ACL checks, idempotency, or human approval behavior.
- Do not commit credentials, private datasets, generated artifacts, local object-store content, or `.env` files.
- Do not enable LangSmith SaaS. Observability must remain self-hosted or use the append-only event table.

## Pull request process

1. Keep each pull request focused on one control-plane change or a tightly related set of changes.
2. Describe the affected graph nodes, persisted artifacts, version bindings, and compatibility impact.
3. Include test evidence and note any PostgreSQL/Redis/object-store integration that was not run locally.
4. Request review from the maintainers responsible for the affected component. Repository owners can define CODEOWNERS after GitHub handles are known.

Business decisions, engineering model validation, data licensing, and approval-role assignment remain with their designated owners; code contributions must not invent these inputs.
