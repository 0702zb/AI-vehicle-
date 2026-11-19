## Summary

<!-- What changed and why? -->

## Control-plane impact

- Graph nodes/edges affected:
- RunState fields affected (must stay contract-compatible):
- Artifact or version bindings affected:
- External integrations affected:

## Verification

- [ ] `python -m pytest -q`
- [ ] `ruff check .`
- [ ] `git diff --check`
- [ ] Added/updated graph snapshot or version invalidation tests where needed
- [ ] Confirmed no secrets, private datasets, or local artifacts are included

## Review notes

<!-- Note untested PostgreSQL/Redis/object-store integrations and any migration or rollout needs. -->
