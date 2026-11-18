"""Transactional PostgreSQL schema bootstrap for the control plane."""

from __future__ import annotations

from typing import Any

from concept_platform.outbox import APPROVAL_SCHEMA_STATEMENTS, POSTGRES_SCHEMA_STATEMENTS
from concept_platform.reality import POSTGRES_RULE_SCHEMA


async def setup_control_plane_schema(connection_pool: Any) -> None:
    """Apply idempotent DDL in one PostgreSQL transaction before serving runs."""

    statements = (*POSTGRES_SCHEMA_STATEMENTS, *APPROVAL_SCHEMA_STATEMENTS, *POSTGRES_RULE_SCHEMA)
    async with connection_pool.connection() as connection, connection.transaction():
        for statement in statements:
            await connection.execute(statement)
