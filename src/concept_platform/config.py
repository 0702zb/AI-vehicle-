"""Validated environment configuration for the control plane."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    PROD = "prod"


class ObjectStoreBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


class RetryOwner(StrEnum):
    """This framework assigns redelivery to Outbox fencing tokens."""

    FENCING_TOKEN = "fencing_token"


class ObservabilityBackend(StrEnum):
    EVENT_TABLE = "event_table"
    LANGFUSE_SELF_HOSTED = "langfuse_self_hosted"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Environment = Environment.DEV
    database_url: SecretStr = SecretStr(
        "postgresql://postgres:postgres@localhost:5432/concept_platform"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    object_store_backend: ObjectStoreBackend = ObjectStoreBackend.LOCAL
    object_store_local_path: Path = Path("./data/objects")
    object_store_bucket: str | None = None
    object_store_endpoint_url: str | None = None
    object_store_region: str = "ap-east-1"

    outbox_retry_owner: RetryOwner = RetryOwner.FENCING_TOKEN
    langgraph_node_retries: int = Field(default=0, ge=0, le=0)

    observability_backend: ObservabilityBackend = ObservabilityBackend.EVENT_TABLE
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None

    @model_validator(mode="after")
    def validate_environment_policy(self) -> Settings:
        if self.app_env is Environment.PROD:
            if self.object_store_backend is not ObjectStoreBackend.S3:
                raise ValueError("Production requires OBJECT_STORE_BACKEND=s3.")
            if not self.object_store_bucket:
                raise ValueError("Production requires OBJECT_STORE_BUCKET.")
            if self.database_url.get_secret_value() == "postgresql://postgres:postgres@localhost:5432/concept_platform":
                raise ValueError("Production requires DATABASE_URL from the deployment secret manager.")
            if self.redis_url.get_secret_value() == "redis://localhost:6379/0":
                raise ValueError("Production requires REDIS_URL from the deployment secret manager.")
            if self.object_store_endpoint_url and not self.object_store_endpoint_url.startswith(
                "https://"
            ):
                raise ValueError("Production object-store endpoint must use HTTPS.")
            if self.object_store_backend is ObjectStoreBackend.LOCAL:
                raise ValueError("Production cannot use local object storage.")

        if self.observability_backend is ObservabilityBackend.LANGFUSE_SELF_HOSTED:
            missing = [
                name
                for name, value in (
                    ("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key),
                    ("LANGFUSE_SECRET_KEY", self.langfuse_secret_key),
                    ("LANGFUSE_HOST", self.langfuse_host),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Self-hosted Langfuse requires: " + ", ".join(missing)
                )
            if self.langfuse_host and self.langfuse_host.startswith("https://cloud.langfuse.com"):
                raise ValueError("Langfuse SaaS is prohibited; configure a self-hosted endpoint.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide validated settings object."""

    return Settings()
