import pytest
from pydantic import ValidationError

from concept_platform.config import Environment, ObjectStoreBackend, Settings


def test_development_defaults_use_local_storage_and_fencing_token_retries() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env is Environment.DEV
    assert settings.object_store_backend is ObjectStoreBackend.LOCAL
    assert settings.outbox_retry_owner.value == "fencing_token"
    assert settings.langgraph_node_retries == 0


def test_retry_ownership_cannot_be_configured_twice() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 0"):
        Settings(_env_file=None, outbox_retry_owner="fencing_token", langgraph_node_retries=1)


def test_production_rejects_local_object_storage() -> None:
    with pytest.raises(ValidationError, match="requires OBJECT_STORE_BACKEND=s3"):
        Settings(_env_file=None, app_env="prod", object_store_backend="local")


def test_production_requires_external_database_and_redis_configuration() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(
            _env_file=None,
            app_env="prod",
            object_store_backend="s3",
            object_store_bucket="private-bucket",
        )


def test_production_accepts_secret_managed_database_redis_and_s3() -> None:
    settings = Settings(
        _env_file=None,
        app_env="prod",
        database_url="postgresql://platform:secret@db.internal:5432/platform?sslmode=verify-full",
        redis_url="rediss://:secret@cache.internal:6379/0",
        object_store_backend="s3",
        object_store_bucket="private-bucket",
    )
    assert settings.app_env is Environment.PROD


def test_self_hosted_langfuse_requires_private_endpoint_configuration() -> None:
    with pytest.raises(ValidationError, match="requires: LANGFUSE_PUBLIC_KEY"):
        Settings(_env_file=None, observability_backend="langfuse_self_hosted")
