"""Immutable JSON artifact storage; workflow state stores only content IDs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from concept_platform.config import ObjectStoreBackend, Settings

ARTIFACT_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ArtifactNotFoundError(FileNotFoundError):
    pass


class ArtifactStore(Protocol):
    async def put_json(self, payload: Mapping[str, Any]) -> str: ...

    async def get_json(self, artifact_id: str) -> dict[str, Any]: ...


def _encode(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    artifact_id = "sha256:" + hashlib.sha256(data).hexdigest()
    return artifact_id, data


def _digest_for(artifact_id: str) -> str:
    if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise ValueError("Invalid artifact ID.")
    return artifact_id.partition(":")[2]


class LocalArtifactStore:
    """Content-addressed dev store. Repeated writes are idempotent and immutable."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, artifact_id: str) -> Path:
        digest = _digest_for(artifact_id)
        return self.root / digest[:2] / f"{digest}.json"

    async def put_json(self, payload: Mapping[str, Any]) -> str:
        artifact_id, data = _encode(payload)
        path = self._path(artifact_id)
        await asyncio.to_thread(self._write_once, path, data)
        return artifact_id

    @staticmethod
    def _write_once(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError("Content-addressed artifact collision detected.")

    async def get_json(self, artifact_id: str) -> dict[str, Any]:
        path = self._path(artifact_id)
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(artifact_id) from exc
        if "sha256:" + hashlib.sha256(data).hexdigest() != artifact_id:
            raise RuntimeError("Stored artifact digest does not match its ID.")
        return json.loads(data)


class S3ArtifactStore:
    """S3/MinIO compatible immutable JSON store using content-addressed keys."""

    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        endpoint_url: str | None = None,
        prefix: str = "artifacts",
    ) -> None:
        if not bucket:
            raise ValueError("An object-store bucket is required.")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        import boto3

        self._client = boto3.client(
            "s3", region_name=region, endpoint_url=endpoint_url
        )

    def _key(self, artifact_id: str) -> str:
        digest = _digest_for(artifact_id)
        return f"{self.prefix}/{digest[:2]}/{digest}.json"

    async def put_json(self, payload: Mapping[str, Any]) -> str:
        artifact_id, data = _encode(payload)
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=self._key(artifact_id),
            Body=data,
            ContentType="application/json; charset=utf-8",
            Metadata={"sha256": artifact_id.partition(":")[2]},
        )
        return artifact_id

    async def get_json(self, artifact_id: str) -> dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                self._client.get_object, Bucket=self.bucket, Key=self._key(artifact_id)
            )
        except self._client.exceptions.NoSuchKey as exc:
            raise ArtifactNotFoundError(artifact_id) from exc
        data = response["Body"].read()
        if "sha256:" + hashlib.sha256(data).hexdigest() != artifact_id:
            raise RuntimeError("Stored artifact digest does not match its ID.")
        return json.loads(data)


def artifact_store_from_settings(settings: Settings) -> ArtifactStore:
    if settings.object_store_backend is ObjectStoreBackend.LOCAL:
        return LocalArtifactStore(settings.object_store_local_path)
    return S3ArtifactStore(
        bucket=settings.object_store_bucket or "",
        region=settings.object_store_region,
        endpoint_url=settings.object_store_endpoint_url,
    )
