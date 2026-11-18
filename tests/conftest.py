"""Isolate graph tests from an App-Control-blocked optional native xxhash DLL."""

from __future__ import annotations

import hashlib
import importlib
import sys
import types

try:
    importlib.import_module("xxhash")
except ImportError as exc:
    if "DLL load failed" not in str(exc):
        raise

    class _TestDigest:
        def __init__(self, data: bytes) -> None:
            self._digest = hashlib.blake2b(data, digest_size=16).digest()

        def digest(self) -> bytes:
            return self._digest

        def intdigest(self) -> int:
            return int.from_bytes(self._digest, "big")

    fallback = types.ModuleType("xxhash")
    fallback.xxh3_128 = _TestDigest  # type: ignore[attr-defined]
    fallback.xxh3_128_hexdigest = (  # type: ignore[attr-defined]
        lambda data: _TestDigest(data).digest().hex()
    )
    sys.modules["xxhash"] = fallback
