"""Runtime model configuration and provider adapters for the orchestrator demo.

Loads .env once at import time and exposes the adapter factories used by
web_demo._build_orchestrator() and the LLM connectivity check.
"""

from __future__ import annotations

import os
from pathlib import Path

from concept_platform.agents import ProviderAdapter
from concept_platform.layer_agents import HttpProviderAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

API_KEY = os.environ.get("DEDICATED_API_KEY", "")
BASE_URL = os.environ.get("DEDICATED_BASE_URL", "")
LAYER_MODEL_ID = os.environ.get("DEDICATED_MODEL_ID", "layer-model")
PROPERTY_MODEL_ID = os.environ.get("DEDICATED_PROPERTY_MODEL_ID", LAYER_MODEL_ID)


def _require_credentials() -> None:
    if not API_KEY or not BASE_URL:
        raise RuntimeError("DEDICATED_API_KEY and DEDICATED_BASE_URL are required in .env.")


def _layer_adapter() -> ProviderAdapter:
    _require_credentials()
    return HttpProviderAdapter(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_id=LAYER_MODEL_ID,
        model_version="layer-model-v1",
    )


def _property_adapter() -> ProviderAdapter:
    _require_credentials()
    return HttpProviderAdapter(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_id=PROPERTY_MODEL_ID,
        model_version="property-model-v1",
    )
