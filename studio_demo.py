"""User-facing Studio demo: run the graph and print a compact result.

Unlike ``demo_property_flow.py`` (which writes a long developer log), this script
shows only the short summary a normal user needs. It runs with mock providers by
default so it works without API credentials.

Run: python studio_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from concept_platform.presentation import build_user_summary, render_summary_text  # noqa: E402

os.environ.setdefault("PROPERTY_DEMO_USE_MOCK", "1")
os.environ.setdefault("PROPERTY_DEMO_LAYER_USE_MOCK", "1")


def _load_state() -> dict[str, Any]:
    state = json.loads((ROOT / "studio_input.json").read_text(encoding="utf-8"))
    state.setdefault("node_status", {})
    state.setdefault("coverage", {})
    state.setdefault("result_ids", [])
    state.setdefault("approval_version", None)
    state.setdefault("pending_writes", [])
    return state


async def _run(state: dict[str, Any]) -> dict[str, Any]:
    from demo_property_flow import run_flow

    result = await run_flow(use_rag=False, state=state)
    return result["result"]


def main() -> None:
    state = _load_state()
    final_state = asyncio.run(_run(state))
    print(render_summary_text(build_user_summary(final_state)))


if __name__ == "__main__":
    main()
