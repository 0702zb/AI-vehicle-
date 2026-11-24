"""Local web demo server for the multi-concept LangGraph orchestrator.

Runs the deterministic orchestrator (Send fan-out -> per-concept chain ->
human interrupt) with real agents configured in .env. A mock fallback is
available via WEB_DEMO_MOCK=1 for offline use.

Run:  python web_demo.py
Open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

WEB_DIR = ROOT / "web"
USE_MOCK = os.environ.get("WEB_DEMO_MOCK", "").strip().lower() in {"1", "true", "yes"}

# Multi-concept orchestrator sessions: thread_id -> compiled graph
_ORCHESTRATOR_GRAPHS: dict[str, Any] = {}


def _wrap_studio_agent(node, node_key: str, base: dict[str, Any], coverage_fn):
    """Adapt a studio node to the orchestrator contract: read state, write state."""
    async def run(state):
        patch = await node({
            "run_id": base["run_id"],
            "brief_version": state["brief_version"],
            "context_manifest_hash": state["context_manifest_hash"],
            "node_status": {},
            "coverage": coverage_fn(state),
            "result_ids": [], "approval_version": None, "pending_writes": [],
        })
        return {
            "status": patch["node_status"][node_key],
            "result_ids": patch.get("result_ids", []),
            "coverage": patch.get("coverage", {}),
        }
    return run


def _build_orchestrator():
    """Build the 5-concept orchestrator with real agents, or mocks for offline use."""
    from langgraph.checkpoint.memory import InMemorySaver
    from concept_platform.concept_orchestrator import build_concept_orchestrator
    from concept_platform.state import NodeStatus

    if USE_MOCK:
        def agent(name):
            async def run(state):
                return {"status": NodeStatus.SUPPORTED,
                        "result_ids": [f"mock:{name}:{state['concept_id']}"]}
            return run

        layer, prop, reality = (agent("layer"), agent("property"), agent("reality"))
    else:
        from concept_platform.runtime_config import _layer_adapter, _property_adapter
        from concept_platform.layer_agents import create_layer_agent_node
        from concept_platform.property_agent import create_property_agent_node
        from concept_platform.demo_reality import build_demo_reality_check_node

        base = _default_state()
        layer_node = create_layer_agent_node(adapter=_layer_adapter())
        prop_node = create_property_agent_node(adapter=_property_adapter(), property_id="mass")
        reality_node = build_demo_reality_check_node()

        layer = _wrap_studio_agent(
            layer_node, "layer_agent", base,
            lambda state: {"selected_concept": state["text"], "errors": []},
        )
        # Property reads the layer's output from state.coverage, not a return value.
        prop = _wrap_studio_agent(
            prop_node, "property_agent", base,
            lambda state: dict(state.get("coverage") or {}),
        )
        reality = _wrap_studio_agent(reality_node, "reality_check", base, lambda state: {})

    return build_concept_orchestrator(
        layer_agent=layer, property_agent=prop, reality_agent=reality,
        checkpointer=InMemorySaver(),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "__str__") and type(value).__module__ != "builtins":
        return value.value if isinstance(getattr(value, "value"), str) else str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _default_state() -> dict[str, Any]:
    """Load the demo state only to reuse its run_id when adapting studio nodes."""
    state = json.loads((ROOT / "studio_input.json").read_text(encoding="utf-8"))
    state.setdefault("run_id", "orch-run")
    return state


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web-demo] {self.address_string()} - {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/llm-check":
            self._handle_llm_check()
        elif path == "/api/orchestrator/info":
            self._send_json(200, {
                "mock": USE_MOCK,
                "node_order": ["layer_agent", "property_agent", "reality_check"],
                "selection": "post_evaluation_human_gate",
                "capabilities": ["Send fan-out", "reducer aggregation", "checkpoint", "interrupt"],
            })
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_llm_check(self) -> None:
        """Make one minimal real LLM call to verify the API configured in .env."""
        if USE_MOCK:
            self._send_json(200, {
                "ok": True, "mock": True,
                "model": "mock", "message": "Mock 模式：未调用真实 API。",
            })
            return
        try:
            from concept_platform.runtime_config import API_KEY, BASE_URL, LAYER_MODEL_ID
            from concept_platform.layer_agents import HttpProviderAdapter
            from concept_platform.agents import ProviderRequest

            adapter = HttpProviderAdapter(
                api_key=API_KEY, base_url=BASE_URL,
                model_id=LAYER_MODEL_ID, model_version="connectivity-check",
            )
            request = ProviderRequest(
                provider_id="web-demo", model_id=LAYER_MODEL_ID, model_version="connectivity-check",
                agent_id="connectivity-check", agent_version="1", prompt_template_version="1",
                system_instruction='只输出一个 JSON 对象：{"status":"ok"}。不要输出其他内容。',
                task_input={"check": "connectivity"}, task_context={},
                output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
                tool_allowlist=(), token_budget=30,
            )
            async def call_with_retry(attempts: int = 3):
                last = None
                for _ in range(attempts):
                    try:
                        return await adapter.complete(request)
                    except Exception as exc:
                        last = exc
                raise last  # type: ignore[misc]

            start = time.monotonic()
            response = asyncio.run(call_with_retry())
            elapsed = round(time.monotonic() - start, 2)
            self._send_json(200, {
                "ok": True, "mock": False, "model": LAYER_MODEL_ID,
                "base_url": BASE_URL, "elapsed_sec": elapsed,
                "response": response.structured_output,
                "message": "真实 LLM 连接正常。",
            })
        except Exception as exc:
            self._send_json(200, {
                "ok": False, "mock": False,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "message": "真实 LLM 连接失败，请检查 .env 中的 API 凭据与网络。",
            })

    def _handle_orchestrate(self, path: str) -> None:
        from langgraph.types import Command
        from concept_platform.concept_orchestrator import initial_state

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")

            if path == "/api/orchestrate":
                concepts = [str(c).strip() for c in payload.get("concepts", []) if str(c).strip()]
                if not concepts[:1] or len(concepts) > 10:
                    self._send_json(400, {"error": "Provide 1-10 concepts"})
                    return
                thread_id = f"orch-{int(time.time() * 1000)}"
                graph = _build_orchestrator()
                _ORCHESTRATOR_GRAPHS[thread_id] = graph
                state = initial_state(
                    run_id=f"orch-run-{thread_id}",
                    brief_version="brief-v1",
                    context_manifest_hash="sha256:demo",
                    concepts=concepts,
                )
                start = time.monotonic()
                result = asyncio.run(graph.ainvoke(state, {"configurable": {"thread_id": thread_id}}))
                elapsed = round(time.monotonic() - start, 1)
                interrupt_value = result.get("__interrupt__", [None])[0]
                portfolio = _jsonable(result.get("concept_portfolio", {}))
                self._send_json(200, {
                    "thread_id": thread_id,
                    "mode": "mock" if USE_MOCK else "real",
                    "elapsed_sec": elapsed,
                    "portfolio": portfolio,
                    "options": interrupt_value.value["options"] if interrupt_value else [],
                    "kind": interrupt_value.value["kind"] if interrupt_value else None,
                })
                return

            thread_id = str(payload.get("thread_id") or "")
            concept_id = str(payload.get("concept_id") or "")
            graph = _ORCHESTRATOR_GRAPHS.get(thread_id)
            if graph is None:
                self._send_json(404, {"error": "Unknown thread_id"})
                return
            if not concept_id:
                self._send_json(400, {"error": "concept_id is required"})
                return
            result = asyncio.run(graph.ainvoke(
                Command(resume={"concept_id": concept_id, "action": "select"}),
                {"configurable": {"thread_id": thread_id}},
            ))
            self._send_json(200, {
                "selection": _jsonable(result.get("selection", {})),
                "portfolio": _jsonable(result.get("concept_portfolio", {})),
            })
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/api/orchestrate", "/api/orchestrate/select"):
            self._handle_orchestrate(path)
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    port = int(os.environ.get("WEB_DEMO_PORT", "8000"))
    mode = "MOCK" if USE_MOCK else "REAL (LLM via .env)"
    server = ThreadingHTTPServer(("0.0.0.0", port), DemoHandler)
    print(f"Web demo [{mode}] ready at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
