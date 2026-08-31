import asyncio
import json
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from config.env_config import EnvConfig
from tests.test_webapp_catalog import write_rows
from tests.test_webapp_service import FakeAgent
from webapp.app import WebRuntime, create_app, initialize_runtime
from webapp.catalog import CatalogPresenter
from webapp.service import SessionManager


class FakeManager:
    """Minimal stand-in for RuntimeManager used by /api/runtime endpoints."""

    def __init__(self, runtime: WebRuntime) -> None:
        self.runtime = runtime
        self.switched: list[dict[str, object]] = []

    def runtime_info(self) -> dict[str, object]:
        return {
            "fingerprint": "cpu|dense=no|llm=no|network=no",
            "lut_recommendation": "bm25_rule",
            "lut_ts": 0.9,
            "active": {
                "config_key": "k1",
                "provider": "none",
                "model": "",
                "rerank_backend": "none",
                "retrieval_backend": "bm25",
                "output_strategy": "holdback",
                "llm_intent_enabled": False,
                "fingerprint_enabled": True,
                "category_expand_enabled": True,
                "paraphrase_enabled": True,
                "api_key_set": False,
                "offline": True,
            },
            "providers": {"none": {"label": "Off (rule-based)"}},
            "rerank_backends": {"none": {"label": "Off (rule order)"}},
            "retrieval_backends": {"bm25": {"label": "BM25 (offline)"}},
            "output_strategies": {"holdback": {"label": "Hold-back (best MRR, default)"}},
            "toggles": {},
        }

    def switch(self, cfg: dict[str, object]) -> tuple[WebRuntime, str]:
        self.switched.append(dict(cfg))
        return self.runtime, "k2"


def test_runtime_config_switch_updates_runtime_and_never_leaks_keys(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    manager = FakeManager(runtime)
    app = create_app(runtime=runtime)
    app.state.runtime_container.manager = manager

    with TestClient(app) as client:
        info = client.get("/api/runtime").json()
        assert info["active"]["offline"] is True
        info_raw = json.dumps(info)
        assert '"api_key"' not in info_raw  # no key-bearing field (api_key_set is a bool)
        assert "sk-secret-123" not in info_raw

        payload = {
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "api_key": "sk-secret-123",
            "rerank_backend": "none",
            "retrieval_backend": "bm25",
            "output_strategy": "holdback",
            "llm_intent_enabled": True,
            "fingerprint": True,
            "category_expand": True,
            "paraphrase": True,
            "unexpected_extra": "should-be-stripped",
        }
        resp = client.post("/api/runtime/config", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["sessions_reset"] is True
        raw = json.dumps(body)
        assert "sk-secret-123" not in raw
        assert '"api_key"' not in raw
        assert manager.switched
        last = manager.switched[-1]
        assert last["api_key"] == "sk-secret-123"
        assert last["llm_intent_enabled"] is True
        assert "unexpected_extra" not in last
        assert app.state.runtime_container.runtime is runtime


def test_runtime_config_defaults_missing_fields(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    manager = FakeManager(runtime)
    app = create_app(runtime=runtime)
    app.state.runtime_container.manager = manager

    with TestClient(app) as client:
        resp = client.post("/api/runtime/config", json={"retrieval_backend": "dense"})
        assert resp.status_code == 200
        last = manager.switched[-1]
        assert last["retrieval_backend"] == "dense"
        assert last["llm_provider"] == "none"
        assert last["rerank_backend"] == "none"
        assert last["output_strategy"] == "holdback"
        assert last["fingerprint"] is True
        assert last["llm_intent_enabled"] is False


def test_runtime_routes_503_without_manager(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        assert client.get("/api/runtime").status_code == 503
        assert client.post("/api/runtime/config", json={}).status_code == 503


def test_concurrent_runtime_switch_publishes_the_managers_latest_runtime(
    tmp_path: Path,
) -> None:
    first_runtime, _ = make_runtime(tmp_path)
    second_runtime = WebRuntime(
        SessionManager(FakeAgent(), first_runtime.catalog, top_k=10),
        first_runtime.catalog,
    )

    class RacingManager:
        def __init__(self) -> None:
            self.active = first_runtime
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()

        def switch(self, cfg: dict[str, object]) -> tuple[WebRuntime, str]:
            if cfg["retrieval_backend"] == "bm25":
                self.active = first_runtime
                self.first_started.set()
                assert self.release_first.wait(timeout=2.0)
                time.sleep(0.05)
                return first_runtime, "first"
            self.active = second_runtime
            self.second_started.set()
            return second_runtime, "second"

        def runtime_info(self) -> dict[str, object]:
            return {"active": {"config_key": "current"}}

    manager = RacingManager()
    app = create_app(runtime=first_runtime)
    app.state.runtime_container.manager = manager
    responses: list[int] = []

    with TestClient(app) as client:
        first = threading.Thread(
            target=lambda: responses.append(
                client.post(
                    "/api/runtime/config", json={"retrieval_backend": "bm25"}
                ).status_code
            )
        )
        second = threading.Thread(
            target=lambda: responses.append(
                client.post(
                    "/api/runtime/config", json={"retrieval_backend": "hybrid"}
                ).status_code
            )
        )
        first.start()
        assert manager.first_started.wait(timeout=1.0)
        second.start()
        manager.second_started.wait(timeout=0.2)
        manager.release_first.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

    assert sorted(responses) == [200, 200]
    assert app.state.runtime_container.runtime is manager.active


def test_cancelled_runtime_switch_still_publishes_completed_runtime(tmp_path: Path) -> None:
    initial, _ = make_runtime(tmp_path)
    replacement = WebRuntime(
        SessionManager(FakeAgent(), initial.catalog, top_k=10),
        initial.catalog,
    )

    class BlockingManager:
        def __init__(self) -> None:
            self.active = initial
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def switch(self, cfg: dict[str, object]) -> tuple[WebRuntime, str]:
            self.started.set()
            assert self.release.wait(timeout=2.0)
            self.active = replacement
            self.finished.set()
            return replacement, "replacement"

        def runtime_info(self) -> dict[str, object]:
            return {"active": {"config_key": "replacement"}}

    async def scenario() -> None:
        manager = BlockingManager()
        app = create_app(runtime=initial)
        app.state.runtime_container.manager = manager
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/runtime/config"
        )
        task = asyncio.create_task(endpoint({"retrieval_backend": "hybrid"}))
        assert await asyncio.to_thread(manager.started.wait, 1.0)
        task.cancel()
        manager.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.finished.is_set()
        assert app.state.runtime_container.runtime is manager.active

    asyncio.run(scenario())


def test_initialize_runtime_validates_selected_catalog_and_disables_duplicate_agent_check(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.jsonl"
    write_rows(selected, [{"parent_asin": "A1", "title": "One"}])
    verified: list[tuple[Path, str, str, bool]] = []
    captured: dict[str, object] = {}

    def verifier(path: Path, expected: str, label: str, skip: bool = False) -> bool:
        verified.append((Path(path), expected, label, skip))
        return True

    def env_loader() -> EnvConfig:
        return EnvConfig.from_env(
            overrides={"skip_data_verify": False, "llm": {"provider": "none"}},
            environ={},
        )

    def agent_factory(path: Path, env: EnvConfig, llm_client: object) -> FakeAgent:
        captured.update(path=Path(path), env=env, llm_client=llm_client)
        return FakeAgent()

    runtime = initialize_runtime(
        selected,
        env_loader=env_loader,
        verifier=verifier,
        agent_factory=agent_factory,
    )

    assert verified[0][0] == selected
    assert captured["path"] == selected
    assert isinstance(captured["env"], EnvConfig)
    assert captured["env"].skip_data_verify is True
    assert runtime.catalog.detail("A1")["title"] == "One"


def test_initialize_runtime_honors_existing_skip_flag(tmp_path: Path) -> None:
    selected = tmp_path / "selected.jsonl"
    write_rows(selected, [{"parent_asin": "A1", "title": "One"}])
    verifier_calls: list[Path] = []

    def env_loader() -> EnvConfig:
        return EnvConfig.from_env(
            overrides={"skip_data_verify": True, "llm": {"provider": "none"}},
            environ={},
        )

    def verifier(path: Path, expected: str, label: str, skip: bool = False) -> bool:
        verifier_calls.append(path)
        return True

    runtime = initialize_runtime(
        selected,
        env_loader=env_loader,
        verifier=verifier,
        agent_factory=lambda path, env, client: FakeAgent(),
    )

    assert verifier_calls == []
    assert runtime.catalog.detail("A1")["title"] == "One"


def test_web_cli_defaults_and_overrides() -> None:
    from webapp.__main__ import parse_args

    defaults = parse_args([])
    assert defaults.catalog == Path("data/catalog.jsonl")
    assert defaults.host == "127.0.0.1"
    assert defaults.port == 8000

    custom = parse_args(["--catalog", "/tmp/catalog.jsonl", "--port", "8080"])
    assert custom.catalog == Path("/tmp/catalog.jsonl")
    assert custom.host == "127.0.0.1"
    assert custom.port == 8080


def make_runtime(tmp_path: Path) -> tuple[WebRuntime, FakeAgent]:
    path = tmp_path / "catalog.jsonl"
    write_rows(path, [{"parent_asin": "A1", "title": "Safe <script> title"}])
    catalog = CatalogPresenter.build(path)
    agent = FakeAgent()
    return WebRuntime(SessionManager(agent, catalog, top_k=10), catalog), agent


def test_session_message_preserves_raw_response_and_separates_products(tmp_path: Path) -> None:
    runtime, agent = make_runtime(tmp_path)
    with TestClient(create_app(runtime=runtime)) as client:
        created = client.post("/api/sessions").json()
        message_id = str(uuid4())
        response = client.post(
            f"/api/sessions/{created['session_id']}/messages",
            json={"message_id": message_id, "message": "cotton"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_response"]["recommendations"] == [{"parent_asin": "A1"}]
    assert payload["products"]["A1"]["title"] == "Safe <script> title"
    assert payload["turn"] == 1
    assert len(agent.calls) == 1


def test_health_session_lookup_validation_and_product_errors(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get("/api/health").json()["status"] == "ready"
        session = client.post("/api/sessions").json()
        assert client.get(f"/api/sessions/{session['session_id']}").json()["next_turn"] == 1
        assert client.get(f"/api/sessions/{uuid4()}").status_code == 404
        invalid_uuid = client.get("/api/sessions/not-a-uuid")
        assert invalid_uuid.status_code == 400
        assert invalid_uuid.json()["error"]["code"] == "invalid_request"
        assert client.get("/api/products/missing").status_code == 404
        bad = client.post(
            f"/api/sessions/{session['session_id']}/messages",
            json={"message_id": str(uuid4()), "message": "   "},
        )
        assert bad.status_code == 400
        assert bad.json()["error"]["code"] == "invalid_message"


def wait_for_status(client: TestClient, expected: str) -> dict[str, object]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        payload = client.get("/api/health").json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"health never reached {expected}")


def test_background_initializer_exposes_loading_then_ready(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    gate = threading.Event()

    def initializer(path: Path) -> WebRuntime:
        assert gate.wait(timeout=2.0)
        return runtime

    app = create_app(catalog_path=tmp_path / "catalog.jsonl", initializer=initializer)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "loading"
        gate.set()
        assert wait_for_status(client, "ready")["status"] == "ready"


def test_shutdown_waits_for_blocking_initializer_worker(tmp_path: Path) -> None:
    """Cancelling only the task wrapper must not end the lifespan early."""
    runtime, _ = make_runtime(tmp_path)
    started = threading.Event()
    unblock = threading.Event()
    initializer_finished = threading.Event()
    lifespan_finished = threading.Event()

    def initializer(path: Path) -> WebRuntime:
        started.set()
        assert unblock.wait(timeout=2.0)
        initializer_finished.set()
        return runtime

    app = create_app(catalog_path=tmp_path / "catalog.jsonl", initializer=initializer)

    async def exercise_lifespan() -> None:
        async with app.router.lifespan_context(app):
            await asyncio.to_thread(started.wait, 1.0)

    def run_lifespan() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(exercise_lifespan())
            lifespan_finished.set()
            unblock.wait(timeout=1.0)
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()

    lifecycle = threading.Thread(target=run_lifespan)
    lifecycle.start()
    try:
        assert started.wait(timeout=1.0)

        assert not lifespan_finished.wait(timeout=0.1)

        unblock.set()
        assert lifespan_finished.wait(timeout=1.0)
        lifecycle.join(timeout=1.0)
        assert not lifecycle.is_alive()
        assert initializer_finished.is_set()
    finally:
        unblock.set()
        lifecycle.join(timeout=1.0)


def test_cancelled_lifespan_waits_for_blocking_initializer_worker(tmp_path: Path) -> None:
    """A cancellation must not complete the lifespan before its worker finishes."""
    runtime, _ = make_runtime(tmp_path)
    started = threading.Event()
    unblock = threading.Event()
    initializer_finished = threading.Event()

    def initializer(path: Path) -> WebRuntime:
        started.set()
        assert unblock.wait(timeout=2.0)
        initializer_finished.set()
        return runtime

    app = create_app(catalog_path=tmp_path / "catalog.jsonl", initializer=initializer)

    async def exercise_lifespan() -> None:
        async with app.router.lifespan_context(app):
            await asyncio.Event().wait()

    async def scenario() -> None:
        lifecycle = asyncio.create_task(exercise_lifespan())
        try:
            assert await asyncio.to_thread(started.wait, 1.0)
            lifecycle.cancel()
            await asyncio.sleep(0.05)
            assert not lifecycle.done()

            unblock.set()
            with pytest.raises(asyncio.CancelledError):
                await lifecycle
            assert initializer_finished.is_set()
        finally:
            unblock.set()
            if not lifecycle.done():
                with suppress(asyncio.CancelledError):
                    await lifecycle

    asyncio.run(scenario())


def test_wait_for_cancelled_initializer_task_propagates_without_busy_loop() -> None:
    script = """
import asyncio

from webapp.app import wait_for_initializer


async def scenario():
    initializer = asyncio.create_task(asyncio.sleep(60))
    initializer.cancel()
    try:
        await wait_for_initializer(initializer)
    except asyncio.CancelledError:
        return
    raise AssertionError("cancelled initializer did not propagate cancellation")


asyncio.run(scenario())
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            cwd=Path.cwd(),
            timeout=2.0,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("wait_for_initializer did not terminate within two seconds")
    assert completed.returncode == 0


def test_sparse_product_summary_is_a_valid_chat_response(tmp_path: Path) -> None:
    class SparseSessions:
        async def send_message(self, session_id, message_id, message):
            return {
                "session_id": str(session_id),
                "message_id": str(message_id),
                "turn": 1,
                "agent_response": {
                    "message": "Here is an option.",
                    "ask_attribute": None,
                    "recommendations": [{"parent_asin": "A1"}, {"parent_asin": "A2"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                "products": {
                    "A1": {},
                    "A2": {"title": "Design contract product", "price": 27.99},
                },
            }

    runtime, _ = make_runtime(tmp_path)
    sparse_runtime = WebRuntime(SparseSessions(), runtime.catalog)
    with TestClient(create_app(runtime=sparse_runtime)) as client:
        response = client.post(
            f"/api/sessions/{uuid4()}/messages",
            json={"message_id": str(uuid4()), "message": "cotton"},
        )

    assert response.status_code == 200
    assert response.json()["products"] == {
        "A1": {},
        "A2": {"title": "Design contract product", "price": 27.99},
    }


def test_initializer_failure_is_sanitized(tmp_path: Path) -> None:
    def initializer(path: Path) -> WebRuntime:
        raise RuntimeError("secret /local/path")

    app = create_app(catalog_path=tmp_path / "catalog.jsonl", initializer=initializer)
    with TestClient(app) as client:
        payload = wait_for_status(client, "failed")
    serialized = json.dumps(payload)
    assert payload["error"]["code"] == "initialization_failed"
    assert "secret" not in serialized
    assert "/local/path" not in serialized


def test_security_headers_are_present_on_error_responses(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.get("/api/products/missing")
    assert response.headers["content-security-policy"] == (
        "default-src 'self'; img-src 'none'; object-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_malformed_adapter_envelope_is_sanitized_and_secured(tmp_path: Path) -> None:
    class MalformedSessions:
        async def send_message(self, session_id, message_id, message):
            return {
                "session_id": str(session_id),
                "message_id": str(message_id),
                "turn": 1,
                "agent_response": "secret malformed adapter envelope",
                "products": {},
            }

    runtime, _ = make_runtime(tmp_path)
    malformed_runtime = WebRuntime(MalformedSessions(), runtime.catalog)
    with TestClient(create_app(runtime=malformed_runtime), raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/sessions/{uuid4()}/messages",
            json={"message_id": str(uuid4()), "message": "cotton"},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert response.headers["content-security-policy"] == (
        "default-src 'self'; img-src 'none'; object-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_outer_framework_errors_are_sanitized_and_secured(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    app = create_app(runtime=runtime)

    @app.get("/test-unhandled-error")
    async def unhandled_error():
        raise RuntimeError("secret unhandled error")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-unhandled-error")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert response.headers["content-security-policy"] == (
        "default-src 'self'; img-src 'none'; object-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize(
    ("method", "path_template"),
    [
        ("POST", "/api/sessions"),
        ("GET", "/api/sessions/{session_id}"),
        ("POST", "/api/sessions/{session_id}/messages"),
        ("GET", "/api/products/A1"),
    ],
    ids=["create-session", "get-session", "send-message", "get-product"],
)
def test_all_runtime_routes_return_stable_503_while_loading_and_after_failure(
    tmp_path: Path, method: str, path_template: str
) -> None:
    path = path_template.format(session_id=uuid4())
    request_kwargs = (
        {"json": {"message_id": str(uuid4()), "message": "cotton"}}
        if path.endswith("/messages")
        else {}
    )
    gate = threading.Event()

    def loading_initializer(path: Path) -> WebRuntime:
        assert gate.wait(timeout=2.0)
        return make_runtime(tmp_path)[0]

    loading_app = create_app(
        catalog_path=tmp_path / "catalog.jsonl", initializer=loading_initializer
    )
    with TestClient(loading_app) as client:
        loading = client.request(method, path, **request_kwargs)
        assert loading.status_code == 503
        assert loading.json()["error"]["code"] == "service_initializing"
        gate.set()

    def failing_initializer(path: Path) -> WebRuntime:
        raise RuntimeError("secret initialization failure")

    failed_app = create_app(
        catalog_path=tmp_path / "catalog.jsonl", initializer=failing_initializer
    )
    with TestClient(failed_app) as client:
        assert wait_for_status(client, "failed")["status"] == "failed"
        failed = client.request(method, path, **request_kwargs)
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "service_failed"
