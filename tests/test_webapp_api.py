import asyncio
import json
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
                    "recommendations": [{"parent_asin": "A1"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                "products": {"A1": {"parent_asin": "A1", "title": "Sparse product"}},
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
        "A1": {"parent_asin": "A1", "title": "Sparse product"}
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
