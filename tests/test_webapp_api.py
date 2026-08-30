import json
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from tests.test_webapp_catalog import write_rows
from tests.test_webapp_service import FakeAgent
from webapp.app import WebRuntime, create_app
from webapp.catalog import CatalogPresenter
from webapp.service import SessionManager


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


def test_runtime_routes_return_stable_503_while_loading_and_after_failure(tmp_path: Path) -> None:
    gate = threading.Event()

    def loading_initializer(path: Path) -> WebRuntime:
        assert gate.wait(timeout=2.0)
        return make_runtime(tmp_path)[0]

    loading_app = create_app(
        catalog_path=tmp_path / "catalog.jsonl", initializer=loading_initializer
    )
    with TestClient(loading_app) as client:
        loading = client.post("/api/sessions")
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
        failed = client.post("/api/sessions")
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "service_failed"
