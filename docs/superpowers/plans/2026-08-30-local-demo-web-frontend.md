# Local Demo Web Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, English, GPT-style shopping chat demo that adapts the existing synchronous Agent without changing its interface, evaluator, retrieval, or decision behavior.

**Architecture:** A single FastAPI process serves a dependency-free static UI and owns one shared Agent. A SessionManager supplies server-side turns and idempotency under a global Agent lock, while a read-only CatalogPresenter enriches ASINs in a separate product map without mutating the original Agent response.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, HTTPX/TestClient, standard-library JSONL indexing, HTML5, CSS, vanilla JavaScript, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-local-demo-web-frontend-design.md`

## Global Constraints

- Do not modify `Agent.reset(session_id, user_profile)`, `Agent.respond(session_id, user_message, turn, top_k)`, `agent/`, `evaluator/`, retrieval logic, decision logic, or the official response contract.
- Keep `agent_response` as an unmodified deep copy of the Agent-returned dictionary; place display data in a separate `products` mapping.
- Use one shared Agent and serialize every `reset()` and `respond()` call with one global lock.
- The frontend sends no profile and no turn; sessions call `reset(session_id, {})`, and the server owns an unbounded incrementing turn.
- Use non-streaming whole-response requests; do not add token streaming.
- Keep all assets local: no Node.js, CDN, remote fonts, external images, analytics, or product lookups.
- Serve English copy in a light theme; do not add dark mode, localization, debug panels, history lists, authentication, databases, or server-side transcript persistence.
- Bind to `127.0.0.1` by default and do not enable CORS.
- Accept messages up to 4,000 Unicode characters, reject whitespace-only messages, and pass accepted text to Agent unchanged.
- Preserve browser sessions across refresh with `localStorage`; treat a missing server session after restart as expired and create a new session.
- Put Web-only dependencies in `requirements-web.txt`; do not change `requirements.txt`.
- Every code task follows red-green TDD and ends in its own commit.

---

## File Map

**Create:**

- `webapp/__init__.py`: package exports only.
- `webapp/__main__.py`: CLI argument parsing and Uvicorn startup.
- `webapp/app.py`: FastAPI application factory, runtime initialization, lifecycle, routes, error mapping, static serving.
- `webapp/service.py`: Agent protocol, session records, global serialization, idempotent message dispatch.
- `webapp/catalog.py`: JSONL byte-offset index and product projection.
- `webapp/schemas.py`: Pydantic request/response and stable error shapes.
- `webapp/static/index.html`: semantic page shell and accessibility landmarks.
- `webapp/static/styles.css`: light responsive chat, cards, loading and drawer styles.
- `webapp/static/app.js`: API client, local state, rendering, retry and drawer behavior.
- `tests/test_webapp_catalog.py`: catalog indexing and projection tests.
- `tests/test_webapp_service.py`: session, turn, idempotency and serialization tests.
- `tests/test_webapp_api.py`: FastAPI health, lifecycle, validation and API-contract tests.
- `tests/test_webapp_static.py`: static security and page-contract tests without Node.js.
- `requirements-web.txt`: isolated FastAPI/Uvicorn/HTTPX dependencies.

**Modify:**

- `README.md`: append local Web demo install, start and offline-use instructions.

---

### Task 1: Read-only catalog presentation index

**Files:**
- Create: `webapp/__init__.py`
- Create: `webapp/catalog.py`
- Create: `tests/test_webapp_catalog.py`

**Interfaces:**
- Consumes: a UTF-8 JSONL `Path` whose records use the official catalog fields.
- Produces: `CatalogPresenter.build(path: Path) -> CatalogPresenter`, `summaries(asins: Sequence[str]) -> dict[str, dict[str, object]]`, and `detail(parent_asin: str) -> dict[str, object] | None`.
- Raises: `CatalogError` for unreadable files, invalid JSON, missing/empty `parent_asin`, or duplicate ASINs.

- [ ] **Step 1: Write focused failing tests for indexing, summaries and details**

```python
# tests/test_webapp_catalog.py
import json
from pathlib import Path

import pytest

from webapp.catalog import CatalogError, CatalogPresenter


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_catalog_returns_order_independent_summary_mapping_and_full_detail(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    write_rows(path, [
        {
            "parent_asin": "A1",
            "title": "Café cotton shirt",
            "price": None,
            "average_rating": 4.6,
            "rating_number": 42,
            "store": "Demo",
            "categories": ["Clothing", "Men", "Shirts"],
            "features": ["Cotton", "Machine washable", "Regular fit"],
            "description": ["A lightweight shirt."],
            "details": {"Department": "mens"},
        },
        {"parent_asin": "A2", "title": "Trail jacket", "features": []},
    ])

    catalog = CatalogPresenter.build(path)
    summaries = catalog.summaries(["A2", "missing", "A1"])

    assert list(summaries) == ["A2", "A1"]
    assert summaries["A1"]["features"] == ["Cotton", "Machine washable"]
    assert summaries["A1"]["categories"] == ["Men", "Shirts"]
    assert summaries["A1"]["price"] is None
    assert catalog.detail("A1")["description"] == ["A lightweight shirt."]
    assert catalog.detail("missing") is None


@pytest.mark.parametrize(
    "rows",
    [
        [{"title": "missing id"}],
        [{"parent_asin": "A1"}, {"parent_asin": "A1"}],
    ],
)
def test_catalog_rejects_missing_and_duplicate_asins(tmp_path: Path, rows: list[dict]) -> None:
    path = tmp_path / "catalog.jsonl"
    write_rows(path, rows)
    with pytest.raises(CatalogError):
        CatalogPresenter.build(path)


def test_catalog_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        CatalogPresenter.build(path)
```

- [ ] **Step 2: Run the tests and verify the intended red state**

Run: `.conda/bin/python -m pytest tests/test_webapp_catalog.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'webapp'`.

- [ ] **Step 3: Implement byte-offset indexing and projection**

```python
# webapp/catalog.py
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


class CatalogError(ValueError):
    """The selected presentation catalog is not safe to serve."""


class CatalogPresenter:
    def __init__(self, path: Path, offsets: dict[str, int]) -> None:
        self.path = path
        self._offsets = offsets

    @classmethod
    def build(cls, path: Path) -> "CatalogPresenter":
        offsets: dict[str, int] = {}
        try:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    row = json.loads(raw.decode("utf-8"))
                    if not isinstance(row, dict):
                        raise CatalogError("catalog row must be a JSON object")
                    asin = str(row.get("parent_asin", "")).strip()
                    if not asin or asin in offsets:
                        raise CatalogError("catalog has a missing or duplicate parent_asin")
                    offsets[asin] = offset
        except CatalogError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("catalog cannot be indexed") from exc
        return cls(path, offsets)

    def summaries(self, asins: Sequence[str]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for asin in dict.fromkeys(asins):
            row = self._read(asin)
            if row is not None:
                result[asin] = self._summary(row)
        return result

    def detail(self, parent_asin: str) -> dict[str, object] | None:
        row = self._read(parent_asin)
        if row is None:
            return None
        return {
            key: row.get(key)
            for key in (
                "parent_asin", "title", "price", "average_rating", "rating_number",
                "store", "categories", "features", "description", "details",
            )
        }

    def _read(self, asin: str) -> dict[str, object] | None:
        offset = self._offsets.get(asin)
        if offset is None:
            return None
        try:
            with self.path.open("rb") as handle:
                handle.seek(offset)
                row = json.loads(handle.readline().decode("utf-8"))
                if not isinstance(row, dict):
                    raise CatalogError("indexed catalog record is not a JSON object")
                return row
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("indexed catalog record cannot be read") from exc

    @staticmethod
    def _summary(row: dict[str, object]) -> dict[str, object]:
        categories = [str(value) for value in row.get("categories") or []]
        features = [str(value) for value in row.get("features") or []]
        return {
            "parent_asin": str(row["parent_asin"]),
            "title": str(row.get("title") or "Untitled product"),
            "price": row.get("price"),
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
            "store": str(row.get("store") or ""),
            "categories": categories[-2:],
            "features": features[:2],
        }
```

Keep `webapp/__init__.py` to a package docstring; do not import the FastAPI app at package import time.

- [ ] **Step 4: Run catalog tests and project Ruff**

Run: `.conda/bin/python -m pytest tests/test_webapp_catalog.py -q`

Expected: all catalog tests pass.

Run: `.conda/bin/python -m ruff check --no-cache webapp/catalog.py tests/test_webapp_catalog.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the catalog layer**

```bash
git add webapp/__init__.py webapp/catalog.py tests/test_webapp_catalog.py
git commit -m "feat: add read-only product presentation index"
```

---

### Task 2: SessionManager and exact Agent response preservation

**Files:**
- Create: `webapp/service.py`
- Create: `tests/test_webapp_service.py`

**Interfaces:**
- Consumes: `AgentProtocol.reset(session_id: str, user_profile: dict) -> None`, `AgentProtocol.respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict`, and `CatalogProtocol.summaries(asins: Sequence[str]) -> dict[str, dict[str, object]]`.
- Produces: `SessionManager.create_session() -> SessionSnapshot`, `get_session(session_id: UUID) -> SessionSnapshot | None`, and `send_message(session_id: UUID, message_id: UUID, message: str) -> dict[str, object]`.
- Raises: `SessionNotFound` and `InvalidMessage`; no exception text crosses the API boundary.

- [ ] **Step 1: Write failing tests for reset, unbounded turns and exact responses**

```python
# tests/test_webapp_service.py
import asyncio
import copy
import threading
import time
from uuid import uuid4

import pytest

from webapp.catalog import CatalogError
from webapp.service import InvalidMessage, SessionManager, SessionNotFound


class FakeCatalog:
    def summaries(self, asins):
        return {asin: {"parent_asin": asin, "title": f"Product {asin}"} for asin in asins}


class FakeAgent:
    def __init__(self) -> None:
        self.resets = []
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.resets.append((session_id, copy.deepcopy(user_profile)))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        response = {
            "message": f"reply {turn}",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": "A1"}],
            "usage": {"prompt_tokens": turn, "completion_tokens": 1},
        }
        self.calls.append((session_id, user_message, turn, top_k))
        with self.guard:
            self.active -= 1
        return response


def test_session_uses_empty_profile_and_allows_turn_eleven() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        for turn in range(1, 12):
            result = await manager.send_message(session.session_id, uuid4(), f"message {turn}")
            assert result["turn"] == turn
        assert agent.resets == [(str(session.session_id), {})]
        assert [call[2] for call in agent.calls] == list(range(1, 12))
    asyncio.run(scenario())


def test_duplicate_message_id_returns_cached_exact_agent_response() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        message_id = uuid4()
        first = await manager.send_message(session.session_id, message_id, "cotton")
        second = await manager.send_message(session.session_id, message_id, "ignored retry body")
        assert second == first
        assert len(agent.calls) == 1
        assert first["agent_response"] == {
            "message": "reply 1",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": "A1"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    asyncio.run(scenario())


def test_agent_calls_are_global_serialized_across_sessions() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        left = await manager.create_session()
        right = await manager.create_session()
        await asyncio.gather(
            manager.send_message(left.session_id, uuid4(), "left"),
            manager.send_message(right.session_id, uuid4(), "right"),
        )
        assert agent.max_active == 1
        assert sorted(call[2] for call in agent.calls) == [1, 1]
    asyncio.run(scenario())


def test_agent_failure_does_not_advance_turn_and_valid_text_is_unchanged() -> None:
    class FailsOnceAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            if self.fail:
                self.fail = False
                raise RuntimeError("temporary failure")
            return super().respond(session_id, user_message, turn, top_k)

    async def scenario() -> None:
        agent = FailsOnceAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        with pytest.raises(RuntimeError):
            await manager.send_message(session.session_id, uuid4(), " first ")
        result = await manager.send_message(session.session_id, uuid4(), " second ")
        assert result["turn"] == 1
        assert agent.calls[0][1] == " second "
    asyncio.run(scenario())


def test_invalid_messages_and_unknown_sessions_do_not_call_agent() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        with pytest.raises(InvalidMessage):
            await manager.send_message(session.session_id, uuid4(), "   ")
        with pytest.raises(InvalidMessage):
            await manager.send_message(session.session_id, uuid4(), "x" * 4001)
        with pytest.raises(SessionNotFound):
            await manager.send_message(uuid4(), uuid4(), "valid")
        assert agent.calls == []
    asyncio.run(scenario())


def test_catalog_failure_after_agent_response_is_cached_without_reinvocation() -> None:
    class FailingCatalog:
        def summaries(self, asins):
            raise CatalogError("presentation file changed")

    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FailingCatalog(), top_k=10)
        session = await manager.create_session()
        message_id = uuid4()
        first = await manager.send_message(session.session_id, message_id, "cotton")
        retry = await manager.send_message(session.session_id, message_id, "cotton")
        assert first == retry
        assert first["products"] == {}
        assert manager.get_session(session.session_id).next_turn == 2
        assert len(agent.calls) == 1
    asyncio.run(scenario())
```

- [ ] **Step 2: Run the service tests and verify they fail for the missing module**

Run: `.conda/bin/python -m pytest tests/test_webapp_service.py -q`

Expected: collection fails because `webapp.service` does not exist.

- [ ] **Step 3: Implement the session records, locks and idempotency cache**

```python
# webapp/service.py
from __future__ import annotations

import asyncio
import copy
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID, uuid4

from webapp.catalog import CatalogError

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    def reset(self, session_id: str, user_profile: dict) -> None:
        raise NotImplementedError

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        raise NotImplementedError


class CatalogProtocol(Protocol):
    def summaries(self, asins: Sequence[str]) -> dict[str, dict[str, object]]:
        raise NotImplementedError


class SessionNotFound(LookupError):
    pass


class InvalidMessage(ValueError):
    pass


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: UUID
    next_turn: int


@dataclass
class _SessionRecord:
    session_id: UUID
    next_turn: int = 1
    responses: OrderedDict[UUID, dict[str, object]] = field(default_factory=OrderedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    def __init__(self, agent: AgentProtocol, catalog: CatalogProtocol, *, top_k: int) -> None:
        self._agent = agent
        self._catalog = catalog
        self._top_k = top_k
        self._agent_lock = asyncio.Lock()
        self._sessions: dict[UUID, _SessionRecord] = {}

    async def create_session(self) -> SessionSnapshot:
        session_id = uuid4()
        async with self._agent_lock:
            await asyncio.to_thread(self._agent.reset, str(session_id), {})
        self._sessions[session_id] = _SessionRecord(session_id=session_id)
        return SessionSnapshot(session_id, 1)

    def get_session(self, session_id: UUID) -> SessionSnapshot | None:
        record = self._sessions.get(session_id)
        return None if record is None else SessionSnapshot(record.session_id, record.next_turn)

    async def send_message(self, session_id: UUID, message_id: UUID, message: str) -> dict[str, object]:
        if not message.strip() or len(message) > 4_000:
            raise InvalidMessage("message must contain 1 to 4000 non-whitespace characters")
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFound("session not found")
        async with record.lock:
            cached = record.responses.get(message_id)
            if cached is not None:
                return copy.deepcopy(cached)
            turn = record.next_turn
            async with self._agent_lock:
                raw = await asyncio.to_thread(
                    self._agent.respond, str(session_id), message, turn, self._top_k
                )
            if not isinstance(raw, dict):
                raise TypeError("Agent.respond() must return a dictionary")
            agent_response = copy.deepcopy(raw)
            asins = [
                str(item.get("parent_asin", ""))
                for item in agent_response.get("recommendations", [])
                if isinstance(item, dict) and item.get("parent_asin")
            ]
            try:
                products = self._catalog.summaries(asins)
            except CatalogError:
                logger.exception("catalog presentation enrichment failed")
                products = {}
            envelope = {
                "session_id": str(session_id),
                "message_id": str(message_id),
                "turn": turn,
                "agent_response": agent_response,
                "products": products,
            }
            record.responses[message_id] = copy.deepcopy(envelope)
            while len(record.responses) > 128:
                record.responses.popitem(last=False)
            record.next_turn += 1
            record.last_accessed_at = datetime.now(timezone.utc)
            return envelope
```

Use one private session lock to prevent two messages in the same session from consuming the same turn. If Agent raises before returning, leave `next_turn` unchanged. Once Agent returns, cache and increment even if product enrichment fails, because Agent state may already be committed.

- [ ] **Step 4: Run service and catalog tests**

Run: `.conda/bin/python -m pytest tests/test_webapp_service.py tests/test_webapp_catalog.py -q`

Expected: all tests pass, including turn 11 and global serialization.

Run: `.conda/bin/python -m ruff check --no-cache webapp tests/test_webapp_service.py tests/test_webapp_catalog.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the service layer**

```bash
git add webapp/service.py tests/test_webapp_service.py
git commit -m "feat: adapt agent sessions for local web chat"
```

---

### Task 3: FastAPI schemas, lifecycle and routes

**Files:**
- Create: `requirements-web.txt`
- Create: `webapp/schemas.py`
- Create: `webapp/app.py`
- Create: `tests/test_webapp_api.py`

**Interfaces:**
- Consumes: `SessionManager`, `CatalogPresenter`, and an injected `initializer(path: Path) -> WebRuntime`.
- Produces: `WebRuntime`, `RuntimeContainer`, `create_app(catalog_path: Path, initializer: Initializer | None = None, runtime: WebRuntime | None = None) -> FastAPI`.
- API: `/api/health`, `/api/sessions`, `/api/sessions/{uuid}`, `/api/sessions/{uuid}/messages`, `/api/products/{asin}`.

- [ ] **Step 1: Add and install isolated Web dependencies**

```text
# requirements-web.txt
fastapi>=0.115,<1
uvicorn>=0.30,<1
httpx>=0.27,<1
```

Run: `.conda/bin/python -m pip install -r requirements-web.txt`

Expected: FastAPI, Uvicorn and HTTPX install without changing `requirements.txt`.

- [ ] **Step 2: Write failing API-contract tests with a prebuilt runtime**

```python
# tests/test_webapp_api.py
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from webapp.app import WebRuntime, create_app
from webapp.catalog import CatalogPresenter
from webapp.service import SessionManager

from tests.test_webapp_catalog import write_rows
from tests.test_webapp_service import FakeAgent


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
```

- [ ] **Step 3: Run API tests and verify the missing-app failure**

Run: `.conda/bin/python -m pytest tests/test_webapp_api.py -q`

Expected: collection fails because `webapp.app` does not exist.

- [ ] **Step 4: Implement Pydantic schemas and stable errors**

```python
# webapp/schemas.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    message_id: UUID
    message: str = Field(max_length=4_000)

    @field_validator("message")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class SessionResponse(BaseModel):
    session_id: UUID
    next_turn: int


class ChatResponse(BaseModel):
    session_id: UUID
    message_id: UUID
    turn: int
    agent_response: dict[str, Any]
    products: dict[str, dict[str, Any]]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
```

Define `error_response(status: int, code: str, message: str) -> JSONResponse` in `app.py`; route exceptions must map to fixed public messages rather than `str(exc)`.

Register a `RequestValidationError` handler. If any error location ends in `message`, return 400 `invalid_message`; otherwise return 400 `invalid_request`. Never expose Pydantic's rejected input value:

```python
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    message_error = any(error.get("loc", ())[-1:] == ("message",) for error in exc.errors())
    if message_error:
        return error_response(400, "invalid_message", "Message must contain 1 to 4000 characters.")
    return error_response(400, "invalid_request", "Request body is invalid.")
```

- [ ] **Step 5: Implement the runtime container, background lifecycle and routes**

```python
# webapp/app.py core types
@dataclass(frozen=True)
class WebRuntime:
    sessions: SessionManager
    catalog: CatalogPresenter


@dataclass
class RuntimeContainer:
    status: Literal["loading", "ready", "failed"] = "loading"
    runtime: WebRuntime | None = None
    error_code: str | None = None
```

`create_app()` must satisfy these exact lifecycle rules:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if runtime is not None:
        container.runtime = runtime
        container.status = "ready"
        yield
        return

    async def initialize() -> None:
        try:
            container.runtime = await asyncio.to_thread(active_initializer, catalog_path)
        except Exception:
            logger.exception("web runtime initialization failed")
            container.error_code = "initialization_failed"
            container.status = "failed"
        else:
            container.status = "ready"

    task = asyncio.create_task(initialize())
    yield
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task
```

Routes call a private `require_runtime()` that returns 503 `service_initializing` for loading and 503 `service_failed` for failed. Catch `SessionNotFound`, `InvalidMessage`, `CatalogError`, and unexpected adapter errors separately. Unexpected errors log server-side and return `internal_error` without exception text.

Add security headers to every response with middleware:

```python
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; img-src 'none'; object-src 'none'; frame-ancestors 'none'"
)
response.headers["X-Content-Type-Options"] = "nosniff"
response.headers["Referrer-Policy"] = "no-referrer"
```

- [ ] **Step 6: Add lifecycle tests for loading and sanitized failure**

```python
import json
import threading
import time


def wait_for_status(client: TestClient, expected: str) -> dict:
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

    with TestClient(create_app(catalog_path=tmp_path / "catalog.jsonl", initializer=initializer)) as client:
        assert client.get("/api/health").json()["status"] == "loading"
        gate.set()
        assert wait_for_status(client, "ready")["status"] == "ready"


def test_initializer_failure_is_sanitized(tmp_path: Path) -> None:
    def initializer(path: Path) -> WebRuntime:
        raise RuntimeError("secret /local/path")

    with TestClient(create_app(catalog_path=tmp_path / "catalog.jsonl", initializer=initializer)) as client:
        payload = wait_for_status(client, "failed")
    serialized = json.dumps(payload)
    assert payload["error"]["code"] == "initialization_failed"
    assert "secret" not in serialized
    assert "/local/path" not in serialized
```

Run: `.conda/bin/python -m pytest tests/test_webapp_api.py -q`

Expected: all API and lifecycle tests pass.

- [ ] **Step 7: Run focused quality checks and commit**

Run: `.conda/bin/python -m ruff check --no-cache webapp tests/test_webapp_api.py`

Expected: `All checks passed!`

```bash
git add requirements-web.txt webapp/app.py webapp/schemas.py tests/test_webapp_api.py
git commit -m "feat: expose local shopping chat API"
```

---

### Task 4: Production runtime initialization and one-command CLI

**Files:**
- Modify: `webapp/app.py`
- Create: `webapp/__main__.py`
- Modify: `tests/test_webapp_api.py`

**Interfaces:**
- Consumes: existing `EnvConfig.from_env()`, `verify_file()`, `create_llm_client()`, and `Agent`.
- Produces: `initialize_runtime(catalog_path: Path, *, env_loader, verifier, agent_factory) -> WebRuntime`, `parse_args(argv: Sequence[str] | None) -> argparse.Namespace`, and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing tests that bind validation and Agent to the selected catalog**

```python
def test_initialize_runtime_validates_selected_catalog_and_disables_duplicate_agent_check(tmp_path):
    selected = tmp_path / "selected.jsonl"
    write_rows(selected, [{"parent_asin": "A1", "title": "One"}])
    verified = []
    captured = {}

    def verifier(path, expected, label, skip=False):
        verified.append((Path(path), expected, label, skip))
        return True

    def env_loader():
        return EnvConfig.from_env(
            overrides={"skip_data_verify": False, "llm": {"provider": "none"}},
            environ={},
        )

    def agent_factory(path, env, llm_client):
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
    assert captured["env"].skip_data_verify is True
    assert runtime.catalog.detail("A1")["title"] == "One"
```

```python
def test_initialize_runtime_honors_existing_skip_flag(tmp_path):
    selected = tmp_path / "selected.jsonl"
    write_rows(selected, [{"parent_asin": "A1", "title": "One"}])
    verifier_calls = []

    def env_loader():
        return EnvConfig.from_env(
            overrides={"skip_data_verify": True, "llm": {"provider": "none"}},
            environ={},
        )

    def verifier(path, expected, label, skip=False):
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
```

- [ ] **Step 2: Verify the initializer tests fail before implementation**

Run: `.conda/bin/python -m pytest tests/test_webapp_api.py -k initialize_runtime -q`

Expected: FAIL because `initialize_runtime` is not defined.

- [ ] **Step 3: Implement production initialization without changing core configuration**

```python
def initialize_runtime(
    catalog_path: Path,
    *,
    env_loader=EnvConfig.from_env,
    verifier=verify_file,
    agent_factory=Agent,
) -> WebRuntime:
    env = env_loader()
    if not env.skip_data_verify:
        verifier(
            catalog_path,
            constants.EXPECTED_SHA256_CATALOG,
            "catalog.jsonl",
            skip=False,
        )
    catalog = CatalogPresenter.build(catalog_path)
    web_env = EnvConfig(replace(env.app_config, skip_data_verify=True))
    llm_client = create_llm_client(web_env.llm)
    llm_client.initialize()
    agent = agent_factory(catalog_path, web_env, llm_client)
    return WebRuntime(SessionManager(agent, catalog, top_k=web_env.top_k), catalog)
```

- [ ] **Step 4: Write failing CLI parsing tests**

```python
from webapp.__main__ import parse_args


def test_web_cli_defaults_and_overrides():
    defaults = parse_args([])
    assert defaults.catalog == Path("data/catalog.jsonl")
    assert defaults.host == "127.0.0.1"
    assert defaults.port == 8000

    custom = parse_args(["--catalog", "/tmp/catalog.jsonl", "--port", "8080"])
    assert custom.catalog == Path("/tmp/catalog.jsonl")
    assert custom.host == "127.0.0.1"
    assert custom.port == 8080
```

- [ ] **Step 5: Implement CLI and Uvicorn startup**

`parse_args()` accepts `--catalog`, `--host`, and `--port`. Restrict `port` to 1–65535. `main()` calls `create_app(catalog_path=args.catalog)` and then:

```python
uvicorn.run(app, host=args.host, port=args.port, log_level="info")
return 0
```

Do not set `reload=True`, do not open a browser automatically, and do not download files.

- [ ] **Step 6: Run API/CLI tests and commit**

Run: `.conda/bin/python -m pytest tests/test_webapp_api.py -q`

Expected: all tests pass.

Run: `.conda/bin/python -m ruff check --no-cache webapp tests/test_webapp_api.py`

Expected: `All checks passed!`

```bash
git add webapp/app.py webapp/__main__.py tests/test_webapp_api.py
git commit -m "feat: initialize and launch local web demo"
```

---

### Task 5: Static chat shell, API client and refresh recovery

**Files:**
- Create: `webapp/static/index.html`
- Create: `webapp/static/styles.css`
- Create: `webapp/static/app.js`
- Create: `tests/test_webapp_static.py`
- Modify: `webapp/app.py`

**Interfaces:**
- Consumes: the five API routes from Task 3.
- Produces: same-origin `/`, `/assets/styles.css`, `/assets/app.js`, and browser functions `bootstrap`, `newChat`, `sendMessage`, `retryMessage`, `persistState`, and `restoreState`.

- [ ] **Step 1: Write failing static-contract and security tests**

```python
# tests/test_webapp_static.py
from pathlib import Path


STATIC = Path("webapp/static")


def test_static_page_has_required_landmarks_and_local_assets() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'lang="en"' in html
    assert 'id="new-chat"' in html
    assert 'id="conversation"' in html
    assert 'id="message-input"' in html
    assert 'id="send-message"' in html
    assert 'id="product-drawer"' in html
    assert 'href="/assets/styles.css"' in html
    assert 'src="/assets/app.js"' in html
    assert "https://" not in html and "http://" not in html


def test_dynamic_javascript_never_uses_html_injection() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert "eval(" not in script
    assert "localStorage" in script
    assert "crypto.randomUUID()" in script
```

- [ ] **Step 2: Run static tests and verify missing-file failures**

Run: `.conda/bin/python -m pytest tests/test_webapp_static.py -q`

Expected: failures because the static files do not exist.

- [ ] **Step 3: Build the semantic HTML shell**

`index.html` must include:

```html
<aside class="sidebar" aria-label="Application">
  <div class="brand">Shopping Copilot</div>
  <p>Your local conversational product finder.</p>
  <button id="new-chat" type="button">New chat</button>
</aside>
<main class="app-main">
  <header><h1>Shopping Copilot</h1><span id="service-status">Local · Loading</span></header>
  <section id="welcome" aria-labelledby="welcome-title">
    <h2 id="welcome-title">What are you shopping for?</h2>
    <div id="prompt-examples"></div>
  </section>
  <section id="conversation" aria-live="polite"></section>
  <p id="status-notice" role="status"></p>
  <form id="composer">
    <label class="sr-only" for="message-input">Shopping request</label>
    <textarea id="message-input" maxlength="4000" rows="1"></textarea>
    <button id="send-message" type="submit">Send</button>
  </form>
</main>
<div id="drawer-backdrop" hidden></div>
<aside id="product-drawer" hidden aria-hidden="true" aria-labelledby="drawer-title">
  <button id="drawer-close" type="button">Close</button>
  <h2 id="drawer-title" tabindex="-1">Product details</h2>
  <div id="drawer-body"></div>
</aside>
```

Add the three exact example prompts from the spec as JavaScript constants; clicking sets `messageInput.value` and focuses the textarea without submitting.

Submit on Enter and preserve a newline on Shift+Enter:

```javascript
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
```

- [ ] **Step 4: Implement local state and same-origin API calls**

Use one versioned key, `shopping-copilot-web:v1`. State shape and persistence:

```javascript
const STORAGE_KEY = "shopping-copilot-web:v1";
const initialState = {
  sessionId: null,
  messages: [],
  pending: false,
  updatedAt: null,
};
const emptyState = () => ({...initialState, messages: []});
let state = emptyState();

function persistState() {
  state.updatedAt = new Date().toISOString();
  localStorage.setItem(STORAGE_KEY, JSON.stringify({version: 1, ...state}));
}
```

Required behaviors:

```javascript
class ApiError extends Error {
  constructor(status, code) {
    super(code);
    this.status = status;
    this.code = code;
  }
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(response.status, payload.error?.code || "request_failed");
  return payload;
}

async function submitExistingMessage(text, messageId) {
  state.pending = true;
  const userMessage = state.messages.find((item) => item.messageId === messageId);
  userMessage.status = "pending";
  persistState();
  renderConversation();
  try {
    const payload = await apiRequest(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({message_id: messageId, message: text}),
    });
    state.messages.find((item) => item.messageId === messageId).status = "sent";
    state.messages.push({role: "assistant", payload});
  } catch (error) {
    if (error instanceof ApiError && error.code === "session_not_found") {
      await replaceExpiredSession();
    } else {
      state.messages.find((item) => item.messageId === messageId).status = "failed";
    }
  } finally {
    state.pending = false;
    persistState();
    renderConversation();
  }
}

async function sendMessage(text) {
  const messageId = crypto.randomUUID();
  state.messages.push({role: "user", text, messageId, status: "pending"});
  return submitExistingMessage(text, messageId);
}
```

`retryMessage()` must reuse both the original text and original `messageId`; it must not append a second user message. `newChat()` posts `/api/sessions`, clears messages, persists and focuses the composer.

Use one explicit expired-session path so a server restart never replays stale history:

```javascript
async function replaceExpiredSession() {
  state = emptyState();
  const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
  state.sessionId = created.session_id;
  showNotice("The local service restarted. Starting a new chat.");
}
```

`showNotice()` writes to a dedicated live-status element with `textContent`; it does not append the old transcript to the new Agent session.

- [ ] **Step 5: Implement bootstrap and refresh recovery**

`bootstrap()` must:

1. Render `Local · Loading` and poll `/api/health` at 500 ms intervals while status is loading.
2. Show a fixed initialization error page for failed status.
3. Restore versioned local state.
4. If a stored session exists, call `GET /api/sessions/{id}`.
5. On `session_not_found`, clear state and create a new session.
6. Render `Local · Ready`, welcome or conversation, then enable the composer.

Do not restore a `pending` state after refresh; convert any pending user message to `failed` so Retry is explicit.

Implement restoration defensively; malformed or stale storage must fall back to a clean state rather than blocking startup:

```javascript
function restoreState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!saved || saved.version !== 1 || !Array.isArray(saved.messages)) return emptyState();
    return {
      sessionId: typeof saved.sessionId === "string" ? saved.sessionId : null,
      messages: saved.messages.map((message) => (
        message.role === "user" && message.status === "pending"
          ? {...message, status: "failed"}
          : message
      )),
      pending: false,
      updatedAt: typeof saved.updatedAt === "string" ? saved.updatedAt : null,
    };
  } catch (error) {
    localStorage.removeItem(STORAGE_KEY);
    return emptyState();
  }
}
```

- [ ] **Step 6: Add the light responsive layout**

CSS must implement a 224px sidebar, centered conversation max-width 800px, sticky composer, visible focus rings, `.sr-only`, user rounded blocks, open assistant rows, and one-column layout below 760px. Use system fonts and CSS custom properties; include no URLs. Start from these exact layout rules and add only selectors used by the HTML:

```css
:root {
  color-scheme: light;
  --surface: #ffffff;
  --surface-muted: #f5f6f8;
  --border: #e3e5e8;
  --text: #202124;
  --text-muted: #676b73;
  --accent: #315c4c;
}

* { box-sizing: border-box; }
body { margin: 0; color: var(--text); background: var(--surface); font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
.sidebar { position: fixed; inset: 0 auto 0 0; width: 224px; padding: 24px 16px; background: var(--surface-muted); border-right: 1px solid var(--border); }
.app-main { min-height: 100vh; margin-left: 224px; }
.app-main > header, #welcome, #conversation, #composer { width: min(800px, calc(100% - 32px)); margin-inline: auto; }
#composer { position: sticky; bottom: 0; display: flex; gap: 8px; padding: 16px 0 24px; background: var(--surface); }
:focus-visible { outline: 3px solid color-mix(in srgb, var(--accent) 45%, white); outline-offset: 2px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }

@media (max-width: 760px) {
  .sidebar { position: static; width: auto; border-right: 0; border-bottom: 1px solid var(--border); }
  .app-main { margin-left: 0; }
}
```

- [ ] **Step 7: Serve static files and verify endpoints**

Mount `webapp/static` at `/assets` and serve `index.html` from `/`. Static serving must work even while runtime status is loading or failed.

Run: `.conda/bin/python -m pytest tests/test_webapp_static.py tests/test_webapp_api.py -q`

Expected: all static and API tests pass.

- [ ] **Step 8: Commit the chat shell**

```bash
git add webapp/static webapp/app.py tests/test_webapp_static.py
git commit -m "feat: add local conversational shopping interface"
```

---

### Task 6: Product cards, detail drawer and complete error UX

**Files:**
- Modify: `webapp/static/index.html`
- Modify: `webapp/static/styles.css`
- Modify: `webapp/static/app.js`
- Modify: `tests/test_webapp_static.py`

**Interfaces:**
- Consumes: `agent_response.recommendations`, separate `products`, and `GET /api/products/{asin}`.
- Produces: `renderProducts(payload)`, `openProductDrawer(asin)`, `closeProductDrawer()`, Retry controls, empty-recommendation copy, and accessible drawer focus behavior.

- [ ] **Step 1: Extend failing static tests for product rendering and drawer controls**

```python
def test_product_drawer_and_ordered_renderer_contract() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="drawer-close"' in html
    assert 'id="drawer-title"' in html
    assert "function renderProducts" in script
    assert "function openProductDrawer" in script
    assert "function closeProductDrawer" in script
    assert "function retryMessage" in script
    assert "payload.agent_response.recommendations" in script
    assert "Object.values(payload.products)" not in script
    assert "innerHTML" not in script
```

Run: `.conda/bin/python -m pytest tests/test_webapp_static.py -q`

Expected: FAIL for the missing product/drawer behavior.

- [ ] **Step 2: Render cards strictly in Agent order**

For each `recommendation` in `payload.agent_response.recommendations`, read `payload.products[recommendation.parent_asin]`. Build elements only with `document.createElement()` and `textContent`.

```javascript
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderProducts(payload) {
  const grid = element("div", "product-grid");
  payload.agent_response.recommendations.forEach((recommendation, index) => {
    const asin = recommendation.parent_asin;
    const product = payload.products[asin];
    const card = element("article", "product-card");
    card.append(element("div", "product-visual", product?.categories?.at(-1) || "Product"));
    card.append(element("span", "product-rank", `#${index + 1}`));
    card.append(element("h3", "product-title", product?.title || "Product details unavailable"));
    card.append(element("p", "product-asin", asin));
    const numericPrice = Number(product?.price);
    card.append(element(
      "p",
      "product-price",
      product?.price !== null && product?.price !== undefined && Number.isFinite(numericPrice)
        ? `$${numericPrice.toFixed(2)}`
        : "Price unavailable",
    ));
    if (product?.average_rating !== null && product?.average_rating !== undefined) {
      const count = product.rating_number ? ` (${product.rating_number})` : "";
      card.append(element("p", "product-rating", `${product.average_rating} stars${count}`));
    }
    if (product?.store) card.append(element("p", "product-store", product.store));
    const badges = element("div", "category-badges");
    (product?.categories || []).slice(0, 2).forEach((category) => {
      badges.append(element("span", "category-badge", category));
    });
    card.append(badges);
    const features = element("ul", "product-features");
    (product?.features || []).slice(0, 2).forEach((feature) => {
      features.append(element("li", "", feature));
    });
    card.append(features);
    const button = element("button", "product-details-button", "View details");
    button.type = "button";
    button.dataset.asin = asin;
    button.addEventListener("click", () => openProductDrawer(asin));
    card.append(button);
    grid.append(card);
  });
  return grid;
}
```

Each card must render:

- one-based recommendation number;
- title or `Product details unavailable`;
- formatted `$12.34` or `Price unavailable`;
- rating plus count when present;
- store when present;
- up to two category badges;
- up to two feature lines;
- a `View details` button carrying the ASIN in `dataset.asin`.

Unknown products remain in their original position and show the ASIN; never remove or reorder the recommendation.

- [ ] **Step 3: Implement detail loading and accessible drawer behavior**

`openProductDrawer(asin)` fetches `/api/products/${encodeURIComponent(asin)}`, renders fields through `textContent`, saves the previously focused element, shows the backdrop, sets `aria-hidden="false"`, and focuses the close button. `closeProductDrawer()` reverses state and restores focus.

```javascript
const drawer = document.getElementById("product-drawer");
const backdrop = document.getElementById("drawer-backdrop");
const drawerTitle = document.getElementById("drawer-title");
const drawerBody = document.getElementById("drawer-body");
const drawerClose = document.getElementById("drawer-close");
let focusBeforeDrawer = null;

function appendDetailList(label, values) {
  if (!Array.isArray(values) || values.length === 0) return;
  drawerBody.append(element("h3", "", label));
  const list = element("ul", "detail-list");
  values.forEach((value) => list.append(element("li", "", String(value))));
  drawerBody.append(list);
}

function renderProductDetail(product) {
  drawerBody.replaceChildren();
  drawerTitle.textContent = product.title || "Product details";
  drawerBody.append(element("p", "product-asin", product.parent_asin));
  appendDetailList("Categories", product.categories);
  appendDetailList("Features", product.features);
  appendDetailList("Description", product.description);
  if (product.details && typeof product.details === "object") {
    drawerBody.append(element("h3", "", "Specifications"));
    const list = element("dl", "detail-pairs");
    Object.entries(product.details).forEach(([key, value]) => {
      list.append(element("dt", "", key));
      list.append(element("dd", "", typeof value === "string" ? value : JSON.stringify(value)));
    });
    drawerBody.append(list);
  }
}

async function openProductDrawer(asin) {
  focusBeforeDrawer = document.activeElement;
  drawer.hidden = false;
  backdrop.hidden = false;
  drawer.setAttribute("aria-hidden", "false");
  drawerTitle.textContent = "Loading product details…";
  drawerClose.focus();
  try {
    const product = await apiRequest(`/api/products/${encodeURIComponent(asin)}`);
    renderProductDetail(product);
  } catch (error) {
    drawerTitle.textContent = "Product details unavailable";
    drawerBody.textContent = "Product details are unavailable for this recommendation.";
  }
}

function closeProductDrawer() {
  drawer.hidden = true;
  backdrop.hidden = true;
  drawer.setAttribute("aria-hidden", "true");
  if (focusBeforeDrawer instanceof HTMLElement) focusBeforeDrawer.focus();
}

drawerClose.addEventListener("click", closeProductDrawer);
backdrop.addEventListener("click", closeProductDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !drawer.hidden) closeProductDrawer();
});
```

Wire close button, backdrop click and Escape. The drawer shows complete categories, features, descriptions, details key/value pairs and ASIN. A 404 renders `Product details are unavailable for this recommendation.` inside the drawer.

- [ ] **Step 4: Complete request, empty and session-expired UX**

- While pending, disable textarea/send/new-chat and show `Understanding your request and searching products...` in an assistant loading row.
- For failed messages, show a Retry button that passes the original `messageId`.
- For zero recommendations, render `Tell me another preference and I’ll refine the search.` beneath the assistant message.
- For `session_not_found`, show `The local service restarted. Starting a new chat.` and create a fresh session; do not replay old messages into Agent.
- For all other errors, show `Something went wrong. Please retry this message.` without raw server text.

Implement retry without adding a second user row:

```javascript
function retryMessage(messageId) {
  const failed = state.messages.find(
    (item) => item.role === "user" && item.messageId === messageId,
  );
  if (!failed || state.pending) return;
  failed.status = "pending";
  return submitExistingMessage(failed.text, failed.messageId);
}
```

- [ ] **Step 5: Add product card and drawer styles**

Use two equal columns above 760px and one below. Cards use a neutral category color block, subtle border and hover/focus state. Drawer is 420px max width, full height, scrollable, and full width on narrow screens. Respect `prefers-reduced-motion` by disabling transitions.

```css
.product-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.product-card { padding: 16px; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); }
#product-drawer { position: fixed; inset: 0 0 0 auto; z-index: 20; width: min(420px, 100vw); overflow-y: auto; padding: 24px; background: var(--surface); box-shadow: -12px 0 36px rgb(0 0 0 / 12%); }
#drawer-backdrop { position: fixed; inset: 0; z-index: 10; background: rgb(0 0 0 / 28%); }
@media (max-width: 760px) { .product-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; } }
```

- [ ] **Step 6: Run static/API tests and a no-external-assets scan**

Run: `.conda/bin/python -m pytest tests/test_webapp_static.py tests/test_webapp_api.py -q`

Expected: all tests pass.

Run: `rg -n "https?://|innerHTML|insertAdjacentHTML|eval\(" webapp/static`

Expected: no matches.

- [ ] **Step 7: Commit product presentation UX**

```bash
git add webapp/static tests/test_webapp_static.py
git commit -m "feat: present recommended product details"
```

---

### Task 7: Documentation, real startup smoke test and full regression

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reproducible install/start instructions and final verification evidence.

- [ ] **Step 1: Append local Web demo instructions to README**

Add a `Local Web Demo` section containing exactly these commands and behavior notes:

```bash
pip install -r requirements-web.txt
LLM_PROVIDER=none python -m webapp
# custom participant-kit catalog:
LLM_PROVIDER=none python -m webapp --catalog /path/to/catalog.jsonl --port 8080
```

Document `http://127.0.0.1:8000`, local-only binding, no external assets, non-streaming responses, browser-only refresh persistence, service-restart reset, no frontend turn cap, and automatic existing-model fallback.

- [ ] **Step 2: Run all focused Web tests**

Run:

```bash
.conda/bin/python -m pytest \
  tests/test_webapp_catalog.py \
  tests/test_webapp_service.py \
  tests/test_webapp_api.py \
  tests/test_webapp_static.py -q
```

Expected: all Web tests pass.

- [ ] **Step 3: Run the complete existing and new test suite**

Run: `.conda/bin/python -m pytest -q`

Expected: zero failures; only pre-existing optional-model/online skips are allowed.

- [ ] **Step 4: Run static quality and repository-integrity checks**

Run: `.conda/bin/python -m ruff check --no-cache .`

Expected: `All checks passed!`

Run: `git diff --check`

Expected: no output and exit code 0.

Run: `git diff --name-only origin/main...HEAD -- evaluator agent`

Expected: no output; the frontend work must not change `evaluator/` or `agent/`.

- [ ] **Step 5: Start the real offline app with the official catalog**

Run in one terminal:

```bash
LLM_PROVIDER=none .conda/bin/python -m webapp --catalog /path/to/participant-kit/catalog.jsonl
```

Run in another terminal after the static page appears:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s -X POST http://127.0.0.1:8000/api/sessions
```

Expected: the health JSON has `status` equal to `ready`, and session creation returns a UUID with `next_turn` equal to `1`.

- [ ] **Step 6: Perform browser acceptance against the checklist**

Verify welcome/examples, New chat, Enter/Shift+Enter, non-streaming loading row, ordered product cards, missing-price copy, detail drawer, Escape/overlay close, refresh recovery, failed-message Retry, service-restart reset, turn 11, narrow-screen single-column layout and keyboard focus restoration.

- [ ] **Step 7: Commit documentation after all verification passes**

```bash
git add README.md
git commit -m "docs: explain local web demo workflow"
```

- [ ] **Step 8: Record final verification state**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: clean tracked worktree and one focused commit per task. Do not push or merge unless the user separately authorizes it.
