import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_webapp_catalog import write_rows
from tests.test_webapp_service import FakeAgent, FakeCatalog
from webapp.app import WebRuntime, create_app
from webapp.catalog import CatalogPresenter
from webapp.metrics import UsageRecorder, estimate_cost_usd
from webapp.service import SessionManager


def test_estimate_cost_usd_known_and_unknown_models() -> None:
    assert estimate_cost_usd("deepseek", "deepseek-chat", 1_000_000, 0) == 0.27
    assert estimate_cost_usd("openai", "gpt-4o-mini", 0, 1_000_000) == 0.60
    assert estimate_cost_usd("unknown", "model", 1_000_000, 1_000_000) == 0.0
    assert estimate_cost_usd("deepseek", "deepseek-chat", 0, 0) == 0.0


def test_usage_recorder_summary_and_recent() -> None:
    recorder = UsageRecorder()
    recorder.record(
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cost_usd": 0.00005,
            "online": True,
        }
    )
    recorder.record(
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "cost_usd": 0.00010,
            "online": True,
        }
    )
    recorder.record(
        {
            "provider": "none",
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "online": False,
        }
    )

    summary = recorder.summary()
    assert summary["total_turns"] == 3
    assert summary["online_turns"] == 2
    assert summary["offline_turns"] == 1
    assert summary["total_prompt_tokens"] == 300
    assert summary["total_completion_tokens"] == 60
    assert summary["total_tokens"] == 360
    assert summary["total_cost_usd"] == 0.00015
    assert summary["per_provider"][0]["provider"] == "deepseek"
    assert summary["per_provider"][0]["turns"] == 2

    recent = recorder.recent(limit=3)
    assert len(recent) == 3
    assert recent[0]["prompt_tokens"] == 0  # most recent first (third record)
    assert recent[1]["prompt_tokens"] == 200
    assert recent[2]["prompt_tokens"] == 100


def test_usage_recorder_jsonl_persistence(tmp_path: Path) -> None:
    log = tmp_path / "usage.jsonl"
    recorder = UsageRecorder(log_path=log)
    recorder.record(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cost_usd": 0.000001,
            "online": True,
        }
    )
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["provider"] == "openai"
    assert "ts" in event


def test_session_manager_records_usage_event(tmp_path: Path) -> None:
    async def scenario() -> None:
        recorder = UsageRecorder()
        agent = FakeAgent()
        manager = SessionManager(
            agent,
            FakeCatalog(),
            top_k=10,
            usage_recorder=recorder,
            usage_context={
                "provider": "deepseek",
                "model": "deepseek-chat",
                "retrieval_backend": "auto",
                "rerank_backend": "none",
                "output_strategy": "holdback",
            },
        )
        session = await manager.create_session()
        await manager.send_message(session.session_id, __import__("uuid").uuid4(), "cotton dress")
        events = recorder.recent()
        assert len(events) == 1
        event = events[0]
        assert event["provider"] == "deepseek"
        assert event["model"] == "deepseek-chat"
        assert event["turn"] == 1
        assert event["prompt_tokens"] == 1
        assert event["completion_tokens"] == 1
        assert event["online"] is True
        assert event["cost_usd"] > 0
        assert event["latency_ms"] >= 0

    asyncio.run(scenario())


def make_runtime(tmp_path: Path) -> tuple[WebRuntime, FakeAgent]:
    catalog_path = tmp_path / "catalog.jsonl"
    write_rows(catalog_path, [{"parent_asin": "A1", "title": "One"}])
    agent = FakeAgent()
    catalog = CatalogPresenter.build(catalog_path)
    return WebRuntime(SessionManager(agent, catalog, top_k=10), catalog), agent


class ManagerWithRecorder:
    def __init__(self, runtime: WebRuntime, recorder: UsageRecorder) -> None:
        self.runtime = runtime
        self.recorder = recorder


def test_metrics_endpoint_returns_summary_and_recent(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    recorder = UsageRecorder()
    recorder.record(
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cost_usd": 0.00005,
            "online": True,
            "session_id": "s1",
            "turn": 1,
        }
    )
    app = create_app(runtime=runtime)
    app.state.runtime_container.manager = ManagerWithRecorder(runtime, recorder)

    with TestClient(app) as client:
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_turns"] == 1
        assert body["summary"]["total_cost_usd"] == 0.00005
        assert body["summary"]["per_provider"][0]["provider"] == "deepseek"
        assert body["recent"][0]["session_id"] == "s1"
        raw = json.dumps(body)
        assert "api_key" not in raw


def test_metrics_endpoint_503_without_manager(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        assert client.get("/api/metrics").status_code == 503
