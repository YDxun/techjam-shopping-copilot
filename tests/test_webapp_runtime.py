import asyncio
from pathlib import Path
from types import SimpleNamespace

from config import constants
from config.env_config import EnvConfig
from tests.test_webapp_catalog import write_rows
from tests.test_webapp_service import FakeAgent, FakeCatalog
from webapp.app import WebRuntime
from webapp.runtime import RuntimeManager, _usage_context, apply_config
from webapp.service import SessionManager


def test_runtime_manager_create_verifies_catalog_by_default(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    write_rows(catalog_path, [{"parent_asin": "A1", "title": "One"}])
    calls: list[tuple[Path, str, str, bool]] = []

    def env_loader() -> EnvConfig:
        return EnvConfig.from_env(
            overrides={"skip_data_verify": False, "llm": {"provider": "none"}},
            environ={},
        )

    def verifier(path: Path, expected: str, label: str, skip: bool = False) -> bool:
        calls.append((Path(path), expected, label, skip))
        return True

    RuntimeManager.create(catalog_path, env_loader=env_loader, verifier=verifier)

    assert calls == [
        (catalog_path, constants.EXPECTED_SHA256_CATALOG, "catalog.jsonl", False)
    ]


def test_runtime_switch_rebuilds_when_api_key_changes(tmp_path: Path) -> None:
    base_env = EnvConfig.from_env(
        overrides={"skip_data_verify": True, "llm": {"provider": "none"}},
        environ={},
    )
    catalog = FakeCatalog()
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=catalog)
    built: list[object] = []

    def build_runtime(env: EnvConfig, cfg: dict[str, object]) -> object:
        runtime = WebRuntime(SessionManager(FakeAgent(), catalog, top_k=10), catalog)
        built.append(runtime)
        return runtime

    manager._build_runtime = build_runtime  # type: ignore[method-assign]
    common = {
        "llm_provider": "deepseek",
        "llm_model": "deepseek-v4-flash",
        "rerank_backend": "none",
        "retrieval_backend": "bm25",
        "output_strategy": "holdback",
    }

    first, first_key = manager.switch({**common, "api_key": "first-secret"})
    second, second_key = manager.switch({**common, "api_key": "second-secret"})

    assert second is not first
    assert second_key != first_key
    assert len(built) == 2


def test_runtime_switch_keeps_active_key_when_form_resubmits_blank(tmp_path: Path) -> None:
    base_env = EnvConfig.from_env(
        overrides={"skip_data_verify": True, "llm": {"provider": "none"}},
        environ={},
    )
    catalog = FakeCatalog()
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=catalog)
    observed_keys: list[str] = []

    def build_runtime(env: EnvConfig, cfg: dict[str, object]) -> object:
        observed_keys.append(env.llm.api_key.reveal())
        return WebRuntime(SessionManager(FakeAgent(), catalog, top_k=10), catalog)

    manager._build_runtime = build_runtime  # type: ignore[method-assign]
    first = {
        "llm_provider": "deepseek",
        "llm_model": "deepseek-v4-flash",
        "api_key": "server-memory-secret",
        "rerank_backend": "none",
        "retrieval_backend": "bm25",
        "output_strategy": "holdback",
    }
    second = {**first, "api_key": "", "retrieval_backend": "hybrid"}

    manager.switch(first)
    manager.switch(second)

    assert observed_keys == ["server-memory-secret", "server-memory-secret"]
    assert manager.runtime_info()["active"]["api_key_set"] is True
    assert "api_key" not in manager.active_config


def test_apply_config_uses_the_model_selected_in_the_web_panel() -> None:
    base_env = EnvConfig.from_env(
        overrides={"skip_data_verify": True, "llm": {"provider": "none"}},
        environ={},
    )

    configured = apply_config(
        base_env,
        {
            "llm_provider": "deepseek",
            "llm_model": "deepseek-reasoner",
            "api_key": "secret",
            "rerank_backend": "none",
            "retrieval_backend": "bm25",
            "output_strategy": "holdback",
        },
    )

    assert configured.llm.model == "deepseek-reasoner"


def test_apply_config_preserves_unrelated_base_configuration() -> None:
    base_env = EnvConfig.from_env(
        overrides={
            "top_k": 7,
            "llm": {
                "provider": "none",
                "timeout_seconds": 19.0,
                "retry": {"max_retries": 1},
            },
        },
        environ={},
    )

    configured = apply_config(
        base_env,
        {
            "llm_provider": "none",
            "rerank_backend": "none",
            "retrieval_backend": "hybrid",
            "output_strategy": "full",
        },
    )

    assert configured.top_k == 7
    assert configured.llm.timeout_seconds == 19.0
    assert configured.llm.retry.max_retries == 1
    assert configured.retrieval_backend == "hybrid"
    assert configured.emit_gate is False


def test_runtime_info_uses_the_canonical_configured_model(tmp_path: Path) -> None:
    base_env = EnvConfig.from_env(environ={})
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=object())
    manager.active_config = {"llm_provider": "deepseek"}

    info = manager.runtime_info()

    assert info["active"]["model"] == "deepseek-v4-flash"
    assert info["providers"]["deepseek"]["models"][0] == "deepseek-v4-flash"


def test_runtime_info_reports_qwen_environment_key_availability(
    tmp_path: Path, monkeypatch
) -> None:
    base_env = EnvConfig.from_env(environ={})
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=object())

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    unavailable = manager.runtime_info()["rerank_backends"]["text"]
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")
    available = manager.runtime_info()["rerank_backends"]["text"]

    assert unavailable["configured"] is False
    assert unavailable["requires_env"] == "DASHSCOPE_API_KEY"
    assert "DASHSCOPE_API_KEY" in unavailable["label"]
    assert available["configured"] is True


def test_qwen_only_runtime_is_reported_as_online(tmp_path: Path, monkeypatch) -> None:
    base_env = EnvConfig.from_env(environ={})
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=object())
    manager.active_config = {
        "llm_provider": "none",
        "rerank_backend": "text",
    }
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")

    info = manager.runtime_info()

    assert info["active"]["offline"] is False
    assert info["active"]["qwen_api_key_set"] is True
    assert "llm=yes" in info["fingerprint"]
    assert "network=yes" in info["fingerprint"]


def test_failed_qwen_probe_reports_configured_but_offline(tmp_path: Path, monkeypatch) -> None:
    base_env = EnvConfig.from_env(environ={})
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=object())
    manager.active_config = {
        "llm_provider": "none",
        "rerank_backend": "text",
    }
    manager.active = SimpleNamespace(
        sessions=SimpleNamespace(
            capability_profile=SimpleNamespace(
                llm_available=False,
                text_rerank_available=False,
            )
        )
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "invalid-key")

    info = manager.runtime_info()

    assert info["rerank_backends"]["text"]["configured"] is True
    assert info["rerank_backends"]["text"]["available"] is False
    assert info["active"]["qwen_available"] is False
    assert info["active"]["offline"] is True
    assert "llm=no" in info["fingerprint"]


def test_qwen_text_rerank_can_run_without_a_chat_llm_provider() -> None:
    base_env = EnvConfig.from_env(environ={})

    configured = apply_config(
        base_env,
        {
            "llm_provider": "none",
            "rerank_backend": "text",
            "retrieval_backend": "bm25",
            "output_strategy": "holdback",
        },
    )

    assert configured.llm.provider == "none"
    assert configured.llm.rerank_enabled is True
    assert configured.llm.rerank_backend == "text"


def test_qwen_only_usage_is_attributed_to_dashscope() -> None:
    context = _usage_context(
        {
            "llm_provider": "none",
            "llm_model": "",
            "rerank_backend": "text",
            "retrieval_backend": "bm25",
            "output_strategy": "holdback",
        }
    )

    assert context["provider"] == "dashscope"
    assert context["model"] == "qwen3-rerank"


def test_cached_engine_does_not_resurrect_sessions_after_config_switch(
    tmp_path: Path,
) -> None:
    base_env = EnvConfig.from_env(
        overrides={"skip_data_verify": True, "llm": {"provider": "none"}},
        environ={},
    )
    catalog = FakeCatalog()
    manager = RuntimeManager(tmp_path / "catalog.jsonl", base_env, catalog=catalog)

    def build_runtime(env: EnvConfig, cfg: dict[str, object]) -> WebRuntime:
        return WebRuntime(SessionManager(FakeAgent(), catalog, top_k=10), catalog)

    manager._build_runtime = build_runtime  # type: ignore[method-assign]
    first_config = {
        "llm_provider": "none",
        "rerank_backend": "none",
        "retrieval_backend": "bm25",
        "output_strategy": "holdback",
    }
    second_config = {**first_config, "retrieval_backend": "hybrid"}

    first_runtime, _ = manager.switch(first_config)
    first_session = asyncio.run(first_runtime.sessions.create_session())
    manager.switch(second_config)
    restored_runtime, _ = manager.switch(first_config)

    assert restored_runtime is not first_runtime
    assert restored_runtime.sessions.get_session(first_session.session_id) is None
