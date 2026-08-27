from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from config.loader import ConfigError, load_config


class ConfigLoaderTest(unittest.TestCase):
    def write_config(self, root: Path, payload: dict) -> Path:
        path = root / "settings.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_layers_json_environment_and_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {
                "top_k": 7,
                "llm": {"model": "json-model", "retry": {"max_retries": 1}},
            })
            config = load_config(
                path=path,
                environ={"LLM_MODEL": "env-model", "DEEPSEEK_API_KEY": "secret-value"},
                overrides={"top_k": 9, "llm": {"retry": {"max_retries": 2}}},
            )
            self.assertEqual(config.top_k, 9)
            self.assertEqual(config.llm.model, "env-model")
            self.assertEqual(config.llm.retry.max_retries, 2)
            self.assertEqual(config.llm.api_key, "secret-value")
            self.assertNotIn("secret-value", repr(config.llm))

    def test_rejects_secret_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {"llm": {"api_key": "forbidden"}})
            with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
                load_config(path=path, environ={})

    def test_rejects_secret_in_explicit_overrides(self) -> None:
        with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
            load_config(environ={}, overrides={"llm": {"api_key": "forbidden"}})

    def test_missing_key_is_valid(self) -> None:
        config = load_config(environ={})
        self.assertEqual(config.llm.api_key, "")
        self.assertEqual(config.llm.provider, "deepseek")

    def test_app_config_path_selects_file_when_direct_path_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {"top_k": 13})
            config = load_config(environ={"APP_CONFIG_PATH": str(path)})
            self.assertEqual(config.top_k, 13)

    def test_direct_path_takes_precedence_over_app_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct_path = self.write_config(root, {"top_k": 14})
            env_path = root / "environment.json"
            env_path.write_text(json.dumps({"top_k": 15}), encoding="utf-8")
            config = load_config(path=direct_path, environ={"APP_CONFIG_PATH": str(env_path)})
            self.assertEqual(config.top_k, 14)

    def test_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "invalid JSON"):
                load_config(path=path, environ={})

    def test_rejects_missing_explicit_config_path(self) -> None:
        with self.assertRaisesRegex(ConfigError, "does not exist"):
            load_config(path="missing-settings.json", environ={})

    def test_rejects_invalid_boolean_text(self) -> None:
        with self.assertRaisesRegex(ConfigError, "LLM_RERANK"):
            load_config(environ={"LLM_RERANK": "sometimes"})

    def test_rejects_unsupported_provider(self) -> None:
        with self.assertRaisesRegex(ConfigError, "llm.provider"):
            load_config(environ={"LLM_PROVIDER": "openai"})

    def test_rejects_non_positive_timeouts(self) -> None:
        for name in ("LLM_CONNECT_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ConfigError, "must be > 0"):
                    load_config(environ={name: "0"})

    def test_rejects_negative_retries(self) -> None:
        with self.assertRaisesRegex(ConfigError, "llm.retry.max_retries"):
            load_config(environ={"LLM_MAX_RETRIES": "-1"})

    def test_rejects_non_positive_circuit_threshold(self) -> None:
        with self.assertRaisesRegex(ConfigError, "llm.circuit_breaker.failure_threshold"):
            load_config(environ={"LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "0"})

    def test_models_are_immutable(self) -> None:
        config = load_config(environ={})
        with self.assertRaises(FrozenInstanceError):
            config.llm.retry.max_retries = 3


if __name__ == "__main__":
    unittest.main()
