from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

from config.env_config import EnvConfig
from config.loader import ConfigError, load_config


def contains_string(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return expected in value
    if isinstance(value, dict):
        return any(contains_string(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_string(item, expected) for item in value)
    return False


class ConfigLoaderTest(unittest.TestCase):
    def test_hybrid_question_configuration_accepts_all_numeric_overrides(self) -> None:
        config = load_config(
            overrides={
                "decision": {
                    "hybrid_question_policy": {
                        "max_replacements_per_session": 0,
                        "pool_size": 400,
                        "prior_alpha": 0.5,
                        "prior_temperature": 2.0,
                        "minimum_coverage": 0.7,
                        "maximum_missing_rate": 0.3,
                        "minimum_expected_shrink": 0.4,
                        "minimum_resolve_at_10": 0.2,
                        "minimum_gain": 0.3,
                        "weights": {
                            "expected_shrink": 0.41,
                            "resolve_at_10": 0.26,
                            "coverage": 0.16,
                            "answer_probability": 0.11,
                            "extraction_confidence": 0.12,
                            "missing_penalty": 0.27,
                            "turn_cost": 0.13,
                        },
                    }
                }
            },
            environ={},
        )

        policy = config.decision.hybrid_question_policy
        self.assertEqual(policy.max_replacements_per_session, 0)
        self.assertEqual(policy.pool_size, 400)
        self.assertEqual(policy.prior_alpha, 0.5)
        self.assertEqual(policy.prior_temperature, 2.0)
        self.assertEqual(policy.minimum_coverage, 0.7)
        self.assertEqual(policy.maximum_missing_rate, 0.3)
        self.assertEqual(policy.minimum_expected_shrink, 0.4)
        self.assertEqual(policy.minimum_resolve_at_10, 0.2)
        self.assertEqual(policy.minimum_gain, 0.3)
        self.assertEqual(policy.weights.expected_shrink, 0.41)
        self.assertEqual(policy.weights.resolve_at_10, 0.26)
        self.assertEqual(policy.weights.coverage, 0.16)
        self.assertEqual(policy.weights.answer_probability, 0.11)
        self.assertEqual(policy.weights.extraction_confidence, 0.12)
        self.assertEqual(policy.weights.missing_penalty, 0.27)
        self.assertEqual(policy.weights.turn_cost, 0.13)

    def test_hybrid_question_configuration_rejects_invalid_domains(self) -> None:
        cases = (
            ({"pool_size": 0}, "pool_size"),
            ({"prior_alpha": 1.1}, "prior_alpha"),
            ({"prior_temperature": 0}, "prior_temperature"),
            ({"minimum_coverage": -0.1}, "minimum_coverage"),
            ({"maximum_missing_rate": 1.1}, "maximum_missing_rate"),
            ({"minimum_expected_shrink": -0.1}, "minimum_expected_shrink"),
            ({"minimum_resolve_at_10": 1.1}, "minimum_resolve_at_10"),
            ({"minimum_gain": -0.1}, "minimum_gain"),
            ({"max_replacements_per_session": 2}, "max_replacements_per_session"),
            ({"only_after_other_asked": False}, "only_after_other_asked"),
            ({"weights": {"coverage": -0.1}}, "weights.coverage"),
            ({"weights": {"coverage": float("inf")}}, "weights.coverage"),
        )
        for policy, field in cases:
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, field):
                load_config(
                    overrides={"decision": {"hybrid_question_policy": policy}}, environ={}
                )

    def test_hybrid_question_configuration_is_mutually_exclusive_with_dynamic_mode(self) -> None:
        with self.assertRaisesRegex(ConfigError, "hybrid_question_policy"):
            load_config(
                overrides={
                    "decision": {
                        "candidate_question_value": {"enabled": True},
                        "hybrid_question_policy": {"enabled": True},
                    }
                },
                environ={},
            )

    def test_dynamic_question_configuration_rejects_invalid_domains(self) -> None:
        cases = (
            (
                {"candidate_question_value": {"pool_size": 0}},
                "candidate_question_value.pool_size",
            ),
            (
                {"candidate_question_value": {"prior_alpha": -0.1}},
                "candidate_question_value.prior_alpha",
            ),
            (
                {"candidate_question_value": {"prior_temperature": 0}},
                "candidate_question_value.prior_temperature",
            ),
            (
                {"candidate_question_value": {"other_answer_probability": 1.1}},
                "candidate_question_value.other_answer_probability",
            ),
            (
                {"candidate_question_value": {"other_vagueness_penalty": -0.1}},
                "candidate_question_value.other_vagueness_penalty",
            ),
            (
                {"candidate_question_value": {"weights": {"coverage": -0.1}}},
                "candidate_question_value.weights.coverage",
            ),
            (
                {"finish_strategy": {"candidate_threshold": 0}},
                "finish_strategy.candidate_threshold",
            ),
            (
                {"finish_strategy": {"remaining_question_threshold": 0}},
                "finish_strategy.remaining_question_threshold",
            ),
            ({"finish_strategy": {"lookahead_depth": 3}}, "finish_strategy.lookahead_depth"),
            (
                {"finish_strategy": {"minimum_finish_gain": -0.1}},
                "finish_strategy.minimum_finish_gain",
            ),
            (
                {"finish_strategy": {"weights": {"resolve_at_10": -0.1}}},
                "finish_strategy.weights.resolve_at_10",
            ),
            ({"question_termination_mode": "utility"}, "question_termination_mode"),
        )
        for decision, field in cases:
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, field):
                load_config(overrides={"decision": decision}, environ={})

    def write_config(self, root: Path, payload: dict) -> Path:
        path = root / "settings.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_layers_json_environment_and_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {
                "top_k": 7,
                "llm": {
                    "providers": {"deepseek": {"model": "json-model"}},
                    "retry": {"max_retries": 1},
                },
            })
            config = load_config(
                path=path,
                environ={"LLM_MODEL": "env-model", "DEEPSEEK_API_KEY": "secret-value"},
                overrides={"top_k": 9, "llm": {"retry": {"max_retries": 2}}},
            )
        self.assertEqual(config.top_k, 9)
        self.assertEqual(config.llm.model, "env-model")
        self.assertEqual(config.llm.retry.max_retries, 2)
        self.assertEqual(config.llm.api_key.reveal(), "secret-value")
        self.assertNotIn("secret-value", repr(config.llm))

    def test_default_provider_profiles(self) -> None:
        config = load_config(environ={})
        self.assertEqual(config.llm.providers.deepseek.model, "deepseek-chat")
        self.assertEqual(config.llm.providers.deepseek.base_url, "https://api.deepseek.com")
        self.assertEqual(config.llm.providers.deepseek.token_limit_parameter, "max_tokens")
        self.assertTrue(config.llm.providers.deepseek.supports_temperature)
        self.assertEqual(config.llm.providers.openai.model, "gpt-4o-mini")
        self.assertEqual(config.llm.providers.openai.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.llm.providers.openai.token_limit_parameter, "max_completion_tokens")
        self.assertTrue(config.llm.providers.openai.supports_temperature)

    def test_dataclass_serialization_never_contains_selected_provider_key(self) -> None:
        secret = "selected-provider-secret"
        config = load_config(environ={
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": secret,
            "OPENAI_API_KEY": "inactive-secret",
        })
        serialized = asdict(config)
        self.assertFalse(contains_string(serialized, secret))
        self.assertFalse(contains_string(serialized, "inactive-secret"))
        self.assertNotIn(secret, json.dumps(serialized, default=str))
        self.assertEqual(config.llm.selected_profile.api_key.reveal(), secret)

    def test_provider_specific_values_persist_and_selected_override_wins(self) -> None:
        config = load_config(environ={
            "LLM_PROVIDER": "openai",
            "DEEPSEEK_MODEL": "deepseek-reasoner",
            "OPENAI_MODEL": "gpt-profile-model",
            "LLM_MODEL": "selected-openai-model",
        })
        self.assertEqual(config.llm.providers.deepseek.model, "deepseek-reasoner")
        self.assertEqual(config.llm.providers.openai.model, "selected-openai-model")
        self.assertEqual(config.llm.selected_profile.model, "selected-openai-model")

    def test_provider_environment_wins_over_legacy_backend(self) -> None:
        config = load_config(environ={"LLM_PROVIDER": "deepseek", "LLM_BACKEND": "openai"})
        self.assertEqual(config.llm.provider, "deepseek")

    def test_legacy_backend_maps_to_provider_when_provider_is_absent(self) -> None:
        for backend, provider in (("openai", "openai"), ("none", "none"), ("local", "none")):
            with self.subTest(backend=backend):
                self.assertEqual(
                    load_config(environ={"LLM_BACKEND": backend}).llm.provider, provider
                )

    def test_selected_base_url_override_does_not_change_inactive_profile(self) -> None:
        config = load_config(environ={
            "LLM_PROVIDER": "openai",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "LLM_BASE_URL": "https://selected.example/v1",
        })
        self.assertEqual(config.llm.providers.deepseek.base_url, "https://deepseek.example/v1")
        self.assertEqual(config.llm.providers.openai.base_url, "https://selected.example/v1")


    def test_explicit_provider_directs_generic_environment_values_to_openai_profile(self) -> None:
        config = load_config(
            environ={
                "LLM_MODEL": "generic-model",
                "LLM_BASE_URL": "https://generic.example/v1",
                "DEEPSEEK_MODEL": "deepseek-environment-model",
                "DEEPSEEK_BASE_URL": "https://deepseek.environment/v1",
            },
            overrides={"llm": {"provider": "openai"}},
        )
        self.assertEqual(config.llm.providers.openai.model, "generic-model")
        self.assertEqual(config.llm.providers.openai.base_url, "https://generic.example/v1")
        self.assertEqual(config.llm.providers.deepseek.model, "deepseek-environment-model")
        self.assertEqual(config.llm.providers.deepseek.base_url, "https://deepseek.environment/v1")

    def test_explicit_selected_profile_values_beat_generic_environment_values(self) -> None:
        config = load_config(
            environ={
                "LLM_MODEL": "generic-model",
                "LLM_BASE_URL": "https://generic.example/v1",
            },
            overrides={
                "llm": {
                    "provider": "openai",
                    "providers": {
                        "openai": {
                            "model": "explicit-openai-model",
                            "base_url": "https://explicit.openai/v1",
                        }
                    },
                }
            },
        )
        self.assertEqual(config.llm.selected_profile.model, "explicit-openai-model")
        self.assertEqual(config.llm.selected_profile.base_url, "https://explicit.openai/v1")
    def test_only_selected_provider_receives_its_matching_key(self) -> None:
        config = load_config(environ={
            "LLM_PROVIDER": "openai",
            "DEEPSEEK_API_KEY": "inactive-secret",
            "OPENAI_API_KEY": "selected-secret",
        })
        self.assertFalse(config.llm.providers.deepseek.api_key)
        self.assertEqual(config.llm.providers.openai.api_key.reveal(), "selected-secret")

    def test_rejects_secret_in_json_or_explicit_overrides(self) -> None:
        payloads = (
            ({"llm": {"api_key": "forbidden"}}, "DEEPSEEK_API_KEY"),
            ({"llm": {"providers": {"openai": {"api_key": "forbidden"}}}}, "OPENAI_API_KEY"),
        )
        for payload, message in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ConfigError, message):
                    load_config(path=self.write_config(Path(directory), payload), environ={})
                with self.assertRaisesRegex(ConfigError, message):
                    load_config(overrides=payload, environ={})

    def test_rejects_invalid_profile_capabilities(self) -> None:
        cases = (
            ({"token_limit_parameter": "tokens"}, "token_limit_parameter"),
            ({"supports_temperature": "true"}, "supports_temperature"),
        )
        for changes, field in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ConfigError, field):
                    load_config(overrides={"llm": {"providers": {"openai": changes}}}, environ={})

    def test_rejects_non_positive_llm_rerank_candidates(self) -> None:
        with self.assertRaisesRegex(ConfigError, "llm.rerank_candidates"):
            load_config(environ={"LLM_RERANK_CANDIDATES": "0"})

    def test_missing_key_is_valid(self) -> None:
        config = load_config(environ={})
        self.assertFalse(config.llm.api_key)
        self.assertEqual(config.llm.provider, "deepseek")

    def test_app_config_path_selects_file_when_direct_path_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {"top_k": 13})
            self.assertEqual(load_config(environ={"APP_CONFIG_PATH": str(path)}).top_k, 13)

    def test_direct_path_takes_precedence_over_app_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct_path = self.write_config(root, {"top_k": 14})
            env_path = root / "environment.json"
            env_path.write_text(json.dumps({"top_k": 15}), encoding="utf-8")
            self.assertEqual(
                load_config(path=direct_path, environ={"APP_CONFIG_PATH": str(env_path)}).top_k,
                14,
            )

    def test_rejects_invalid_json_and_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "invalid JSON"):
                load_config(path=path, environ={})
        with self.assertRaisesRegex(ConfigError, "does not exist"):
            load_config(path="missing-settings.json", environ={})

    def test_rejects_invalid_boolean_text_and_provider(self) -> None:
        with self.assertRaisesRegex(ConfigError, "LLM_RERANK"):
            load_config(environ={"LLM_RERANK": "sometimes"})
        with self.assertRaisesRegex(ConfigError, "llm.provider"):
            load_config(environ={"LLM_PROVIDER": "unknown"})

    def test_rejects_invalid_numeric_configuration(self) -> None:
        for name in ("LLM_CONNECT_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"):
            with self.subTest(name=name), self.assertRaisesRegex(ConfigError, "must be > 0"):
                load_config(environ={name: "0"})
        with self.assertRaisesRegex(ConfigError, "llm.retry.max_retries"):
            load_config(environ={"LLM_MAX_RETRIES": "-1"})
        with self.assertRaisesRegex(ConfigError, "llm.circuit_breaker.failure_threshold"):
            load_config(environ={"LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "0"})
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ConfigError, "llm.timeout_seconds"
            ):
                load_config(environ={"LLM_TIMEOUT_SECONDS": value})

    def test_models_are_immutable(self) -> None:
        config = load_config(environ={})
        with self.assertRaises(FrozenInstanceError):
            config.llm.retry.max_retries = 3


class EnvConfigCompatibilityTest(unittest.TestCase):
    def test_exposes_existing_flat_fields_and_nested_llm(self) -> None:
        env = EnvConfig.from_env(
            environ={
                "TOP_K": "8",
                "SAMPLE_LIMIT": "4",
                "LLM_MODEL": "deepseek-reasoner",
                "DEEPSEEK_API_KEY": "secret-value",
            }
        )
        self.assertEqual(env.top_k, 8)
        self.assertEqual(env.sample_limit, 4)
        self.assertEqual(env.llm.model, "deepseek-reasoner")
        self.assertEqual(env.llm_model, "deepseek-reasoner")
        self.assertFalse(env.offline)
        self.assertFalse(hasattr(env, "env_overrides"))
        self.assertNotIn("secret-value", repr(env))

    def test_openai_compatibility_fields_use_selected_profile(self) -> None:
        env = EnvConfig.from_env(environ={
            "LLM_PROVIDER": "openai", "OPENAI_API_KEY": "legacy-secret",
            "OPENAI_BASE_URL": "https://legacy.invalid", "OPENAI_MODEL": "legacy-reranker-model",
        })
        self.assertEqual(env.llm_backend, "openai")
        self.assertEqual(env.openai_api_key, "legacy-secret")
        self.assertEqual(env.openai_base_url, "https://legacy.invalid")
        self.assertEqual(env.llm_model, "legacy-reranker-model")
        self.assertNotIn("legacy-secret", repr(env))

    def test_submit_mode_requires_the_selected_provider_key(self) -> None:
        self.assertTrue(
            EnvConfig.from_env(environ={"ENV_MODE": "submit", "LLM_PROVIDER": "openai"}).offline
        )
        self.assertFalse(
            EnvConfig.from_env(
                environ={
                    "ENV_MODE": "submit",
                    "LLM_BACKEND": "openai",
                    "OPENAI_API_KEY": "selected-secret",
                }
            ).offline
        )

    def test_inactive_provider_key_does_not_make_selected_provider_online(self) -> None:
        for environ in (
            {"ENV_MODE": "submit", "LLM_PROVIDER": "openai", "DEEPSEEK_API_KEY": "deepseek-secret"},
            {"ENV_MODE": "submit", "LLM_PROVIDER": "deepseek", "OPENAI_API_KEY": "openai-secret"},
        ):
            with self.subTest(environ=environ):
                self.assertTrue(EnvConfig.from_env(environ=environ).offline)


if __name__ == "__main__":
    unittest.main()
