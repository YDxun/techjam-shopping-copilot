from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from config.env_config import EnvConfig
from llm.base import LLMErrorCategory, LLMState, LLMStatus
from run_local_eval import ROOT, initialize_llm, main


class LLMStartupTest(unittest.TestCase):
    def _run_main_with_client(self, env: EnvConfig, client: Mock) -> tuple[Mock, str]:
        """Run the CLI through initialization while isolating evaluator I/O."""
        agent = Mock()
        agent.intent_recognition_statistics.return_value = {
            "total_turns": 0,
            "rule_resolutions": 0,
            "llm_attempts": 0,
            "llm_accepted": 0,
            "llm_fallbacks": 0,
            "fallback_reasons": {},
        }
        agent.transition_guard_statistics.return_value = {
            "enabled": False,
            "total": 0,
            "actions": {},
            "reasons": {},
            "dialogue_acts": {},
            "recognition_sources": {},
        }
        agent.dialogue_decision_statistics.return_value = {
            "enabled": False,
            "recorded": 0,
            "total_seen": 0,
            "decision_reasons": {},
            "guard_actions": {},
            "sanitizations": 0,
        }
        agent_constructor = Mock(return_value=agent)
        output = io.StringIO()
        with (
            patch("run_local_eval.EnvConfig.from_env", return_value=env),
            patch("run_local_eval.create_llm_client", return_value=client) as factory,
            patch("run_local_eval.Agent", agent_constructor),
            patch("run_local_eval.load_jsonl", return_value=[]),
            patch("run_local_eval.catalog_index", return_value=(set(), set(), {})),
            patch("run_local_eval.evaluate", return_value={}),
            patch("run_local_eval.Path.write_text"),
            patch("run_local_eval.sys.argv", ["run_local_eval.py"]),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 0)
        factory.assert_called_once_with(env.llm)
        client.initialize.assert_called_once_with()
        return agent_constructor, output.getvalue()

    def test_disabled_client_is_reported_without_error(self) -> None:
        env = EnvConfig.from_env(environ={})
        client = Mock()
        client.initialize.return_value = LLMStatus(
            state=LLMState.DISABLED,
            provider="deepseek",
            model="deepseek-chat",
        )
        output = io.StringIO()

        with patch("run_local_eval.create_llm_client", return_value=client):
            with redirect_stdout(output):
                returned = initialize_llm(env)

        self.assertIs(returned, client)
        self.assertIn("provider=deepseek", output.getvalue())
        self.assertIn("model=deepseek-chat", output.getvalue())
        self.assertIn("state=disabled", output.getvalue())
        self.assertIn("attempts=0", output.getvalue())
        self.assertNotIn("test-key", output.getvalue())

    def test_unavailable_client_does_not_raise(self) -> None:
        env = EnvConfig.from_env(environ={"DEEPSEEK_API_KEY": "test-key"})
        client = Mock()
        client.initialize.return_value = LLMStatus(
            state=LLMState.UNAVAILABLE,
            provider="deepseek",
            model="deepseek-chat",
            attempts=3,
            error_category=LLMErrorCategory.TIMEOUT,
            error_message="request timed out",
        )
        output = io.StringIO()

        with patch("run_local_eval.create_llm_client", return_value=client):
            with redirect_stdout(output):
                returned = initialize_llm(env)

        self.assertIs(returned, client)
        self.assertIn("provider=deepseek", output.getvalue())
        self.assertIn("model=deepseek-chat", output.getvalue())
        self.assertIn("state=unavailable", output.getvalue())
        self.assertIn("attempts=3", output.getvalue())
        self.assertIn("error=timeout", output.getvalue())
        self.assertNotIn("test-key", output.getvalue())


    def test_disabled_and_unavailable_clients_are_injected_without_reconstruction(self) -> None:
        for state, env_vars in (
            (LLMState.DISABLED, {"LLM_PROVIDER": "none"}),
            (LLMState.UNAVAILABLE, {"DEEPSEEK_API_KEY": "deepseek-test-secret"}),
        ):
            with self.subTest(state=state):
                env = EnvConfig.from_env(environ={"SKIP_DATA_VERIFY": "1", **env_vars})
                client = Mock()
                client.initialize.return_value = LLMStatus(
                    state=state,
                    provider=env.llm.provider,
                    model=env.llm.model,
                )
                agent_constructor, _ = self._run_main_with_client(env, client)
                self.assertIs(agent_constructor.call_args.kwargs.get("llm_client"), client)

    def test_available_clients_share_injection_path_and_hide_secrets(self) -> None:
        deepseek_secret = "deepseek-test-secret"
        openai_secret = "openai-test-secret"
        for provider in ("deepseek", "openai"):
            with self.subTest(provider=provider):
                env = EnvConfig.from_env(environ={
                    "SKIP_DATA_VERIFY": "1",
                    "LLM_PROVIDER": provider,
                    "DEEPSEEK_API_KEY": deepseek_secret,
                    "OPENAI_API_KEY": openai_secret,
                })
                client = Mock()
                client.initialize.return_value = LLMStatus(
                    state=LLMState.AVAILABLE,
                    provider=provider,
                    model=env.llm.model,
                )
                agent_constructor, output = self._run_main_with_client(env, client)
                self.assertIs(agent_constructor.call_args.kwargs.get("llm_client"), client)
                self.assertIn(f"provider={provider}", output)
                self.assertNotIn(deepseek_secret, output)
                self.assertNotIn(openai_secret, output)

    def test_submit_without_selected_key_stays_offline(self) -> None:
        env = EnvConfig.from_env(environ={
            "ENV_MODE": "submit",
            "LLM_PROVIDER": "openai",
            "DEEPSEEK_API_KEY": "inactive-test-secret",
            "SKIP_DATA_VERIFY": "1",
        })
        client = Mock()
        client.initialize.return_value = LLMStatus(
            state=LLMState.DISABLED,
            provider="openai",
            model=env.llm.model,
        )
        agent_constructor, _ = self._run_main_with_client(env, client)
        self.assertTrue(env.offline)
        self.assertIs(agent_constructor.call_args.kwargs.get("llm_client"), client)

    def test_submit_with_selected_key_rejects_before_initialization(self) -> None:
        env = EnvConfig.from_env(environ={
            "ENV_MODE": "submit",
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "openai-test-secret",
        })
        with (
            patch("run_local_eval.EnvConfig.from_env", return_value=env),
            patch("run_local_eval.initialize_llm") as initialize,
            patch("run_local_eval.sys.argv", ["run_local_eval.py"]),
        ):
            with self.assertRaises(AssertionError):
                main()
        initialize.assert_not_called()

    def test_main_writes_intent_recognition_statistics_to_results(self) -> None:
        env = EnvConfig.from_env(
            overrides={
                "diagnostics": {
                    "decision_trace": {
                        "enabled": True,
                        "output_path": "diagnostics/traces.jsonl",
                    }
                }
            },
            environ={"SKIP_DATA_VERIFY": "1", "LLM_PROVIDER": "none"},
        )
        client = Mock()
        client.initialize.return_value = LLMStatus(
            state=LLMState.DISABLED,
            provider="none",
            model="",
        )
        agent = Mock()
        agent.intent_recognition_statistics.return_value = {
            "total_turns": 3,
            "rule_resolutions": 2,
            "llm_attempts": 1,
            "llm_accepted": 0,
            "llm_fallbacks": 1,
            "fallback_reasons": {"invalid_json": 1},
        }
        agent.transition_guard_statistics.return_value = {
            "enabled": True,
            "total": 3,
            "actions": {"apply": 2, "clarify": 1},
            "reasons": {"guard_passed": 2, "replace_confidence_below_threshold": 1},
            "dialogue_acts": {"add_constraint": 2, "replace_constraint": 1},
            "recognition_sources": {"llm": 1, "rule": 2},
        }
        agent.dialogue_decision_statistics.return_value = {
            "enabled": True,
            "recorded": 1,
            "total_seen": 2,
            "decision_reasons": {"ask_other_first": 2},
            "guard_actions": {"apply": 2},
            "sanitizations": 0,
        }
        with (
            patch("run_local_eval.EnvConfig.from_env", return_value=env),
            patch("run_local_eval.create_llm_client", return_value=client),
            patch("run_local_eval.Agent", return_value=agent),
            patch("run_local_eval.load_jsonl", return_value=[]),
            patch("run_local_eval.catalog_index", return_value=(set(), set(), {})),
            patch(
                "run_local_eval.evaluate",
                return_value={"sample_count": 0, "sessions": [{"turns": []}]},
            ),
            patch("run_local_eval.Path.write_text") as write_text,
            patch("run_local_eval.sys.argv", ["run_local_eval.py"]),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)

        written_result = json.loads(write_text.call_args.args[0])
        self.assertEqual(
            written_result["intent_recognition_statistics"],
            agent.intent_recognition_statistics.return_value,
        )
        self.assertEqual(
            written_result["transition_guard_statistics"],
            agent.transition_guard_statistics.return_value,
        )
        self.assertEqual(
            written_result["dialogue_decision_statistics"],
            agent.dialogue_decision_statistics.return_value,
        )
        self.assertEqual(
            set(written_result),
            {
                "sample_count",
                "sessions",
                "intent_recognition_statistics",
                "transition_guard_statistics",
                "dialogue_decision_statistics",
            },
        )
        self.assertNotIn("decision_trace", written_result["sessions"][0])
        agent.dialogue.decision_trace_recorder.export_jsonl.assert_called_once_with(
            ROOT / "diagnostics/traces.jsonl"
        )

    def test_disabled_trace_export_does_not_resolve_or_touch_its_path(self) -> None:
        # Calling export while disabled could create or truncate a user-selected trace file.
        env = EnvConfig.from_env(environ={"SKIP_DATA_VERIFY": "1", "LLM_PROVIDER": "none"})
        client = Mock()
        client.initialize.return_value = LLMStatus(LLMState.DISABLED, "none", "")
        agent = Mock()
        agent.intent_recognition_statistics.return_value = {}
        agent.transition_guard_statistics.return_value = {}
        agent.dialogue_decision_statistics.return_value = {"enabled": False}
        agent.dialogue.decision_trace_recorder.config.enabled = False
        with (
            patch("run_local_eval.EnvConfig.from_env", return_value=env),
            patch("run_local_eval.create_llm_client", return_value=client),
            patch("run_local_eval.Agent", return_value=agent),
            patch("run_local_eval.load_jsonl", return_value=[]),
            patch("run_local_eval.catalog_index", return_value=(set(), set(), {})),
            patch("run_local_eval.evaluate", return_value={"sessions": [{"turns": []}]}),
            patch("run_local_eval.Path.write_text"),
            patch("run_local_eval.sys.argv", ["run_local_eval.py"]),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        agent.dialogue.decision_trace_recorder.export_jsonl.assert_not_called()

    def test_trace_export_path_is_repo_relative_and_protects_inputs(self) -> None:
        # A relative trace path must be stable, and a symlink must not bypass protected input paths.
        from run_local_eval import resolve_trace_output_path

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.jsonl"
            dataset = root / "dataset.jsonl"
            catalog.write_text("catalog", encoding="utf-8")
            dataset.write_text("dataset", encoding="utf-8")
            self.assertEqual(
                resolve_trace_output_path("diagnostics/traces.jsonl", catalog, dataset, root),
                (root / "diagnostics/traces.jsonl").resolve(),
            )
            alias = root / "catalog-alias.jsonl"
            alias.symlink_to(catalog)
            with self.assertRaisesRegex(ValueError, "catalog or dataset"):
                resolve_trace_output_path(alias, catalog, dataset, root)
            with self.assertRaisesRegex(ValueError, "catalog or dataset"):
                resolve_trace_output_path(catalog, catalog, dataset, root)
            dataset_alias = root / "dataset-alias.jsonl"
            dataset_alias.symlink_to(dataset)
            with self.assertRaisesRegex(ValueError, "catalog or dataset"):
                resolve_trace_output_path(dataset_alias, catalog, dataset, root)
            with self.assertRaisesRegex(ValueError, "catalog or dataset"):
                resolve_trace_output_path(dataset, catalog, dataset, root)

    def test_trace_export_refusal_keeps_completed_evaluation_result(self) -> None:
        # Failing before serialization would discard a completed evaluation for an optional export.
        env = EnvConfig.from_env(
            overrides={
                "diagnostics": {
                    "decision_trace": {"enabled": True, "output_path": "data/catalog.jsonl"}
                }
            },
            environ={"SKIP_DATA_VERIFY": "1", "LLM_PROVIDER": "none"},
        )
        client = Mock()
        client.initialize.return_value = LLMStatus(LLMState.DISABLED, "none", "")
        agent = Mock()
        agent.intent_recognition_statistics.return_value = {}
        agent.transition_guard_statistics.return_value = {}
        agent.dialogue_decision_statistics.return_value = {"enabled": True}
        with (
            patch("run_local_eval.EnvConfig.from_env", return_value=env),
            patch("run_local_eval.create_llm_client", return_value=client),
            patch("run_local_eval.Agent", return_value=agent),
            patch("run_local_eval.load_jsonl", return_value=[]),
            patch("run_local_eval.catalog_index", return_value=(set(), set(), {})),
            patch("run_local_eval.evaluate", return_value={"sample_count": 0}),
            patch("run_local_eval.Path.write_text") as write_text,
            patch("run_local_eval.sys.argv", ["run_local_eval.py"]),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 1)
        self.assertTrue(write_text.called)
        agent.dialogue.decision_trace_recorder.export_jsonl.assert_not_called()

if __name__ == "__main__":
    unittest.main()
