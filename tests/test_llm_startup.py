from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from config.env_config import EnvConfig
from llm.base import LLMErrorCategory, LLMState, LLMStatus
from run_local_eval import initialize_llm, main


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

    def test_available_deepseek_and_openai_share_the_same_injection_path_and_hide_secrets(self) -> None:
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
            environ={"SKIP_DATA_VERIFY": "1", "LLM_PROVIDER": "none"}
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
            self.assertEqual(main(), 0)

        written_result = json.loads(write_text.call_args.args[0])
        self.assertEqual(
            written_result["intent_recognition_statistics"],
            agent.intent_recognition_statistics.return_value,
        )

if __name__ == "__main__":
    unittest.main()
