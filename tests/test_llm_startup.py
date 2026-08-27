from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from config.env_config import EnvConfig
from llm.base import LLMErrorCategory, LLMState, LLMStatus
from run_local_eval import initialize_llm


class LLMStartupTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
