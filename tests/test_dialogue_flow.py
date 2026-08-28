from __future__ import annotations

import unittest

from agent.main_agent import Agent
from config.env_config import EnvConfig
from evaluator.local_evaluator import ALLOWED_ATTRIBUTES
from llm.base import DisabledLLMClient, LLMState, LLMStatus, LLMUsage


PRODUCTS = (
    {
        "parent_asin": "A",
        "title": "Cotton running shoe",
        "features": ["cotton"],
        "categories": ["Shoes"],
    },
    {
        "parent_asin": "B",
        "title": "Leather running shoe",
        "features": ["leather"],
        "categories": ["Shoes"],
    },
    {
        "parent_asin": "C",
        "title": "Blue walking shoe",
        "features": ["blue"],
        "categories": ["Shoes"],
    },
)


class StaticRetriever:
    def iter_products(self) -> tuple[dict, ...]:
        return PRODUCTS

    def search(self, route, top_k: int, mode: str) -> list[dict]:
        return [
            {"parent_asin": item["parent_asin"], "rrf": 1.0 / index}
            for index, item in enumerate(PRODUCTS, start=1)
        ][:top_k]

    def close(self) -> None:
        return None


class StaticReranker:
    def __init__(self, order: tuple[str, ...]) -> None:
        self.order = order
        self.last_usage = {"prompt_tokens": 5, "completion_tokens": 2}

    def rerank(self, retriever, candidates, state, route, top_k: int, mode: str) -> list[str]:
        return list(self.order[:top_k])


class UnavailableClient:
    @property
    def status(self) -> LLMStatus:
        return LLMStatus(LLMState.UNAVAILABLE, "deepseek", "deepseek-chat")

    @property
    def cumulative_usage(self) -> LLMUsage:
        return LLMUsage()

    def initialize(self) -> LLMStatus:
        return self.status

    def chat(self, messages, *, temperature=None, max_tokens=None):
        raise AssertionError("unavailable client must not be called")


class DialogueFlowTest(unittest.TestCase):
    def env(self, mode: str = "rule_only") -> EnvConfig:
        return EnvConfig.from_env(
            overrides={
                "skip_data_verify": True,
                "dialogue_understanding": {"mode": mode},
                "llm": {"rerank_enabled": False},
            },
            environ={"LLM_PROVIDER": "none"},
        )

    def test_offline_response_preserves_existing_ranked_order_and_contract(self) -> None:
        agent = Agent(
            env=self.env(),
            llm_client=DisabledLLMClient(),
            retriever=StaticRetriever(),
            reranker=StaticReranker(("B", "A", "C")),
        )
        agent.reset("s1", {})

        response = agent.respond(
            "s1",
            "I'm looking for shoes. A key requirement is: cotton.",
            1,
            3,
        )

        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES | {None})
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["B", "A", "C"],
        )
        self.assertEqual(response["usage"], {"prompt_tokens": 5, "completion_tokens": 2})

    def test_unavailable_llm_uses_rule_path_and_keeps_session_valid(self) -> None:
        agent = Agent(
            env=self.env(mode="cascaded"),
            llm_client=UnavailableClient(),
            retriever=StaticRetriever(),
            reranker=StaticReranker(("A", "B", "C")),
        )
        agent.reset("s1", {})

        first = agent.respond("s1", "I'm looking for shoes, but I'm still exploring.", 1, 3)
        second = agent.respond("s1", "What matters is: cotton.", 2, 3)

        self.assertEqual(len(first["recommendations"]), 3)
        self.assertEqual(len(second["recommendations"]), 3)
        session = agent.dialogue.session("s1")
        self.assertEqual(session.dialogue.turn, 2)
        self.assertEqual(session.dialogue.active_constraints[0].value, "cotton")
        self.assertEqual(session.dialogue.intent_version, 1)

    def test_actual_shown_results_are_recorded_after_response(self) -> None:
        agent = Agent(
            env=self.env(),
            llm_client=DisabledLLMClient(),
            retriever=StaticRetriever(),
            reranker=StaticReranker(("C", "B", "A")),
        )
        agent.reset("s1", {})

        agent.respond("s1", "I'm looking for shoes.", 1, 2)

        observations = agent.dialogue.session("s1").products.observations
        self.assertEqual([item.asin for item in observations], ["C", "B"])


if __name__ == "__main__":
    unittest.main()
