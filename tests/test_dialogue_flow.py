from __future__ import annotations

import json
import unittest

from agent.dialogue.models import GuardAction
from agent.dialogue.pipeline import DialogueUnderstandingPipeline
from agent.main_agent import Agent
from config.env_config import EnvConfig
from evaluator.local_evaluator import ALLOWED_ATTRIBUTES
from llm.base import DisabledLLMClient, LLMResult, LLMState, LLMStatus, LLMUsage

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

    def rerank(
        self,
        retriever,
        candidates,
        state,
        route,
        top_k: int,
        mode: str,
        use_reranker_model: bool = False,
        use_llm_rerank: bool = False,
    ) -> list[str]:
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


class ScriptedIntentClient:
    def __init__(self, response: dict[str, object]) -> None:
        self._response = json.dumps(response)

    @property
    def status(self) -> LLMStatus:
        return LLMStatus(LLMState.AVAILABLE, "test", "test-model")

    @property
    def cumulative_usage(self) -> LLMUsage:
        return LLMUsage()

    def initialize(self) -> LLMStatus:
        return self.status

    def chat(self, messages, *, temperature=None, max_tokens=None) -> LLMResult:
        return LLMResult(True, "test", "test-model", content=self._response)


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

    @staticmethod
    def low_confidence_rejection(asin: str) -> dict[str, object]:
        return {
            "dialogue_act": "reject_products",
            "category": None,
            "constraint_operations": [],
            "explicit_rejected_asins": [asin],
            "confidence": 0.70,
            "ambiguities": [],
        }

    def build_pipeline(
        self, *, guard_enabled: bool, llm_response: dict[str, object]
    ) -> DialogueUnderstandingPipeline:
        env = EnvConfig.from_env(
            overrides={
                "skip_data_verify": True,
                "dialogue_understanding": {
                    "mode": "cascaded",
                    "rule_confidence_threshold": 0.90,
                    "transition_guard": {"enabled": guard_enabled},
                },
                "llm": {"rerank_enabled": False},
            },
            environ={"LLM_PROVIDER": "none"},
        )
        return DialogueUnderstandingPipeline(
            env=env,
            llm_client=ScriptedIntentClient(llm_response),
            products=PRODUCTS,
        )

    def build_rule_pipeline(self, *, guard_enabled: bool) -> DialogueUnderstandingPipeline:
        env = EnvConfig.from_env(
            overrides={
                "skip_data_verify": True,
                "dialogue_understanding": {
                    "mode": "rule_only",
                    "transition_guard": {"enabled": guard_enabled},
                },
                "llm": {"rerank_enabled": False},
            },
            environ={"LLM_PROVIDER": "none"},
        )
        return DialogueUnderstandingPipeline(
            env=env,
            llm_client=DisabledLLMClient(),
            products=PRODUCTS,
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
        self.assertEqual(
            set(response),
            {"message", "ask_attribute", "recommendations", "usage"},
        )
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

    def test_guarded_rejection_does_not_mutate_products_or_dialogue(self) -> None:
        pipeline = self.build_pipeline(
            guard_enabled=True,
            llm_response=self.low_confidence_rejection("A"),
        )
        pipeline.reset("s", {})
        pipeline.record_shown("s", ["A"], turn=1)
        before = pipeline.session("s")

        result = pipeline.process_turn("s", "Reject A", turn=2)
        session = pipeline.session("s")

        self.assertEqual(result.guard_decision.action, GuardAction.CLARIFY)
        self.assertEqual(result.question_decision.ask_attribute, "other")
        self.assertEqual(session, before)
        self.assertEqual(session.dialogue.active_constraints, ())
        self.assertEqual(session.products.context_lists(1).hard_rejected_asins, ())

    def test_disabled_guard_preserves_existing_feedback_flow(self) -> None:
        pipeline = self.build_pipeline(
            guard_enabled=False,
            llm_response=self.low_confidence_rejection("A"),
        )
        pipeline.reset("s", {})
        pipeline.record_shown("s", ["A"], turn=1)

        result = pipeline.process_turn("s", "Reject A", turn=2)
        session = pipeline.session("s")

        self.assertEqual(result.guard_decision.action, GuardAction.APPLY)
        self.assertEqual(session.dialogue.turn, 2)
        self.assertEqual(session.products.context_lists(1).hard_rejected_asins, ("A",))

    def test_guard_switch_preserves_legacy_rule_state_but_enables_generalized_stop(self) -> None:
        disabled = self.build_rule_pipeline(guard_enabled=False)
        enabled = self.build_rule_pipeline(guard_enabled=True)
        disabled.reset("disabled", {})
        enabled.reset("enabled", {})

        disabled_turn = disabled.process_turn("disabled", "No more prefernces.", turn=1)
        enabled_turn = enabled.process_turn("enabled", "No more prefernces.", turn=1)

        self.assertEqual(disabled_turn.recognition.dialogue_act.value, "ambiguous")
        self.assertFalse(disabled.session("disabled").dialogue.no_more_preferences)
        self.assertEqual(enabled_turn.recognition.dialogue_act.value, "no_more_preferences")
        self.assertTrue(enabled.session("enabled").dialogue.no_more_preferences)

    def test_negated_explicit_asin_is_not_hard_rejected_when_guard_is_enabled(self) -> None:
        pipeline = self.build_rule_pipeline(guard_enabled=True)
        pipeline.reset("s", {})
        pipeline.record_shown("s", ["B012345678"], turn=1)

        result = pipeline.process_turn("s", "Don't reject B012345678.", turn=2)
        product_lists = pipeline.session("s").products.context_lists(1)

        self.assertEqual(result.recognition.dialogue_act.value, "ambiguous")
        self.assertEqual(result.recognition.explicit_rejected_asins, ())
        self.assertEqual(product_lists.hard_rejected_asins, ())
        self.assertEqual(product_lists.soft_demoted_asins, ())

    def test_ungrounded_llm_stop_is_blocked_without_state_or_history_mutation(self) -> None:
        pipeline = self.build_pipeline(
            guard_enabled=True,
            llm_response={
                "dialogue_act": "no_more_preferences",
                "category": None,
                "constraint_operations": [],
                "explicit_rejected_asins": [],
                "confidence": 0.99,
                "ambiguities": [],
            },
        )
        pipeline.reset("s", {})
        pipeline.record_shown("s", ["A"], turn=1)
        before = pipeline.session("s")

        result = pipeline.process_turn("s", "Please show me blue options.", turn=2)

        self.assertEqual(result.guard_decision.action, GuardAction.CLARIFY)
        self.assertEqual(result.guard_decision.reason_code, "no_more_preferences_not_grounded")
        self.assertEqual(pipeline.session("s"), before)


if __name__ == "__main__":
    unittest.main()
