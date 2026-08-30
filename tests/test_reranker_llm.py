from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.main_agent import Agent
from agent.reranker import Reranker
from config.env_config import EnvConfig
from llm.base import (
    DisabledLLMClient,
    LLMErrorCategory,
    LLMResult,
    LLMState,
    LLMStatus,
    LLMUsage,
)


class FakeRetriever:
    def __init__(self, products: dict[str, dict]) -> None:
        self.products = products

    def product(self, asin: str) -> dict | None:
        return self.products.get(asin)

    def text_lower(self, asin: str) -> str:
        return " ".join(str(value) for value in self.products[asin].values()).lower()


class FakeAvailableClient:
    def __init__(self, result: LLMResult, provider: str = "deepseek") -> None:
        self._status = LLMStatus(LLMState.AVAILABLE, provider, result.model)
        self.chat = Mock(return_value=result)

    @property
    def status(self) -> LLMStatus:
        return self._status


class FakeUnavailableClient:
    def __init__(self) -> None:
        self._status = LLMStatus(LLMState.UNAVAILABLE, "deepseek", "deepseek-chat")
        self.chat = Mock()

    @property
    def status(self) -> LLMStatus:
        return self._status


class FalseyLLMClient:
    def __init__(self) -> None:
        self._status = LLMStatus(LLMState.AVAILABLE, "deepseek", "deepseek-chat")

    def __bool__(self) -> bool:
        return False

    @property
    def status(self) -> LLMStatus:
        return self._status

    @property
    def cumulative_usage(self) -> LLMUsage:
        return LLMUsage()

    def initialize(self) -> LLMStatus:
        return self._status

    def chat(
        self, messages: object, *, temperature: float | None = None, max_tokens: int | None = None
    ) -> LLMResult:
        return LLMResult(True, "deepseek", "deepseek-chat", content='["A", "B"]')


def env_with_rerank(enabled: bool = True, candidates: int = 12) -> object:
    return SimpleNamespace(
        llm=SimpleNamespace(
            rerank_enabled=enabled,
            rerank_candidates=candidates,
            rerank_backend="chat",  # these tests cover the legacy chat LLM semantic-rerank path
        )
    )


def make_rule_case() -> tuple[FakeRetriever, list[dict], object, object, list[str]]:
    products = {
        "A": {
            "parent_asin": "A",
            "title": "Alpha",
            "categories": ["Shoes"],
            "features": ["light"],
            "rating_number": 0,
            "average_rating": 0,
        },
        "B": {
            "parent_asin": "B",
            "title": "Beta",
            "categories": ["Shoes"],
            "features": ["wide"],
            "rating_number": 0,
            "average_rating": 0,
        },
    }
    retriever = FakeRetriever(products)
    candidates = [{"parent_asin": "A", "rrf": 2.0}, {"parent_asin": "B", "rrf": 1.0}]
    state = SimpleNamespace(hard=[], soft=[], active=[], user_profile={})
    route = SimpleNamespace(category_tokens=[])
    return retriever, candidates, state, route, ["A", "B"]


def user_payload(client: FakeAvailableClient) -> dict:
    messages = client.chat.call_args.args[0]
    content = next(message["content"] for message in messages if message["role"] == "user")
    return json.loads(content)


class RerankerLLMTest(unittest.TestCase):
    def test_disabled_client_preserves_rule_order(self) -> None:
        retriever, candidates, state, route, rule_order = make_rule_case()
        reranker = Reranker(env=env_with_rerank(True), llm_client=DisabledLLMClient())
        actual = reranker.rerank(
            retriever, candidates, state, route, top_k=10, mode="probe", use_llm_rerank=True
        )
        self.assertEqual(actual, rule_order)

    def test_failed_client_preserves_rule_order_and_reports_returned_usage(self) -> None:
        retriever, candidates, state, route, rule_order = make_rule_case()
        client = FakeAvailableClient(
            result=LLMResult(
                success=False,
                provider="deepseek",
                model="deepseek-chat",
                usage=LLMUsage(prompt_tokens=4, completion_tokens=0),
                error_category=LLMErrorCategory.TIMEOUT,
            )
        )
        reranker = Reranker(env=env_with_rerank(True), llm_client=client)
        actual = reranker.rerank(
            retriever, candidates, state, route, top_k=10, mode="probe", use_llm_rerank=True
        )
        self.assertEqual(actual, rule_order)
        self.assertEqual(reranker.last_usage, {"prompt_tokens": 4, "completion_tokens": 0})

    def test_skipped_paths_do_not_call_chat_and_zero_usage(self) -> None:
        retriever, candidates, state, route, rule_order = make_rule_case()
        cases = (
            (
                env_with_rerank(False),
                FakeAvailableClient(LLMResult(True, "deepseek", "model")),
                candidates,
            ),
            (env_with_rerank(True), FakeUnavailableClient(), candidates),
            (
                env_with_rerank(True),
                FakeAvailableClient(LLMResult(True, "deepseek", "model")),
                candidates[:1],
            ),
        )
        for env, client, case_candidates in cases:
            with self.subTest(client=type(client).__name__, candidates=len(case_candidates)):
                reranker = Reranker(env=env, llm_client=client)
                reranker.last_usage = {"prompt_tokens": 99, "completion_tokens": 88}
                actual = reranker.rerank(
                    retriever,
                    case_candidates,
                    state,
                    route,
                    top_k=10,
                    mode="probe",
                    use_llm_rerank=True,
                )
                self.assertEqual(actual, rule_order[: len(case_candidates)])
                client.chat.assert_not_called()
                self.assertEqual(reranker.last_usage, {"prompt_tokens": 0, "completion_tokens": 0})

    def test_payload_is_compact_and_excludes_non_ranking_context(self) -> None:
        retriever, candidates, state, route, _ = make_rule_case()
        retriever.products["A"].update(
            {
                "title": "T" * 300,
                "categories": ["C" * 200, "D" * 200],
                "features": [" first\n", " second\t", "F" * 900],
                "details": {"secret": "must not be sent"},
                "description": "conversation history must not be sent",
            }
        )
        state.active = [SimpleNamespace(value="constraint " + "X" * 900)]
        state.user_profile = {"private": "profile must not be sent"}
        client = FakeAvailableClient(
            LLMResult(True, "deepseek", "deepseek-chat", content='["B", "A"]')
        )
        actual = Reranker(env=env_with_rerank(True), llm_client=client).rerank(
            retriever,
            candidates,
            state,
            route,
            top_k=10,
            mode="probe",
            use_llm_rerank=True,
        )
        payload = user_payload(client)
        self.assertEqual(actual, ["B", "A"])
        self.assertEqual(set(payload), {"constraints", "candidates"})
        self.assertEqual(payload["constraints"], ("constraint " + "X" * 900)[:800])
        self.assertEqual(len(payload["candidates"]), 2)
        first = payload["candidates"][0]
        self.assertEqual(set(first), {"parent_asin", "title", "categories", "features"})
        self.assertEqual(first["title"], "T" * 240)
        self.assertEqual(first["categories"], (("C" * 200) + " " + ("D" * 200))[:240])
        self.assertEqual(first["features"], ("first second " + "F" * 900)[:800])
        self.assertNotIn("details", json.dumps(payload))
        self.assertNotIn("profile", json.dumps(payload))
        self.assertNotIn("conversation history", json.dumps(payload))

    def test_valid_response_forms_filter_unknown_duplicates_and_append_rule_order(self) -> None:
        retriever, candidates, state, route, _ = make_rule_case()
        candidates.extend(
            [
                {"parent_asin": "C", "rrf": 0.5},
                {"parent_asin": "D", "rrf": 0.25},
            ]
        )
        retriever.products.update(
            {
                "C": {
                    "parent_asin": "C",
                    "title": "Gamma",
                    "categories": [],
                    "features": [],
                    "rating_number": 0,
                    "average_rating": 0,
                },
                "D": {
                    "parent_asin": "D",
                    "title": "Delta",
                    "categories": [],
                    "features": [],
                    "rating_number": 0,
                    "average_rating": 0,
                },
            }
        )
        forms = (
            '{"ranked_parent_asins": ["B", "unknown", "B"]}',
            '["B", "unknown", "B"]',
            '```json\n{"ranked_parent_asins": ["B", "unknown", "B"]}\n```',
        )
        for provider in ("deepseek", "openai"):
            for content in forms:
                with self.subTest(provider=provider, content=content):
                    client = FakeAvailableClient(
                        LLMResult(
                            True,
                            provider,
                            f"{provider}-model",
                            content=content,
                            usage=LLMUsage(prompt_tokens=11, completion_tokens=7),
                        ),
                        provider=provider,
                    )
                    reranker = Reranker(env=env_with_rerank(True, candidates=2), llm_client=client)
                    actual = reranker.rerank(
                        retriever,
                        candidates,
                        state,
                        route,
                        top_k=10,
                        mode="probe",
                        use_llm_rerank=True,
                    )
                    self.assertEqual(actual, ["B", "A", "C", "D"])
                    self.assertEqual(
                        reranker.last_usage, {"prompt_tokens": 11, "completion_tokens": 7}
                    )
                    self.assertEqual(
                        [item["parent_asin"] for item in user_payload(client)["candidates"]],
                        ["A", "B"],
                    )

    def test_empty_or_malformed_response_preserves_complete_rule_order(self) -> None:
        retriever, candidates, state, route, rule_order = make_rule_case()
        for content in ("", "not json", "[]", '{"ranked_parent_asins": []}', '{"wrong": ["B"]}'):
            with self.subTest(content=content):
                client = FakeAvailableClient(LLMResult(True, "deepseek", "model", content=content))
                actual = Reranker(env=env_with_rerank(True), llm_client=client).rerank(
                    retriever, candidates, state, route, top_k=10, mode="probe"
                )
                self.assertEqual(actual, rule_order)

    def test_explicit_client_is_passed_to_reranker(self) -> None:
        client = object()
        env = EnvConfig.from_env(overrides={"skip_data_verify": True}, environ={})
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                json.dumps({"parent_asin": "A", "title": "Alpha"}) + "\n", encoding="utf-8"
            )
            with patch("agent.main_agent.Reranker") as reranker_class:
                agent = Agent(catalog_path=catalog, env=env, llm_client=client)
        agent.retriever.close()
        self.assertIs(reranker_class.call_args.kwargs["llm_client"], client)

    def test_falsey_injected_client_retains_identity(self) -> None:
        client = FalseyLLMClient()
        reranker = Reranker(env=env_with_rerank(True), llm_client=client)
        self.assertIs(reranker.llm_client, client)

        env = EnvConfig.from_env(overrides={"skip_data_verify": True}, environ={})
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                json.dumps({"parent_asin": "A", "title": "Alpha"}) + "\n", encoding="utf-8"
            )
            agent = Agent(catalog_path=catalog, env=env, llm_client=client)
        agent.retriever.close()
        self.assertIs(agent.llm_client, client)
        self.assertIs(agent.reranker.llm_client, client)

    def test_direct_construction_uses_disabled_clients_without_factory(self) -> None:
        env = EnvConfig.from_env(overrides={"skip_data_verify": True}, environ={})
        with patch("llm.factory.create_llm_client") as factory:
            reranker = Reranker(env=env)
            with tempfile.TemporaryDirectory() as directory:
                catalog = Path(directory) / "catalog.jsonl"
                catalog.write_text(
                    json.dumps({"parent_asin": "A", "title": "Alpha"}) + "\n", encoding="utf-8"
                )
                agent = Agent(catalog_path=catalog, env=env)
                agent.retriever.close()
        self.assertIsInstance(reranker.llm_client, DisabledLLMClient)
        self.assertIsInstance(agent.reranker.llm_client, DisabledLLMClient)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
