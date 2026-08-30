"""Non-dev-machine fallback-path tests (automation-control finalization).

Simulates cpu + dense=no (missing blair path) + llm=no (missing key) + network=no:
- RuntimeController.decide() falls back to retrieval=bm25 / intent=rule / clarify=rule /
  rerank=rule / reranker_model=no / strategy=bm25_rule；
- the agent can complete a full session without errors in the degraded environment (tiny catalog).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.capability_probe import CapabilityProfile
from agent.main_agent import Agent
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient

PRODUCTS = [
    {"parent_asin": "B000000001", "title": "Nike Men's Cotton Classic T-Shirt Black",
     "features": ["100% cotton"], "price": 19.99,
     "categories": ["Clothing, Shoes & Jewelry", "Men"], "details": {"Material": "Cotton"},
     "average_rating": 4.5, "rating_number": 100, "store": "Nike", "description": ["cotton tee"]},
    {"parent_asin": "B000000002", "title": "Columbia Men's Waterproof Rain Jacket",
     "features": ["waterproof"], "price": 59.99,
     "categories": ["Clothing, Shoes & Jewelry", "Men"], "details": {"Material": "Nylon"},
     "average_rating": 4.2, "rating_number": 50, "store": "Columbia", "description": ["jacket"]},
]


def degraded_env() -> EnvConfig:
    """Degraded environment: cpu + dense=no + llm=no + network=no (missing blair path -> probe
        dense=False)."""
    return EnvConfig.from_env(
        overrides={
            "skip_data_verify": True,
            "retrieval_backend": "auto",
            "blair_offline_embedding_path": "data/_missing_blair.npy",
            "llm": {"provider": "none"},
            "reranker_model_enabled": False,
        }
    )


class EnvFallbackTest(unittest.TestCase):
    def test_runtime_controller_falls_back_to_rule(self) -> None:
        env = degraded_env()
        profile = CapabilityProfile(
            device="cpu", dense_available=False, llm_state="disabled",
            llm_provider="none", network_available=False,
        )
        d = RuntimeController(env, profile).decide()
        self.assertEqual(d.retrieval_backend, "bm25")
        self.assertFalse(d.use_dense)
        self.assertFalse(d.use_llm_intent)
        self.assertFalse(d.use_llm_clarify)
        self.assertFalse(d.use_llm_rerank)
        self.assertFalse(d.use_reranker_model)
        self.assertEqual(d.strategy, "bm25_rule")

    def test_agent_runs_a_session_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.jsonl"
            with catalog.open("w", encoding="utf-8") as f:
                for p in PRODUCTS:
                    f.write(json.dumps(p) + "\n")
            env = degraded_env()
            agent = Agent(catalog_path=str(catalog), env=env, llm_client=DisabledLLMClient())
            # auto resolves to bm25 in the degraded environment
            self.assertEqual(agent.decisions.retrieval_backend, "bm25")
            agent.reset("fb_1", {})
            resp = agent.respond("fb_1", "I need cotton", turn=1, top_k=10)
            self.assertIsInstance(resp, dict)
            self.assertIsInstance(resp.get("message"), str)
            self.assertIsInstance(resp.get("recommendations"), list)
            # a follow-up turn is still stable
            resp2 = agent.respond(
                "fb_1", "For that, what matters is: waterproof.", turn=2, top_k=10
            )
            self.assertIsInstance(resp2.get("recommendations"), list)

    def test_agent_decisions_rule_in_degraded_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.jsonl"
            with catalog.open("w", encoding="utf-8") as f:
                for p in PRODUCTS:
                    f.write(json.dumps(p) + "\n")
            env = degraded_env()
            agent = Agent(catalog_path=str(catalog), env=env, llm_client=DisabledLLMClient())
            self.assertFalse(agent.decisions.use_llm_intent)
            self.assertFalse(agent.decisions.use_dense)
            self.assertEqual(agent.decisions.strategy, "bm25_rule")


if __name__ == "__main__":
    unittest.main()
