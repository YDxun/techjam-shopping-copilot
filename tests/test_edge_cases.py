"""边界测试（竞赛交付物）：空消息 / 最后一轮 / 并发会话隔离 / override 流程 / 推荐过滤。

用微型 catalog（5 商品）构建 Agent，避免全量 50k 建索引耗时；不改评估器、不改现有测试。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.main_agent import Agent
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient

PRODUCTS = [
    {"parent_asin": "B000000001", "title": "Nike Men's Cotton Classic T-Shirt Black",
     "features": ["100% cotton", "black"], "price": 19.99, "categories": ["Clothing, Shoes & Jewelry", "Men"],
     "details": {"Material": "Cotton"}, "average_rating": 4.5, "rating_number": 100, "store": "Nike",
     "description": ["Comfortable cotton tee"]},
    {"parent_asin": "B000000002", "title": "Columbia Men's Waterproof Rain Jacket",
     "features": ["waterproof", "nylon"], "price": 59.99, "categories": ["Clothing, Shoes & Jewelry", "Men"],
     "details": {"Material": "Nylon"}, "average_rating": 4.2, "rating_number": 50, "store": "Columbia",
     "description": ["Waterproof shell"]},
    {"parent_asin": "B000000003", "title": "Leather Wallet for Men Slim",
     "features": ["leather", "slim"], "price": 29.99, "categories": ["Clothing, Shoes & Jewelry", "Men"],
     "details": {"Material": "Leather"}, "average_rating": 4.0, "rating_number": 80, "store": "Generic",
     "description": ["Full grain leather"]},
    {"parent_asin": "B000000004", "title": "Silk Scarf Women Luxury",
     "features": ["silk", "soft"], "price": 39.99, "categories": ["Clothing, Shoes & Jewelry", "Women"],
     "details": {"Material": "Silk"}, "average_rating": 4.8, "rating_number": 30, "store": "Generic",
     "description": ["Elegant silk"]},
    {"parent_asin": "B000000005", "title": "Running Sneakers Men Lightweight",
     "features": ["running", "lightweight"], "price": 49.99, "categories": ["Clothing, Shoes & Jewelry", "Men"],
     "details": {"Material": "Mesh"}, "average_rating": 4.6, "rating_number": 200, "store": "Generic",
     "description": ["Athletic shoes"]},
]


class EdgeCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        catalog = Path(cls.tmp.name) / "catalog.jsonl"
        with catalog.open("w", encoding="utf-8") as f:
            for p in PRODUCTS:
                f.write(json.dumps(p) + "\n")
        env = EnvConfig.from_env(overrides={"skip_data_verify": True, "retrieval_backend": "bm25"})
        cls.agent = Agent(catalog_path=str(catalog), env=env, llm_client=DisabledLLMClient())
        cls.catalog_ids = {p["parent_asin"] for p in PRODUCTS}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def respond(self, sid: str, msg: str, turn: int = 1) -> dict:
        return self.agent.respond(sid, msg, turn, 10)

    def test_empty_message_does_not_crash(self) -> None:
        self.agent.reset("e_empty", {})
        resp = self.respond("e_empty", "", turn=1)
        self.assertIsInstance(resp, dict)
        self.assertIsInstance(resp.get("message"), str)
        self.assertIn("recommendations", resp)

    def test_last_turn_returns_valid(self) -> None:
        self.agent.reset("e_last", {})
        resp = self.respond("e_last", "I need waterproof", turn=10)
        self.assertIsInstance(resp, dict)
        self.assertIsInstance(resp.get("recommendations"), list)

    def test_concurrent_sessions_do_not_cross_talk(self) -> None:
        self.agent.reset("e_a", {})
        self.agent.reset("e_b", {})
        ra = self.respond("e_a", "I need cotton", turn=1)
        rb = self.respond("e_b", "I need leather", turn=1)
        asins_a = [r["parent_asin"] for r in ra["recommendations"]]
        asins_b = [r["parent_asin"] for r in rb["recommendations"]]
        # 会话 A 应优先推荐 cotton 商品，会话 B 应优先 leather（互不串扰）
        self.assertEqual(asins_a[0], "B000000001")  # cotton tee
        self.assertEqual(asins_b[0], "B000000003")  # leather wallet

    def test_override_flow(self) -> None:
        self.agent.reset("e_ov", {})
        r1 = self.respond("e_ov", "I'm looking for men's clothing. I need cotton", turn=1)
        self.assertEqual([r["parent_asin"] for r in r1["recommendations"]][0], "B000000001")
        r2 = self.respond(
            "e_ov",
            "Actually, ignore my earlier preference. What I need is: leather.",
            turn=2,
        )
        recs = [r["parent_asin"] for r in r2["recommendations"]]
        # override 后新意图（leather）优先，cotton 旧偏好不再主导
        self.assertIn("B000000003", recs[:3])

    def test_recommendations_unique_valid_and_bounded(self) -> None:
        self.agent.reset("e_uniq", {})
        for turn in range(1, 4):
            resp = self.respond("e_uniq", "I need leather", turn=turn)
            recs = [r["parent_asin"] for r in resp["recommendations"]]
            self.assertLessEqual(len(recs), 10)
            self.assertEqual(len(recs), len(set(recs)))  # 去重
            self.assertTrue(all(a in self.catalog_ids for a in recs))  # 目录内有效

    def test_invalid_duplicate_payload_tolerated(self) -> None:
        # Agent 侧不产出重复/无效；评估器还会再过滤。这里验证 respond 结构稳定、无异常
        self.agent.reset("e_dup", {})
        resp = self.respond("e_dup", "cotton", turn=1)
        payload = resp.get("recommendations")
        self.assertIsInstance(payload, list)
        for item in payload:
            self.assertIsInstance(item, dict)
            self.assertIn("parent_asin", item)


if __name__ == "__main__":
    unittest.main()
