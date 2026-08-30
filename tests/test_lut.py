"""Step 3：配置-环境-性能 LUT 单元测试（RuntimeController 按 env 选配置）。

- recommend()：同环境下按 technical_score 选最优（延迟/内存预算过滤）；
- 环境不在表内 / LUT 缺失 → None（回退默认策略）；
- RuntimeController.decide() 设置 strategy_lut（数据驱动启动默认）。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.capability_probe import CapabilityProfile
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from utils import lut as lut_utils

FAKE_LUT = {
    "environments": {
        "device=cuda;dense=yes;llm=no;network=no": {
            "configs": [
                {"config_id": "rule_bm25", "technical_score": 0.880, "latency_ms_per_turn": 80},
                {"config_id": "hybrid_dense", "technical_score": 0.879, "latency_ms_per_turn": 90},
                {
                    "config_id": "fingerprint_combo",
                    "technical_score": 0.881,
                    "latency_ms_per_turn": 95,
                },
            ]
        }
    }
}


class LutRecommendTest(unittest.TestCase):
    def test_recommend_picks_highest_score(self) -> None:
        rec = lut_utils.recommend("device=cuda;dense=yes;llm=no;network=no", lut=FAKE_LUT)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["config_id"], "fingerprint_combo")

    def test_recommend_respects_latency_budget(self) -> None:
        rec = lut_utils.recommend(
            "device=cuda;dense=yes;llm=no;network=no", lut=FAKE_LUT, max_latency_ms=85
        )
        self.assertEqual(rec["config_id"], "rule_bm25")

    def test_recommend_none_when_env_absent(self) -> None:
        rec = lut_utils.recommend("device=cpu;dense=no;llm=no;network=no", lut=FAKE_LUT)
        self.assertIsNone(rec)


class RuntimeControllerLutTest(unittest.TestCase):
    def test_decide_sets_strategy_lut(self) -> None:
        env = EnvConfig.from_env()
        prof = CapabilityProfile(device="cuda", dense_available=True, llm_state="disabled")
        with patch(
            "agent.runtime_controller.lut_utils.recommend",
            return_value=FAKE_LUT["environments"]["device=cuda;dense=yes;llm=no;network=no"][
                "configs"
            ][2],
        ):
            d = RuntimeController(env, prof).decide()
        self.assertEqual(d.strategy_lut, "fingerprint_combo")

    def test_decide_fallback_when_lut_missing(self) -> None:
        env = EnvConfig.from_env()
        prof = CapabilityProfile(device="cuda", dense_available=True, llm_state="disabled")
        with patch("agent.runtime_controller.lut_utils.recommend", return_value=None):
            d = RuntimeController(env, prof).decide()
        self.assertIsNone(d.strategy_lut)


if __name__ == "__main__":
    unittest.main()
