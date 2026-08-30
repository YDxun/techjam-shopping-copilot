"""P3 config-as-data: tests validating CONFIG_PROFILES as the single source of truth.

- CONFIG_PROFILES covers all known profiles (rule_bm25/hybrid_dense/fingerprint_combo/
  text_rerank/reranker_model）；
- profile_overrides / requires_met logic is correct;
- validate_lut: a bad LUT (unknown config_id) reports problems; a good LUT passes.
"""
from __future__ import annotations

import unittest

from config.profiles import (
    profile_ids,
    profile_overrides,
    requires_met,
    validate_lut,
)


class ConfigProfilesTest(unittest.TestCase):
    def test_covers_all_profiles(self) -> None:
        self.assertEqual(
            sorted(profile_ids()),
            ["fingerprint_combo", "hybrid_dense", "reranker_model", "rule_bm25", "text_rerank"],
        )

    def test_profile_overrides_returns_mapping(self) -> None:
        ov = profile_overrides("rule_bm25")
        self.assertEqual(ov["retrieval_backend"], "bm25")
        self.assertEqual(ov["fingerprint"]["enable"], False)
        self.assertEqual(profile_overrides("not_exist"), {})

    def test_requires_met(self) -> None:
        # text_rerank requires llm + network
        self.assertTrue(
            requires_met("text_rerank", dense=True, llm=True, network=True, model=False)
        )
        self.assertFalse(
            requires_met("text_rerank", dense=True, llm=False, network=False, model=False)
        )
        # rule_bm25 has no requirements
        self.assertTrue(
            requires_met("rule_bm25", dense=False, llm=False, network=False, model=False)
        )
        # unknown profile
        self.assertFalse(
            requires_met("unknown", dense=True, llm=True, network=True, model=True)
        )

    def test_validate_lut(self) -> None:
        bad = {
            "environments": {
                "device=cuda;dense=yes;llm=no;network=no": {
                    "configs": [
                        {"config_id": "fingerprint_combo", "technical_score": 0.88},
                        {"config_id": "mystery_profile", "technical_score": 0.9},
                    ]
                }
            }
        }
        problems = validate_lut(bad)
        self.assertEqual(len(problems), 1)
        self.assertIn("mystery_profile", problems[0])
        good = {
            "environments": {
                "device=cuda;dense=yes;llm=no;network=no": {
                    "configs": [{"config_id": cid, "technical_score": 0.9} for cid in profile_ids()]
                }
            }
        }
        self.assertEqual(validate_lut(good), [])


if __name__ == "__main__":
    unittest.main()
