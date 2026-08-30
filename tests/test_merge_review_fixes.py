from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from config.env_config import EnvConfig
from utils import field_mapping


class MergeReviewFixesTest(unittest.TestCase):
    def test_optional_dependencies_do_not_block_rule_only_startup(self) -> None:
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.', 1)[0] in {'httpx', 'numpy'}:
                        raise ModuleNotFoundError(f'blocked optional dependency: {fullname}')
                    return None

            for name in tuple(sys.modules):
                if name.split('.', 1)[0] in {'httpx', 'numpy'}:
                    del sys.modules[name]
            sys.meta_path.insert(0, BlockOptional())

            from agent.main_agent import Agent
            from config.env_config import EnvConfig
            from llm.base import DisabledLLMClient

            class Retriever:
                def iter_products(self): return ()
                def close(self): return None

            env = EnvConfig.from_env(
                overrides={'skip_data_verify': True, 'llm': {'provider': 'none'}}, environ={}
            )
            agent = Agent(env=env, llm_client=DisabledLLMClient(), retriever=Retriever())
            assert agent.dialogue.recognizer.mode == 'rule_only'
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_documented_aliases_and_explicit_overrides_are_canonical(self) -> None:
        config = EnvConfig.from_env(
            environ={
                "ASSET_CATEGORY_EXPAND": "false",
                "ASSET_PARAPHRASE": "1",
                "ASSET_VOCAB_EXPAND": "yes",
                "COMBO_BONUS_WEIGHT": "0.37",
                "COMBO_FINGERPRINT_BONUS_UNIQUE": "1.4",
                "COMBO_FINGERPRINT_BONUS_TEN": "0.6",
                "COMBO_FINGERPRINT_BONUS_FIFTY": "0.25",
            },
            overrides={
                "asset_paraphrase": False,
                "rerank_weights": {"combo": 0.19},
                "fingerprint": {"bonus_ten": 0.8},
            },
        )
        self.assertFalse(config.asset_category_expand)
        self.assertFalse(config.asset_paraphrase)
        self.assertTrue(config.asset_vocab_expand)
        self.assertEqual(config.rerank_weights["combo"], 0.19)
        self.assertEqual(config.fingerprint.bonus_unique, 1.4)
        self.assertEqual(config.fingerprint.bonus_ten, 0.8)
        self.assertEqual(config.fingerprint.bonus_fifty, 0.25)

    def test_field_mapping_and_vocab_caches_are_selection_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps({"default": {"marker": "first"}}), encoding="utf-8")
            second.write_text(json.dumps({"default": {"marker": "second"}}), encoding="utf-8")
            self.assertEqual(field_mapping.load(first)["default"]["marker"], "first")
            self.assertEqual(field_mapping.load(second)["default"]["marker"], "second")

            base_vocab = root / "base_vocab.json"
            asset_vocab = root / "asset_vocab.json"
            base_vocab.write_text(json.dumps({"dictionaries": {"material": {}}}), encoding="utf-8")
            asset_vocab.write_text(
                json.dumps({"dictionaries": {"material": {"cotton": {"synonyms": ["base"]}}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                field_mapping.expand_with_vocab(
                    "material", "cotton", use_asset_vocab=True, asset_path=asset_vocab
                ),
                ["cotton", "base"],
            )
            self.assertEqual(
                field_mapping.expand_with_vocab(
                    "material", "cotton", use_asset_vocab=False, vocab_path=base_vocab
                ),
                ["cotton"],
            )
