from __future__ import annotations

import unittest

from utils.data_assets import DataAssets


class NormalizationVocabularyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assets = DataAssets(
            vocab={
                "dictionaries": {
                    "category_product_type": {
                        "jacket": {
                            "synonyms": ["jacket", "windbreaker"],
                            "product_count": 20,
                        }
                    },
                    "color": {
                        "navy": {
                            "synonyms": ["navy", "navy blue"],
                            "product_count": 15,
                        },
                        "comfortable": {
                            "synonyms": ["comfortable"],
                            "product_count": 0,
                            "sources": ["public_set"],
                        },
                    },
                    "feature": {
                        "waterproof": {
                            "synonyms": ["waterproof"],
                            "product_count": 99,
                        }
                    },
                }
            }
        )

    def test_catalog_supported_values_exclude_public_only_noise(self) -> None:
        try:
            vocabulary = self.assets.normalization_vocabulary(min_product_count=1)
        except AttributeError as error:
            self.fail(f"normalization vocabulary is unavailable: {error}")

        self.assertEqual(vocabulary.allowed_values["category"], ("jacket",))
        self.assertEqual(vocabulary.allowed_values["color"], ("navy",))
        self.assertNotIn("feature", vocabulary.allowed_values)

    def test_synonyms_are_canonicalized_within_their_attribute(self) -> None:
        try:
            vocabulary = self.assets.normalization_vocabulary(min_product_count=1)
        except AttributeError as error:
            self.fail(f"normalization vocabulary is unavailable: {error}")

        self.assertEqual(vocabulary.canonicalize("category", "windbreaker"), "jacket")
        self.assertEqual(vocabulary.canonicalize("color", "navy blue"), "navy")
        self.assertEqual(vocabulary.canonicalize("color", "unknown shade"), "unknown shade")


if __name__ == "__main__":
    unittest.main()
