from __future__ import annotations

import unittest
from collections.abc import Iterable
from types import MappingProxyType

from agent.dialogue.catalog_attributes import (
    AttributeProfile,
    CatalogAttributeCache,
    RuleVocabularyExtractor,
    load_vocabulary,
)


class PreparingExtractor:
    @property
    def vocabulary_version(self) -> str:
        return "test-v1"

    def prepare(self, products: Iterable[dict[str, object]]) -> "PreparingExtractor":
        self.prepared_asins = tuple(str(product["parent_asin"]) for product in products)
        return self

    def extract(self, product: dict[str, object]) -> AttributeProfile:
        asin = str(product["parent_asin"])
        return AttributeProfile(
            parent_asin=asin,
            values={"material": {"cotton"}},
            confidence={"material": 1.0},
            sources={"material": ("details",)},
        )


class CatalogAttributesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = load_vocabulary()

    def test_structured_fields_beat_free_text_and_synonyms_collapse(self) -> None:
        product = {
            "parent_asin": "A",
            "title": "Soft cotton blend running top",
            "features": ["95% cotton with 5% spandex"],
            "details": {"Material": "100% Cotton", "Color": "Jet Black"},
            "description": ["polyester-like appearance"],
            "categories": ["Women", "Tops"],
            "store": "Example Brand",
            "price": 29.99,
        }

        profile = RuleVocabularyExtractor(self.vocabulary).extract(product)

        self.assertEqual(profile.values["material"], frozenset({"cotton"}))
        self.assertEqual(profile.values["color"], frozenset({"black"}))
        self.assertGreater(profile.confidence["material"], profile.confidence["use_case"])

    def test_missing_attribute_stays_missing(self) -> None:
        profile = RuleVocabularyExtractor(self.vocabulary).extract(
            {"parent_asin": "B", "title": "Generic item", "features": [], "details": {}}
        )

        self.assertEqual(profile.values["material"], frozenset())

    def test_size_values_keep_apparel_and_shoe_context_distinct(self) -> None:
        extractor = RuleVocabularyExtractor(self.vocabulary)
        apparel = extractor.extract(
            {
                "parent_asin": "shirt",
                "title": "Cotton shirt",
                "categories": ["Women", "Tops"],
                "details": {"Size": "Medium"},
            }
        )
        shoes = extractor.extract(
            {
                "parent_asin": "shoe",
                "title": "Running sneakers",
                "categories": ["Women", "Shoes"],
                "details": {"Size": "8.5"},
            }
        )

        self.assertEqual(apparel.values["size"], frozenset({"apparel_size:m"}))
        self.assertEqual(shoes.values["size"], frozenset({"shoe_size:8.5"}))

    def test_prepare_assigns_price_quartiles_within_each_category(self) -> None:
        products = [
            {
                "parent_asin": asin,
                "categories": ["Women", "Shoes"],
                "price": price,
            }
            for asin, price in (("s1", 10), ("s2", 20), ("s3", 30), ("s4", 40))
        ] + [
            {
                "parent_asin": asin,
                "categories": ["Women", "Tops"],
                "price": price,
            }
            for asin, price in (("t1", 100), ("t2", 200), ("t3", 300), ("t4", 400))
        ]

        cache = CatalogAttributeCache.from_products(
            products, RuleVocabularyExtractor(self.vocabulary)
        )

        self.assertEqual(cache.for_asin("s1").values["budget"], frozenset({"budget_low"}))
        self.assertEqual(cache.for_asin("s2").values["budget"], frozenset({"budget_mid"}))
        self.assertEqual(cache.for_asin("s3").values["budget"], frozenset({"budget_mid"}))
        self.assertEqual(cache.for_asin("s4").values["budget"], frozenset({"budget_high"}))
        self.assertEqual(cache.for_asin("t1").values["budget"], frozenset({"budget_low"}))

    def test_brand_filter_and_feature_vocabulary_reject_generic_or_free_text(self) -> None:
        extractor = RuleVocabularyExtractor(self.vocabulary)
        generic = extractor.extract(
            {
                "parent_asin": "generic",
                "title": "Breathable lightweight running shoe",
                "features": ["Breathable mesh", "Lightweight construction", "One-off claim"],
                "store": "Generic",
            }
        )
        branded = extractor.extract(
            {
                "parent_asin": "brand",
                "details": {"Brand": "Northwind"},
                "store": "Generic",
            }
        )

        self.assertEqual(generic.values["brand"], frozenset())
        self.assertEqual(generic.values["feature"], frozenset({"breathable", "lightweight"}))
        self.assertEqual(branded.values["brand"], frozenset({"northwind"}))

    def test_cache_prepares_a_single_materialized_pass_and_has_stable_fingerprint(self) -> None:
        products = [
            {"parent_asin": "B"},
            {"parent_asin": "A"},
        ]
        extractor = PreparingExtractor()

        cache = CatalogAttributeCache.from_products((product for product in products), extractor)
        reordered = CatalogAttributeCache.from_products(reversed(products), PreparingExtractor())

        self.assertEqual(extractor.prepared_asins, ("A", "B"))
        self.assertEqual(cache.for_asin("A").parent_asin, "A")
        self.assertIsNone(cache.for_asin("missing"))
        self.assertEqual(cache.catalog_fingerprint, reordered.catalog_fingerprint)

    def test_cache_skips_blank_asins_and_selects_duplicate_row_deterministically(self) -> None:
        products = [
            {
                "parent_asin": "A",
                "categories": ["Shoes"],
                "details": {"Material": "Cotton"},
                "price": 10,
            },
            {
                "parent_asin": "A",
                "categories": ["Shoes"],
                "details": {"Material": "Leather"},
                "price": 1000,
            },
            {"parent_asin": "B", "categories": ["Shoes"], "price": 20},
            {"parent_asin": None, "categories": ["Shoes"], "price": 0},
            {"parent_asin": "  ", "categories": ["Shoes"], "price": 10000},
        ]

        cache = CatalogAttributeCache.from_products(
            products, RuleVocabularyExtractor(self.vocabulary)
        )
        reversed_cache = CatalogAttributeCache.from_products(
            reversed(products), RuleVocabularyExtractor(self.vocabulary)
        )

        self.assertEqual(set(cache.profiles), {"A", "B"})
        self.assertEqual(cache.for_asin("A").values["material"], frozenset({"cotton"}))
        self.assertEqual(cache.for_asin("A").values["budget"], frozenset({"budget_low"}))
        self.assertEqual(cache.for_asin("B").values["budget"], frozenset({"budget_high"}))
        self.assertEqual(cache.profiles, reversed_cache.profiles)
        self.assertEqual(cache.catalog_fingerprint, reversed_cache.catalog_fingerprint)

    def test_numeric_footwear_text_requires_explicit_size_context(self) -> None:
        extractor = RuleVocabularyExtractor(self.vocabulary)
        without_context = extractor.extract(
            {"parent_asin": "pack", "title": "2 pack running shoes", "categories": ["Shoes"]}
        )
        with_context = extractor.extract(
            {
                "parent_asin": "sized",
                "title": "Running shoes, size 8.5",
                "categories": ["Shoes"],
            }
        )

        self.assertEqual(without_context.values["size"], frozenset())
        self.assertEqual(with_context.values["size"], frozenset({"shoe_size:8.5"}))

    def test_bootcut_tokens_do_not_imply_footwear(self) -> None:
        profile = RuleVocabularyExtractor(self.vocabulary).extract(
            {
                "parent_asin": "jeans",
                "title": "Bootcut jeans size 8",
                "categories": ["Women", "Jeans"],
            }
        )

        self.assertEqual(profile.values["size"], frozenset({"apparel_size:numeric"}))

    def test_profiles_and_cache_defensively_freeze_nested_mappings(self) -> None:
        values = {"material": {"cotton"}}
        confidence = {"material": 0.9}
        sources = {"material": ["details"]}
        profile = AttributeProfile("A", values, confidence, sources)
        cache = CatalogAttributeCache(
            profiles={"A": profile},
            vocabulary_version="test-v1",
            catalog_fingerprint="fingerprint",
        )
        values["material"].add("wool")
        confidence["material"] = 0.1
        sources["material"].append("description")

        self.assertIsInstance(profile.values, MappingProxyType)
        self.assertEqual(profile.values["material"], frozenset({"cotton"}))
        self.assertEqual(profile.confidence["material"], 0.9)
        self.assertEqual(profile.sources["material"], ("details",))
        with self.assertRaises(TypeError):
            profile.values["material"] = frozenset()  # type: ignore[index]
        with self.assertRaises(TypeError):
            cache.profiles["B"] = profile  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
