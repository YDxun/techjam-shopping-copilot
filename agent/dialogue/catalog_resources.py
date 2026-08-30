"""Immutable catalog-derived resources shared by dialogue pipelines."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from agent.dialogue.catalog_attributes import CatalogAttributeCache, RuleVocabularyExtractor
from agent.dialogue.catalog_signals import CatalogQuestionSignals

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogueCatalogResources:
    """Derived catalog structures safe to reuse across independent dialogues."""

    catalog_signals: CatalogQuestionSignals
    attribute_cache: CatalogAttributeCache | None

    @classmethod
    def from_products(
        cls,
        products: Iterable[dict],
        *,
        include_attribute_cache: bool,
    ) -> "DialogueCatalogResources":
        """Materialize products once and derive optional candidate-signal data."""
        product_rows = tuple(products)
        catalog_signals = CatalogQuestionSignals.from_products(product_rows)
        attribute_cache = None
        if include_attribute_cache:
            try:
                attribute_cache = CatalogAttributeCache.from_products(
                    product_rows, RuleVocabularyExtractor()
                )
            except Exception:
                logger.exception(
                    "[dialogue] dynamic catalog setup failed; using static question policy"
                )
        return cls(catalog_signals=catalog_signals, attribute_cache=attribute_cache)
