"""P1 phase-level circuit-breaker tests.

- PhaseCircuitBreaker: trip on consecutive failures / reset on success / open state / threshold;
- retriever dense breaker integration: consecutive _route_dense failures -> _dense becomes
(None,None) (hybrid -> bm25).
"""
from __future__ import annotations

import unittest

from utils.circuit_breaker import PhaseCircuitBreaker


class CircuitBreakerTest(unittest.TestCase):
    def test_opens_after_threshold(self) -> None:
        b = PhaseCircuitBreaker("dense", failure_threshold=2)
        self.assertFalse(b.open)
        self.assertFalse(b.record_failure("err1"))  # 1st failure: not tripped
        self.assertFalse(b.open)
        self.assertTrue(b.record_failure("err2"))  # 2nd failure: tripped
        self.assertTrue(b.open)
        self.assertIn("dense", b.trip_reason)
        self.assertEqual(b.trip_count, 1)

    def test_success_resets(self) -> None:
        b = PhaseCircuitBreaker("reranker", failure_threshold=3)
        b.record_failure("a")
        b.record_failure("b")
        b.record_success()  # reset the streak
        self.assertFalse(b.open)
        b.record_failure("c")
        self.assertFalse(b.open)  # threshold 3; only 1 failure so far
        b.record_failure("d")
        b.record_failure("e")
        self.assertTrue(b.open)

    def test_tripped_skips(self) -> None:
        b = PhaseCircuitBreaker("llm_intent", failure_threshold=1)
        b.record_failure("boom")
        self.assertTrue(b.open)
        # once tripped, callers should take the degraded path (open=True skips the phase)


class RetrieverDenseBreakerTest(unittest.TestCase):
    def test_dense_breaker_disables_dense(self) -> None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.retriever import HybridRetriever
        from config.env_config import EnvConfig

        env = EnvConfig.from_env(overrides={"skip_data_verify": True})
        # lightweight construction (no full FTS): __new__ + minimal manual state
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.env = env
        retriever._dense = None
        retriever._text_lower = {}
        from utils.circuit_breaker import PhaseCircuitBreaker
        retriever._dense_breaker = PhaseCircuitBreaker("dense", failure_threshold=2)
        # _route_dense needs _ensure_dense; simulate consecutive failures: encoder.encode raises
        class BadEncoder:
            def encode(self, text):  # pragma: no cover
                raise RuntimeError("model down")

        class BadStore:
            matrix = None
            asins = []
            available = True

        bad_pair = (BadEncoder(), BadStore())
        retriever._dense = bad_pair  # set _dense directly so _route_dense uses it
        retriever._ensure_dense = lambda: bad_pair  # type: ignore[method-assign]
        from agent.intent_router import IntentRoute
        route = IntentRoute(category_tokens=[], query_terms=["cotton"])
        pool: dict = {}
        retriever._accumulate = lambda pool, asin, score, source: None  # type: ignore[method-assign]
        # 1st failure -> not tripped; dense still attempts
        retriever._route_dense(route, pool, top_k=10, mode="recover")
        self.assertIsNotNone(retriever._dense)  # still holds (encoder, store)
        # 2nd failure -> trips and disables dense
        retriever._route_dense(route, pool, top_k=10, mode="recover")
        self.assertTrue(retriever._dense_breaker.open)
        self.assertEqual(retriever._dense, (None, None))  # hybrid -> bm25 in-process degradation


if __name__ == "__main__":
    unittest.main()
