"""Retrieval-pipeline demo: build a mock session_state, run the full steps 4-6 chain, and print
    the Top-10 asins.

Run: python retrieval_pipeline/test_pipeline.py
(Uses the frozen catalog data/catalog.jsonl by default; the dense channel auto-disables when the
offline npy is missing;
 reranking auto-degrades to fused ordering without FlagEmbedding -- the demo chain depends on no
 paid API.)
"""
from __future__ import annotationsimport loggingimport sysfrom pathlib import Pathsys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_pipeline.models import PipelineOutput, SessionState, StrategyConfigfrom retrieval_pipeline.pipeline import RetrievalPipelinelogging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def show(session: SessionState, pipeline: RetrievalPipeline, label: str) -> None:
    print("\n" + "=" * 90)
    print(f"[{label}]")
    print(f"  constraints    : {session.constraints}")
    print(f"  recovery_mode  : {session.recovery_mode}")
    print(f"  raw_query      : {session.user_raw_query!r}")
    out: PipelineOutput = pipeline.run(session)
    print(f"  raw_fused n    : {len(out.raw_fused_candidates)}")
    for asin, score in out.raw_fused_candidates[:8]:
        title = (pipeline.catalog.get(asin) or {}).get("title", "")[:60]
        print(f"    fused {score:+.4f}  {asin}  {title}")
    print(f"  reranked_top10 : {out.reranked_top10}")


def main() -> None:
    pipeline = RetrievalPipeline()   # reuse one instance (index built once)

    # scenario A: normal buying session (hard filter)
    show(SessionState(
        constraints={"material": "cotton", "color": "black"},
        recovery_mode=False,
        strategy_config=StrategyConfig(retrieval_pool_size=50),
        user_raw_query="I'm looking for T-Shirts. A key requirement is: cotton.",
    ), pipeline, "normal mode (step 5 channel 1 hard filter)")

    # scenario B: RECOVER mode (miss streak >= 2) -> penalty scoring / synonyms / variants / larger
    # pool
    show(SessionState(
        constraints={"material": "cotton", "color": "black", "budget_max": 40},
        recovery_mode=True,
        strategy_config=StrategyConfig(
            rrf_alpha=0.8,
            retrieval_pool_size=100,
            enable_query_variant=True,
            enable_synonym=True,
        ),
        user_raw_query="I'm looking for a cotton jumper under $50.",
    ), pipeline, "RECOVER mode (penalty scoring + synonyms + variants + pool 100)")

    # scenario C: override cleared the constraints -> only user_raw_query
    show(SessionState(
        constraints={},
        recovery_mode=False,
        strategy_config=StrategyConfig(),
        user_raw_query="Actually, ignore my earlier preference. What I need is: leather.",
    ), pipeline, "override cleared constraints (raw query only)")


if __name__ == "__main__":
    main()
