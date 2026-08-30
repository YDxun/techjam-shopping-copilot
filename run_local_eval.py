"""Local dev / submit-simulation evaluation launcher (calls the official evaluator without
    modifying it).

Usage:
    python run_local_eval.py                        # ENV_MODE=dev by default
    $env:ENV_MODE="dev";   python run_local_eval.py
    $env:ENV_MODE="submit"; python run_local_eval.py  # submit simulation: enforces
    offline-constraint checks
    $env:SAMPLE_LIMIT="10"; python run_local_eval.py  # smoke test

Official-evaluator compatible: directly reuses evaluator.local_evaluator's
load_jsonl / catalog_index / evaluate functions without changing a single line of evaluator source.
Output: results.json (per-session + overall + scenario metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ensure imports work from the repository root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.main_agent import Agent  # noqa: E402
from config import constants  # noqa: E402
from config.env_config import EnvConfig  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
)  # official evaluator (only called, never modified)
from llm import create_llm_client  # noqa: E402
from llm.base import LLMClient  # noqa: E402


def _summarize_session_logs(session_logs: dict[str, dict] | None) -> list[dict]:
    """Compress per-session structured logs (drop per-turn detail, keep summary fields, control
        results.json size)."""
    if not isinstance(session_logs, dict):  # safe degradation for test stubs / missing logs
        return []
    summary = []
    for session_id in sorted(session_logs):
        log = session_logs[session_id]
        summary.append(
            {
                "session_id": session_id,
                "strategy": log.get("strategy"),
                "strategy_lut": log.get("strategy_lut"),
                "latency_ms": round(float(log.get("latency_ms", 0.0)), 1),
                "prompt_tokens": int(log.get("prompt_tokens", 0)),
                "completion_tokens": int(log.get("completion_tokens", 0)),
                "phase_timings": {
                    k: round(float(v), 1) for k, v in (log.get("phase_timings") or {}).items()
                },
                "degradation": list(log.get("degradation", [])),
                "reasons": list(log.get("reasons", []))[:8],
            }
        )
    return summary


def initialize_llm(env: EnvConfig) -> LLMClient:
    """Initialize the optional LLM and report its sanitized availability."""
    client = create_llm_client(env.llm)
    status = client.initialize()
    details = (
        f"provider={status.provider} model={status.model} "
        f"state={status.state.value} attempts={status.attempts}"
    )
    if status.error_category is not None:
        details += f" error={status.error_category.value}"
    print(f"    LLM: {details}")
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description="TechJam2026 Shopping Copilot Agent local evaluation")  # noqa: E501
    parser.add_argument("--catalog", default=str(constants.CATALOG_PATH))
    parser.add_argument("--dataset", default=str(constants.PUBLIC_SET_PATH))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    env = EnvConfig.from_env()
    output = args.output or env.output_path
    print("\n=== TechJam2026 Shopping Copilot Agent local evaluation ===")
    print(f"    {env.summary()}")
    if env.env_mode == "submit":
        print(
            "    [submit] submit-simulation mode: enforces offline-constraint checks\n"
            "    (set LLM_PROVIDER=none first; LLM_BACKEND=none/local are also accepted)."
        )
        assert env.offline, (
            "submit mode forbids depending on external paid APIs (set LLM_PROVIDER=none first;\n"
            "LLM_BACKEND=none or local are also accepted)"
        )

    llm_client = initialize_llm(env)

    # Dataset integrity verification (Pillar IV / hard constraint 3)
    if not env.skip_data_verify:
        from utils import data_verify

        data_verify.verify_dataset(skip=False)

    # Official evaluator data loading
    t0 = time.time()
    samples = load_jsonl(args.dataset)
    if env.sample_limit:
        samples = samples[: env.sample_limit]
        print(f"    [dev] SAMPLE_LIMIT={env.sample_limit} (smoke-test subset)")
    catalog_ids, categories, products = catalog_index(args.catalog)
    print(
        f"    data loaded: {len(samples)} sessions / {len(catalog_ids)} products ({time.time() - t0:.1f}s)"  # noqa: E501
    )

    # Instantiate the business Agent (Pillar I-IV)
    agent = Agent(catalog_path=args.catalog, env=env, llm_client=llm_client)

    # Call the official evaluator evaluate() (the only scoring entry point; unmodified)
    t0 = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    print(f"    evaluation finished: {time.time() - t0:.1f}s")

    # P2 observability: aggregate per-session structured logs
    # (strategy/latency/tokens/degradation/reasons)
    # into an extra results.json field (official metric fields stay untouched and are not scored)
    result["agent_session_logs"] = _summarize_session_logs(
        getattr(agent, "session_logs", None)
    )

    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "sessions"}
    print("\n--- Overall metrics (computed by the official evaluator) ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults written to: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
