"""Competition deliverable: demo session (hard task requirement).

Runs one public session through the official evaluator, printing per-turn
[turn / user_message / ask_attribute / message / top-10 parent_asin]，
and saving the verbatim log to docs/demo_session.log.

Does not modify the evaluator: reuses evaluator.local_evaluator's initial_message / customer_reply /
materialize_hidden_fields / coarse_category / normalize_recommendations (only calls them).
Usage: python scripts/demo_session.py [--index 0] [--out docs/demo_session.log]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.main_agent import Agent  # noqa: E402
from config.env_config import EnvConfig  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402 official evaluator (only called, never modified)
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from llm.base import DisabledLLMClient  # noqa: E402

MAX_TURNS = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="public session index (default 0)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo_session.log"))
    args = ap.parse_args()

    env = EnvConfig.from_env()
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    sample = samples[args.index]
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")

    agent = Agent(
        catalog_path=ROOT / "data" / "catalog.jsonl", env=env, llm_client=DisabledLLMClient()
    )
    session_id = "demo_public_" + str(sample["sample_id"])
    agent.reset(session_id, sample["user_profile"])

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )

    lines: list[str] = []
    lines.append(f"# Demo session (official evaluator, public session {sample['sample_id']})")
    lines.append(f"# Scenario: {sample['scenario_type']} | Target: {target}")
    lines.append(f"# User profile: {str(sample['user_profile'])[:200]}")
    hit = None

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, 10)
        ask = response.get("ask_attribute")
        message = str(response.get("message", ""))
        recs = normalize_recommendations(response.get("recommendations", []), catalog_ids)[:10]
        block = (
            f"\n[turn {turn}]\n"
            f"  user_message : {user_message}\n"
            f"  ask_attribute: {ask}\n"
            f"  message      : {message}\n"
            f"  top-10 asin  : {recs}"
        )
        print(block)
        lines.append(block)
        if override_applied and target in recs:
            hit = recs.index(target) + 1
            break
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(effective, ask, disclosed, boundary_used)

    lines.append(f"\n# Result: {'HIT at rank ' + str(hit) if hit else 'MISS'} (target={target})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nLog written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
