"""Paraphrase robustness harness (tools/paraphrase_eval.py).

重放官方会话循环，但对顾客消息施加两层改写（私有集可能出现的自然语言改写）：
  L0 无改写（基线）
  L1 模板改写：只改官方模板措辞，约束取值原样保留
  L2 值同义改写：约束取值用 vocab 同义词替换（如 grey->gray / jumper->sweater）
衡量 Agent 在改写下的 HR/MRR/MTTC/TS 退化程度。

用法（在仓库根）：
  python tools/paraphrase_eval.py --level L1 --llm 0
  python tools/paraphrase_eval.py --level L2 --llm 1 --key <DEEPSEEK_API_KEY>
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.main_agent import Agent  # noqa: E402
from config.env_config import EnvConfig  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    behavior_for,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
    load_jsonl,
)

_LOOK = re.compile(r"i'?m looking for (.+?)(,|\.)", re.I | re.S)
_KEY = re.compile(r"a key requirement is:\s*(.+?)\s*$", re.I | re.S)
_MATTERS = re.compile(r"what matters is:\s*(.+?)\s*$", re.I | re.S)
_NEED = re.compile(r"what i need is:\s*(.+?)\s*$", re.I | re.S)
_NO_MORE = re.compile(r"i don'?t have an additional preference for\s+(\w+)", re.I)
_NO_PREF = re.compile(r"i don'?t have a preference for\s+(\w+)", re.I)
_STILL = re.compile(r",?\s*but i'?m still exploring\.?\s*$", re.I)

_LOOK_ALT = ["I want ", "I need ", "I'm after ", "I'm hoping for "]
_KEY_ALT = ["A must-have is: ", "One key thing is: ", "Important to me: ", "I specifically need: "]
_MATTERS_ALT = [
    "For that, I care about: ",
    "For that, what's important is: ",
    "That said, I really value: ",
]
_NEED_ALT = ["Scrap that; what I actually need is: ", "Change of plan; the real requirement is: "]
_NO_MORE_ALT = ["That's all I've got on ", "Nothing further for "]
_NO_PREF_ALT = ["I've no preference for ", "Either works for "]


def _pick(alts, seed):
    return alts[seed % len(alts)]


def paraphrase_l1(msg, seed):
    m = _KEY.search(msg)
    if m:
        return _pick(_KEY_ALT, seed) + m.group(1).strip()
    m = _MATTERS.search(msg)
    if m:
        return _pick(_MATTERS_ALT, seed) + m.group(1).strip()
    m = _NEED.search(msg)
    if m:
        return _pick(_NEED_ALT, seed) + m.group(1).strip()
    m = _NO_MORE.search(msg)
    if m:
        return _pick(_NO_MORE_ALT, seed) + m.group(1).lower() + "."
    m = _NO_PREF.search(msg)
    if m:
        return _pick(_NO_PREF_ALT, seed) + m.group(1).lower() + "."
    if "not quite right yet" in msg:
        return "Those don't fit. Ask me about one specific thing."
    m = _LOOK.search(msg)
    if m:
        cat = m.group(1).strip()
        rest = msg[m.end() :]
        if _STILL.search(rest):
            return _pick(_LOOK_ALT, seed) + cat + ", but I'm still exploring."
        return _pick(_LOOK_ALT, seed) + cat + ". " + rest.strip()
    return msg


_VOCAB = None


def _load_vocab():
    global _VOCAB
    if _VOCAB is None:
        try:
            _VOCAB = json.loads(
                (ROOT / "data/assets/vocab_v2_clean.json").read_text(encoding="utf-8")
            )
        except Exception:
            _VOCAB = {}
    return _VOCAB


def paraphrase_l2(msg, seed):
    out = paraphrase_l1(msg, seed)
    vocab = _load_vocab()
    if not vocab:
        return out
    dicts = vocab.get("dictionaries", {})
    # 收集 canonical->首个非自身同义词的映射
    alias = {}
    for entries in dicts.values():
        for canonical, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            syns = entry.get("synonyms") or []
            if isinstance(syns, str):
                syns = [syns]
            for s in syns:
                s = str(s)
                if s.lower() != str(canonical).lower():
                    alias[str(canonical).lower()] = s
                    break

    def sub(m):
        value = m.group(1).strip()
        key = value.lower()
        if key in alias:
            return m.group(0).replace(value, alias[key], 1)
        return m.group(0)

    out = re.sub(r"(is:\s*)([^;]+?)(?:\s*;|\s*$)", sub, out, flags=re.I)
    return out


def run_paraphrase(agent, samples, products, categories, level):
    total = len(samples)
    hits = 0
    rr = 0.0
    mttc_sum = 0.0
    for i, sample in enumerate(samples):
        target = str(sample["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        behavior = behavior_for(
            sample["scenario_type"],
            card,
            random.Random(f"{sample['sample_id']}\0{sample['scenario_type']}"),
        )
        eff = {**sample, "intent_card": card, "behavior": behavior}
        disclosed = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        seed = i * 7 + 3
        user_message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)
        if level == "L1":
            user_message = paraphrase_l1(user_message, seed)
        elif level == "L2":
            user_message = paraphrase_l2(user_message, seed)
        session_id = f"para_{i}"
        agent.reset(session_id, sample["user_profile"])
        hit_turn = None
        best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            resp = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = [r["parent_asin"] for r in resp.get("recommendations", [])]
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                if override.get("new_value"):
                    disclosed.add(str(override["new_value"]))
                user_message = str(override.get("message", ""))
                if level != "L0":
                    user_message = (
                        paraphrase_l2(user_message, seed)
                        if level == "L2"
                        else paraphrase_l1(user_message, seed)
                    )
            else:
                user_message, boundary_used = customer_reply(
                    eff, resp.get("ask_attribute"), disclosed, boundary_used
                )
                if level != "L0":
                    user_message = (
                        paraphrase_l2(user_message, seed)
                        if level == "L2"
                        else paraphrase_l1(user_message, seed)
                    )
        if hit_turn is not None:
            hits += 1
            rr += 1.0 / best_rank
            mttc_sum += hit_turn
        else:
            mttc_sum += MAX_TURNS + 1
    mttc = mttc_sum / total
    return {
        "n": total,
        "HR": hits / total,
        "MRR": rr / total,
        "MTTC": mttc,
        "Efficiency": max(0.0, min(1.0, (11 - mttc) / 10)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["L0", "L1", "L2"], default="L1")
    parser.add_argument("--llm", type=int, default=0, help="1=启用 LLM 意图（需 --key）")
    parser.add_argument("--key", default="", help="DEEPSEEK_API_KEY")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = load_jsonl(ROOT / "data/public_set.jsonl")
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(ROOT / "data/catalog.jsonl")

    environ = {
        "SKIP_DATA_VERIFY": "1",
        "ENV_MODE": "dev",
        "COMBO_FINGERPRINT_ENABLE": "1",
        "EMIT_GATE": "1",
        "EMIT_K0": "1",
        "EMIT_K1": "1",
        "EMIT_K2": "1",
        "EMIT_LATE_TURN": "5",
    }
    if args.llm and args.key:
        environ.update(
            {"LLM_PROVIDER": "deepseek", "LLM_INTENT_ENABLE": "1", "DEEPSEEK_API_KEY": args.key}
        )

    env = EnvConfig.from_env(environ=environ)
    llm_client = None
    if args.llm and args.key:
        from llm import create_llm_client

        llm_client = create_llm_client(env.llm)
        status = llm_client.initialize()
        print(f"LLM state={status.state.value}", flush=True)
    agent = Agent(catalog_path=ROOT / "data/catalog.jsonl", env=env, llm_client=llm_client)
    r = run_paraphrase(agent, samples, products, categories, args.level)
    ts = 0.5 * r["HR"] + 0.3 * r["MRR"] + 0.2 * r["Efficiency"]
    print(
        f"[paraphrase {args.level} | llm={bool(args.llm)} | n={r['n']}] "
        f"HR={r['HR']:.4f} MRR={r['MRR']:.4f} MTTC={r['MTTC']:.4f} "
        f"Efficiency={r['Efficiency']:.4f} TS={ts:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
