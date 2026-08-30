"""Config-environment-performance lookup table (LUT): loads data/assets/env_config_lut.json and
    recommends the best config per environment.

- load_lut(): lazily loads the static asset (missing -> None, RuntimeController falls back to
defaults);
- env_fingerprint(): builds the environment-fingerprint string from probe results;
- recommend(): returns the config_id with the highest technical_score (meeting latency/memory
budgets) for that environment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
_LUT_PATH = ROOT / "data" / "assets" / "env_config_lut.json"
_lut_cache: dict[str, Any] | None = None


def load_lut(path: str | Path | None = None) -> dict[str, Any] | None:
    """Read the LUT (lazy cache); missing/corrupt -> None (fall back to the default strategy)."""
    global _lut_cache
    if _lut_cache is not None:
        return _lut_cache
    p = Path(path) if path else _LUT_PATH
    if not p.exists():
        logger.warning("[lut] %s does not exist -> use the default strategy (baseline)", p)
        _lut_cache = {}
        return _lut_cache
    try:
        _lut_cache = __import__("json").loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[lut] load failed (%s) -> use the default strategy", exc)
        _lut_cache = {}
    return _lut_cache


def env_fingerprint(*, device: str, dense: bool, llm: bool, network: bool) -> str:
    """Environment fingerprint: 'device=cuda;dense=yes;llm=no;network=no'."""
    return (
        f"device={device};dense={'yes' if dense else 'no'};"
        f"llm={'yes' if llm else 'no'};network={'yes' if network else 'no'}"
    )


def recommend(
    fingerprint: str,
    lut: dict[str, Any] | None = None,
    max_latency_ms: float | None = None,
    max_memory_mb: float | None = None,
) -> dict[str, Any] | None:
    """Return the best config profile for an environment fingerprint (highest score meeting
        latency/memory); None when unmatched."""
    data = lut if lut is not None else load_lut()
    env_entry = (data or {}).get("environments", {}).get(fingerprint)
    if not env_entry:
        return None
    best: dict[str, Any] | None = None
    for cfg in env_entry.get("configs", []):
        if max_latency_ms is not None and cfg.get("latency_ms_per_turn", 0) > max_latency_ms:
            continue
        if max_memory_mb is not None and cfg.get("memory_mb", 0) > max_memory_mb:
            continue
        if best is None or cfg.get("technical_score", 0) > best.get("technical_score", 0):
            best = cfg
    return best
