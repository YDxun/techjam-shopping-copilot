"""配置-环境-性能查找表（LUT）：加载 data/assets/env_config_lut.json，按环境推荐最优配置。

- load_lut(): 惰性加载静态资产（缺失返回 None，RuntimeController 回退默认）；
- env_fingerprint(): 由探测结果生成环境指纹字符串；
- recommend(): 返回该环境下 technical_score 最高（且延迟/内存达标）的 config_id。
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
    """读取 LUT（惰性缓存）；文件缺失/损坏 → None（回退默认策略）。"""
    global _lut_cache
    if _lut_cache is not None:
        return _lut_cache
    p = Path(path) if path else _LUT_PATH
    if not p.exists():
        logger.warning("[lut] %s 不存在 → 使用默认策略（保底）", p)
        _lut_cache = {}
        return _lut_cache
    try:
        _lut_cache = __import__("json").loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[lut] 加载失败（%s）→ 使用默认策略", exc)
        _lut_cache = {}
    return _lut_cache


def env_fingerprint(*, device: str, dense: bool, llm: bool, network: bool) -> str:
    """环境指纹：'device=cuda;dense=yes;llm=no;network=no'。"""
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
    """按环境指纹返回最优配置档案（score 最高且延迟/内存达标）；无匹配 → None。"""
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
