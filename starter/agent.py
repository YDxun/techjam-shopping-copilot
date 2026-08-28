"""官方入口：直接复用新 Agent（避免官方评估器跑到旧 BM25 基线）。

官方评估器 `evaluator/local_evaluator.py` 固定从本模块导入 `Agent`：
    from starter.agent import Agent

此处仅做"再导出"（re-export），不改动官方评估器任何一行。
新的完整实现位于 `agent/main_agent.py`；旧 BM25 基线保留在 git 历史中。

注意：依赖模块（agent/dialogue/、agent/capability_probe.py、agent/runtime_controller.py 等）
必须随提交包一起提供，否则干净检出下本入口 import 会失败。
"""
from __future__ import annotations

from agent.main_agent import Agent

__all__ = ["Agent"]
