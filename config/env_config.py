"""环境变量配置（Pillar III/IV：支持 dev/submit、LLM/检索后端切换、TOP_K 注入）。

全部外部模型调用都通过环境变量隔离，密钥只留占位、禁止硬编码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config import constants


@dataclass(frozen=True)
class EnvConfig:
    """运行时环境配置：由环境变量一次性解析，供各模块只读使用。"""

    env_mode: str = "dev"                     # dev=本地开发测试 / submit=提交模拟模式
    llm_backend: str = "none"                 # local / openai / none（none=纯离线规则，无需付费 API）
    retrieval_backend: str = "bm25"           # bm25 / dense / hybrid
    top_k: int = constants.DEFAULT_TOP_K      # 匹配 HitRate@K 设置
    llm_model: str = ""                       # 本地/API LLM 模型名（占位，不写死密钥）
    openai_api_key: str = ""                  # 环境变量占位：OPENAI_API_KEY
    openai_base_url: str = ""                 # 可选：自定义 OpenAI 兼容端点
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    clarify_strategy: str = "other"           # other=最大信息量澄清 / attribute=按属性优先级澄清
    llm_rerank: bool = True                   # LLM_RERANK=0 closes LLM rerank, force rule order
    override_erase: bool = False              # OVERRIDE_ERASE=1 erases old-preference slot on override (default: keep as weak soft)
    skip_data_verify: bool = False            # SKIP_DATA_VERIFY=1 可跳过 sha256 校验
    sample_limit: int | None = None           # SAMPLE_LIMIT：开发时只跑前 N 条会话（冒烟测试）
    output_path: str = "results.json"         # 本地评估输出
    rerank_candidates: int = 300              # 送入重排的候选池大小
    max_constraint_asks: int = 3              # 澄清最大轮数（防无效冗余对话，优化 MTTC）
    env_overrides: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EnvConfig":
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name, "")
            try:
                return int(raw) if raw else default
            except ValueError:
                return default

        def _bool(name: str, default: bool = False) -> bool:
            raw = os.environ.get(name, "")
            return raw.strip().lower() in {"1", "true", "yes", "on"} if raw else default

        env_mode = os.environ.get("ENV_MODE", "dev").strip().lower() or "dev"
        llm_backend = os.environ.get("LLM_BACKEND", "none").strip().lower() or "none"
        retrieval_backend = os.environ.get("RETRIEVAL_BACKEND", "bm25").strip().lower() or "bm25"

        # 提交模式下的安全默认：不允许悄悄依赖外部付费 API
        if env_mode == "submit" and llm_backend == "openai":
            llm_backend = "none" if not os.environ.get("OPENAI_API_KEY") else llm_backend

        return cls(
            env_mode=env_mode,
            llm_backend=llm_backend,
            retrieval_backend=retrieval_backend,
            top_k=_int("TOP_K", constants.DEFAULT_TOP_K),
            llm_model=os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B").strip(),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "").strip(),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
            reranker_model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
            clarify_strategy=os.environ.get("CLARIFY_STRATEGY", "other").strip().lower(),
            llm_rerank=_bool("LLM_RERANK", True),
            override_erase=_bool("OVERRIDE_ERASE"),
            skip_data_verify=_bool("SKIP_DATA_VERIFY"),
            sample_limit=_int("SAMPLE_LIMIT", 0) or None,
            output_path=os.environ.get("OUTPUT_PATH", "results.json").strip(),
            rerank_candidates=_int("RERANK_CANDIDATES", 300),
            max_constraint_asks=_int("MAX_CONSTRAINT_ASKS", 3),
            env_overrides=dict(os.environ),
        )

    @property
    def offline(self) -> bool:
        """是否完全离线可运行（无外部 API 依赖）。"""
        return self.llm_backend in {"none", "local"}

    def summary(self) -> str:
        return (
            f"ENV_MODE={self.env_mode} LLM_BACKEND={self.llm_backend} "
            f"RETRIEVAL_BACKEND={self.retrieval_backend} TOP_K={self.top_k} "
            f"offline={self.offline}"
        )

