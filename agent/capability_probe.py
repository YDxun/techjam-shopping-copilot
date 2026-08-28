"""环境与模型能力探测（Agent 自主决策的"感知层"）。

探测项：
- 设备：cuda / cpu（torch 可用时）
- LLM：对配置的 provider（deepseek/openai/none）执行健康检查 initialize()，
  得到 available / disabled / unavailable 三态（真实探测网络+鉴权，超时/熔断由客户端保证）；
  无 key → disabled（不发网络请求）。
- 稠密检索（BLaIR）：transformers/sentence-transformers 任一可导入，且离线商品向量
  npy（data/offline_blair_embeds.npy，scripts/encode_catalog_blair.py 产物）存在。
  模型实际加载仍由 retriever 懒加载并失败回退（环境自感知：缺任一环节 → 回退 BM25）。
- 交叉编码重排：FlagEmbedding 可导入，且 bge-reranker-v2-m3 已本地缓存或可下载
  （模型实际加载失败仍由 reranker 降级 fused 排序）。
- 可选网络探测：LLM 未配置时，若 CAPABILITY_NETWORK_PROBE=1 则用 httpx 探测外网连通性。

设计定位（团队特色）：Agent 启动时做一次探测，runtime_controller 依据探测结果
自主决定"LLM/模型能不能用、用不用"，全部可配置开关、默认关、失败回退规则。
"""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field

from config.env_config import EnvConfig
from llm.base import LLMClient

logger = logging.getLogger(__name__)

_LLM_READY_STATES = {"available"}


def _spec_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class CapabilityProfile:
    """一次探测的完整结果。"""

    device: str = "cpu"                        # cuda / cpu
    llm_provider: str = "none"                 # 配置的 provider
    llm_model: str = ""                        # 配置的模型名
    llm_state: str = "disabled"                # available / disabled / unavailable
    llm_error: str = ""                        # 失败原因（脱敏）
    sdk_available: bool = False                # openai SDK 可导入
    transformers_available: bool = False       # transformers 可导入（BLaIR 查询编码）
    dense_encoder_available: bool = False      # transformers / sentence-transformers 任一可导入
    blair_npy_ready: bool = False              # 离线商品向量 npy 存在（预先 BLaIR 编码产物）
    dense_available: bool = False              # 稠密通道真正可用（编码器 + npy 都在）
    reranker_available: bool = False           # FlagEmbedding 可导入 + 模型可获取
    reranker_model_cached: bool = False        # bge-reranker-v2-m3 是否已本地缓存
    network_available: bool = False            # 外网连通性（探测到）
    notes: list[str] = field(default_factory=list)

    @property
    def llm_available(self) -> bool:
        return self.llm_state in _LLM_READY_STATES

    def summary(self) -> str:
        parts = [
            f"device={self.device}",
            f"llm={self.llm_provider}:{self.llm_state}",
            f"dense={'yes' if self.dense_available else 'no'}",
            f"reranker={'yes' if self.reranker_available else 'no'}",
            f"network={'yes' if self.network_available else 'no'}",
        ]
        if self.llm_error:
            parts.append(f"llm_error={self.llm_error}")
        return " ".join(parts)


class CapabilityProbe:
    """启动期一次性环境探测（结果缓存，避免重复健康检查）。"""

    def __init__(self, env: EnvConfig, llm_client: LLMClient) -> None:
        self.env = env
        self.llm_client = llm_client
        self._profile: CapabilityProfile | None = None

    def probe(self) -> CapabilityProfile:
        if self._profile is not None:
            return self._profile
        profile = CapabilityProfile()
        profile.device = self._detect_device()

        # SDK / 模型库可用性（导入级探测，模型实际加载仍由下游懒加载+回退）
        profile.sdk_available = _spec_available("openai")
        profile.transformers_available = _spec_available("transformers")
        profile.dense_encoder_available = (profile.transformers_available
                                           or _spec_available("sentence_transformers"))
        profile.blair_npy_ready = self._blair_npy_exists()
        # 稠密通道：编码器 + 离线商品向量 都就绪才算可用（BLaIR 稠密检索）
        profile.dense_available = profile.dense_encoder_available and profile.blair_npy_ready
        profile.reranker_available = _spec_available("FlagEmbedding")
        profile.reranker_model_cached = self._reranker_model_cached()

        # LLM 健康检查（真实网络+鉴权探测；无 key 立即 disabled，不发请求）
        status = self.llm_client.initialize()
        profile.llm_provider = status.provider or self.env.llm_backend
        profile.llm_model = status.model or self.env.llm_model
        profile.llm_state = status.state.value
        profile.llm_error = status.error_message or ""
        profile.network_available = profile.llm_available

        # 可选外网探测（LLM 不可用时仍想了解网络状态）
        if not profile.llm_available and _network_probe_enabled():
            profile.network_available = self._network_probe()

        profile.notes.append(
            "LLM 可用性由客户端健康检查决定；稠密通道=BLaIR查询编码器+离线npy双就绪；"
            "重排=FlagEmbedding可导入（加载失败仍会回退 fused 排序）"
        )
        self._profile = profile
        logger.info("[capability] %s", profile.summary())
        return profile

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _blair_npy_exists(self) -> bool:
        """离线商品向量 npy 是否已生成（scripts/encode_catalog_blair.py 产物）。"""
        from pathlib import Path
        path = Path(self.env.blair_offline_embedding_path)
        emb = path if path.suffix == ".npy" else path.with_suffix(".npy")
        return emb.exists()

    @staticmethod
    def _reranker_model_cached() -> bool:
        """bge-reranker-v2-m3 是否已下载到本地 HF 缓存（可离线加载）。"""
        import os
        try:
            from huggingface_hub import scan_cache_dir
            model_name = os.environ.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
            repos = scan_cache_dir()
            for r in repos.repos:
                if r.repo_id == model_name and r.repo_type == "model":
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _network_probe() -> bool:
        try:
            import httpx
            r = httpx.get("https://api.deepseek.com", timeout=2.0, follow_redirects=True)
            return r.status_code < 500
        except Exception:
            return False


def _network_probe_enabled() -> bool:
    import os
    return os.environ.get("CAPABILITY_NETWORK_PROBE", "0").strip().lower() in {"1", "true", "yes", "on"}
