"""Environment & model capability probe (the "perception layer" of the agent's autonomous
    decisions).

Probed capabilities:
- Device: cuda / cpu (when torch is importable)
- LLM: run the health-check initialize() against the configured provider (deepseek/openai/none),
   yielding available / disabled / unavailable (real network + auth probe; timeout/circuit-break
   handled by the client);
   no key -> disabled (no network request).
- Dense retrieval (BLaIR): transformers or sentence-transformers importable AND the offline
product-vector
  npy (data/offline_blair_embeds.npy, produced by scripts/encode_catalog_blair.py) exists.
  Actual model loading stays lazy in the retriever with failure fallback (environment-aware: any
  missing piece -> fall back to BM25).
- Cross-encoder rerank: FlagEmbedding importable AND bge-reranker-v2-m3 already cached locally or
downloadable
  (a real load failure still degrades to fused-order ranking in the reranker).
- Optional network probe: when the LLM is unconfigured and CAPABILITY_NETWORK_PROBE=1, probe
external connectivity with httpx.

Design (team highlight): the agent probes once at startup; runtime_controller then decides
autonomously
"whether the LLM/models are usable and whether to use them"; all configurable switches, off by
default, rule fallback on failure.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field

from config.env_config import EnvConfig
from llm.base import LLMClient
from llm.rerank import RerankClient, RerankState
from utils.rex_reranker import is_generation_reranker

logger = logging.getLogger(__name__)

_LLM_READY_STATES = {"available"}


def _spec_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class CapabilityProfile:
    """Complete result of one probe."""

    device: str = "cpu"  # cuda / cpu
    llm_provider: str = "none"  # configured provider
    llm_model: str = ""  # configured model name
    llm_state: str = "disabled"  # available / disabled / unavailable
    llm_error: str = ""  # failure reason (sanitized)
    sdk_available: bool = False  # openai SDK importable
    transformers_available: bool = False  # transformers importable (BLaIR query encoding)
    dense_encoder_available: bool = False  # transformers or sentence-transformers importable
    blair_npy_ready: bool = False  # offline product-vector npy exists (pre-computed BLaIR encoding)
    dense_available: bool = False  # dense channel truly usable (encoder + npy both present)
    reranker_available: bool = False  # FlagEmbedding importable + model obtainable
    reranker_model_cached: bool = False  # whether bge-reranker-v2-m3 is cached locally
    rerank_backend: str = "text"  # rerank backend text/chat/auto
    text_rerank_available: bool = False  # qwen3-rerank MaaS truly probed available
    text_rerank_error: str = ""  # probe failure reason (sanitized)
    network_available: bool = False  # external connectivity (probed)
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
            f"text_rerank={'yes' if self.text_rerank_available else 'no'}",
            f"network={'yes' if self.network_available else 'no'}",
        ]
        if self.llm_error:
            parts.append(f"llm_error={self.llm_error}")
        return " ".join(parts)


class CapabilityProbe:
    """One-time environment probe at startup (result cached to avoid repeated health checks)."""

    def __init__(self, env: EnvConfig, llm_client: LLMClient) -> None:
        self.env = env
        self.llm_client = llm_client
        self._profile: CapabilityProfile | None = None

    def probe(self) -> CapabilityProfile:
        if self._profile is not None:
            return self._profile
        profile = CapabilityProfile()
        profile.device = self._detect_device()

        # SDK / model-lib availability (import-level probe; actual model loading stays lazy
        # downstream with fallback)
        profile.sdk_available = _spec_available("openai")
        profile.transformers_available = _spec_available("transformers")
        profile.dense_encoder_available = profile.transformers_available or _spec_available(
            "sentence_transformers"
        )
        profile.blair_npy_ready = self._blair_npy_exists()
        # Dense channel: usable only when the encoder + offline product vectors are both ready
        # (BLaIR dense retrieval)
        profile.dense_available = profile.dense_encoder_available and profile.blair_npy_ready
        # Reranker-model availability: RexReranker/Qwen3-Reranker (generative) -> transformers
        # importable;
        # cross-encoders like bge-reranker-v2-m3 -> FlagEmbedding importable.
        if is_generation_reranker(self.env.reranker_model):
            profile.reranker_available = _spec_available("transformers")
        else:
            profile.reranker_available = _spec_available("FlagEmbedding")
        profile.reranker_model_cached = self._reranker_model_cached()

        # qwen3-rerank text rerank (MaaS): real probe when backend is text/auto
        # (missing key/base_url -> disabled, no network; configured but failing calls ->
        # unavailable)
        profile.rerank_backend = self.env.llm.rerank_backend
        if profile.rerank_backend in ("text", "auto"):
            try:
                rc = RerankClient(
                    model=self.env.llm.qwen_rerank_model,
                    workspace_id=self.env.llm.dashscope_workspace_id,
                    base_url=self.env.llm.qwen_rerank_base_url,
                )
                st = rc.initialize()
                profile.text_rerank_available = st.state == RerankState.AVAILABLE
                profile.text_rerank_error = st.error_message
            except Exception as exc:  # a probe exception never blocks startup
                profile.text_rerank_available = False
                profile.text_rerank_error = str(exc)[:120]

        # LLM health check (real network + auth probe; no key -> immediately disabled, no request)
        status = self.llm_client.initialize()
        profile.llm_provider = status.provider or self.env.llm_backend
        profile.llm_model = status.model or self.env.llm_model
        profile.llm_state = status.state.value
        profile.llm_error = status.error_message or ""
        profile.network_available = profile.llm_available

        # Optional external-network probe (still learn network status when the LLM is unavailable)
        if not profile.llm_available and _network_probe_enabled():
            profile.network_available = self._network_probe()

        profile.notes.append(
            "LLM availability is decided by the client health check; dense channel requires both the BLaIR query encoder and the offline npy;"  # noqa: E501
            "rerank requires FlagEmbedding importable (load failures still fall back to fused ordering)"  # noqa: E501
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
        """Whether the offline product-vector npy exists (output of
            scripts/encode_catalog_blair.py)."""
        from pathlib import Path

        path = Path(self.env.blair_offline_embedding_path)
        emb = path if path.suffix == ".npy" else path.with_suffix(".npy")
        return emb.exists()

    def _reranker_model_cached(self) -> bool:
        """Whether the configured reranker model is already in the local HF cache (loadable
            offline)."""
        try:
            from huggingface_hub import scan_cache_dir

            model_name = self.env.reranker_model
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

    return os.environ.get("CAPABILITY_NETWORK_PROBE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
