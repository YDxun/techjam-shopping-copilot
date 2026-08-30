"""qwen3-rerank text-rerank client (Alibaba Cloud MaaS /reranks compatible endpoint).

Replaces the legacy "LLM semantic rerank" (chat JSON scoring) branch with a true text-reranking
model:
- Endpoint: POST {base_url}/reranks with body {model, query, documents, top_n}
- The key is read only from the environment (DASHSCOPE_API_KEY); no key is ever written in code
- base_url resolution: QWEN_RERANK_BASE_URL (full) > DASHSCOPE_WORKSPACE_ID subdomain concatenation
- Three-state availability: available / disabled (missing key or base_url) / unavailable (probe
failure)
- Failures/timeouts always return None; the caller falls back to rule ordering (environment-aware)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_REGION = "cn-beijing"
# International DashScope generic endpoint (no workspace ID needed; user-verified)
_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-api/v1"
_DEFAULT_MODEL = "qwen3-rerank"


class RerankState(str, Enum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass
class RerankStatus:
    state: RerankState
    model: str = ""
    base_url: str = ""
    error_message: str = ""
    latency_ms: float = 0.0
    attempts: int = 0


@dataclass
class RerankResult:
    """Relevance score of a single document."""

    index: int
    score: float
    document: str = ""


class RerankClient:
    """Alibaba Cloud MaaS qwen3-rerank text-rerank client (degrades gracefully on failure)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = _DEFAULT_MODEL,
        workspace_id: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None
            else os.environ.get("DASHSCOPE_API_KEY", "").strip()
        )
        self._model = model or _DEFAULT_MODEL
        self._workspace_id = workspace_id
        self._base_url = self._resolve_base_url(base_url)
        self._timeout_seconds = timeout_seconds
        self._status: RerankStatus | None = None
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------------
    def _resolve_base_url(self, explicit: str | None) -> str:
        if explicit and explicit.strip():
            return explicit.strip().rstrip("/")
        ws = (self._workspace_id or "").strip()
        if ws:
            return f"https://{ws}.{_DEFAULT_REGION}.maas.aliyuncs.com/compatible-api/v1"
        # fallback: env full base_url > default international endpoint
        env_url = (
            os.environ.get("QWEN_RERANK_BASE_URL", "").strip()
            or os.environ.get("DASHSCOPE_BASE_URL", "").strip()
        )
        if env_url:
            return env_url.rstrip("/")
        return _DEFAULT_BASE_URL

    @property
    def available(self) -> bool:
        return self._status is not None and self._status.state == RerankState.AVAILABLE

    @property
    def status(self) -> RerankStatus:
        if self._status is None:
            self._status = RerankStatus(state=RerankState.UNAVAILABLE, model=self._model)
        return self._status

    # ------------------------------------------------------------------
    def initialize(self) -> RerankStatus:
        """Probe: whether key/base_url are set + one minimal rerank validation (real network +
            auth)."""
        if not self._api_key:
            self._status = RerankStatus(state=RerankState.DISABLED, model=self._model,
                                        error_message="DASHSCOPE_API_KEY is not configured")
            return self._status
        if not self._base_url:
            self._status = RerankStatus(
                state=RerankState.DISABLED, model=self._model,
                error_message=(
                    "QWEN_RERANK_BASE_URL or DASHSCOPE_WORKSPACE_ID is missing (cannot build base_url)"  # noqa: E501
                ),
            )
            return self._status
        t0 = time.time()
        try:
            res = self._call_rerank(
                query="test",
                documents=["alpha", "beta"],
                top_n=1,
            )
            latency = (time.time() - t0) * 1000
            if res is None:
                raise RuntimeError("rerank returned empty")
            self._status = RerankStatus(
                state=RerankState.AVAILABLE, model=self._model,
                base_url=self._base_url, latency_ms=round(latency, 1), attempts=1,
            )
            logger.info("[rerank] qwen3-rerank available: %s (%.0fms)", self._base_url, latency)
        except Exception as exc:
            self._status = RerankStatus(
                state=RerankState.UNAVAILABLE, model=self._model,
                base_url=self._base_url, error_message=str(exc)[:200], attempts=1,
            )
            logger.warning("[rerank] qwen3-rerank unavailable (%s) -> rerank falls back to rule ordering", exc)  # noqa: E501
        return self._status

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult] | None:
        """Score documents by relevance to the query; returns RerankResult list sorted by
            descending score."""
        if not self.available:
            return None
        try:
            res = self._call_rerank(query, documents, top_n)
            if res is None:
                return None
            return res
        except Exception as exc:
            logger.warning("[rerank] qwen3-rerank call failed (%s) -> fallback to rule ordering", exc)  # noqa: E501
            return None

    # ------------------------------------------------------------------
    def _call_rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> list[RerankResult] | None:
        """POST {base_url}/reranks (requests; user-verified international endpoint), then parse
            results."""
        import requests

        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            body["top_n"] = top_n
        response = requests.post(
            f"{self._base_url}/reranks",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            results = data.get("results") or data.get("data") or []
        else:
            results = []
        out: list[RerankResult] = []
        for item in results:
            if isinstance(item, dict):
                idx = int(item.get("index", -1))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
            else:
                idx = int(getattr(item, "index", -1))
                score = float(getattr(item, "relevance_score", getattr(item, "score", 0.0)))
            if 0 <= idx < len(documents):
                out.append(RerankResult(index=idx, score=score, document=documents[idx]))
        out.sort(key=lambda r: r.score, reverse=True)
        return out or None
