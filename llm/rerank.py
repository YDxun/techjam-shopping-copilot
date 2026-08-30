"""qwen3-rerank 文本重排客户端（阿里云 MaaS /reranks 兼容端点）。

替换原"LLM 语义重排"（chat JSON 打分）分支为真正的文本重排序模型：
- 端点：POST {base_url}/reranks，body {model, query, documents, top_n}
- 密钥只从环境变量读取（DASHSCOPE_API_KEY），代码不落任何 key
- base_url 解析：QWEN_RERANK_BASE_URL（完整）> DASHSCOPE_WORKSPACE_ID 拼子域
- 三态可用性：available / disabled（缺 key 或缺 base_url）/ unavailable（探测失败）
- 失败/超时一律返回 None，由上层回退规则排序（环境自感知）
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

_DEFAULT_REGION = "cn-beijing"
# 国际版 DashScope 通用端点（无需 workspace ID，用户实测可用）
_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-api/v1"
_DEFAULT_MODEL = "qwen3-rerank"


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    """Reject API redirects before a bearer-authenticated follow-up request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib_error.HTTPError(
            req.full_url,
            code,
            "qwen3-rerank redirects are not allowed",
            headers,
            fp,
        )


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
    """单条文档的相关性得分。"""

    index: int
    score: float
    document: str = ""


class RerankClient:
    """阿里云 MaaS qwen3-rerank 文本重排客户端（可失败降级）。"""

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
            api_key if api_key is not None else os.environ.get("DASHSCOPE_API_KEY", "").strip()
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
        # 兜底：环境变量完整 base_url > 默认国际版端点
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
        """探测：key/base_url 是否配置 + 一次最小 rerank 验证（真实网络+鉴权）。"""
        if not self._api_key:
            self._status = RerankStatus(
                state=RerankState.DISABLED,
                model=self._model,
                error_message="DASHSCOPE_API_KEY 未配置",
            )
            return self._status
        if not self._base_url:
            self._status = RerankStatus(
                state=RerankState.DISABLED,
                model=self._model,
                error_message=(
                    "缺少 QWEN_RERANK_BASE_URL 或 DASHSCOPE_WORKSPACE_ID（无法拼出 base_url）"
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
                raise RuntimeError("rerank 返回空")
            self._status = RerankStatus(
                state=RerankState.AVAILABLE,
                model=self._model,
                base_url=self._base_url,
                latency_ms=round(latency, 1),
                attempts=1,
            )
            logger.info("[rerank] qwen3-rerank available: %s (%.0fms)", self._base_url, latency)
        except Exception as exc:
            self._status = RerankStatus(
                state=RerankState.UNAVAILABLE,
                model=self._model,
                base_url=self._base_url,
                error_message=str(exc)[:200],
                attempts=1,
            )
            logger.warning("[rerank] qwen3-rerank 不可用（%s）→ 重排回退规则排序", exc)
        return self._status

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult] | None:
        """对 documents 按与 query 的相关性打分，返回按分数降序的 RerankResult 列表。"""
        if not self.available:
            return None
        try:
            res = self._call_rerank(query, documents, top_n)
            if res is None:
                return None
            return res
        except Exception as exc:
            logger.warning("[rerank] qwen3-rerank 调用失败（%s）→ 回退规则排序", exc)
            return None

    # ------------------------------------------------------------------
    def _call_rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> list[RerankResult] | None:
        """POST {base_url}/reranks（标准库 HTTP，解析 results）。"""
        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            body["top_n"] = top_n
        request = urllib_request.Request(
            f"{self._base_url}/reranks",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib_request.build_opener(_RejectRedirectHandler())
        with opener.open(request, timeout=self._timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
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
