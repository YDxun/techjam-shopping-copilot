from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID, uuid4

from webapp.metrics import UsageRecorder, estimate_cost_usd

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    def reset(self, session_id: str, user_profile: dict) -> None:
        raise NotImplementedError

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        raise NotImplementedError


class CatalogProtocol(Protocol):
    def summaries(self, asins: Sequence[str]) -> dict[str, dict[str, object]]:
        raise NotImplementedError


class SessionNotFound(LookupError):
    pass


class InvalidMessage(ValueError):
    pass


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: UUID
    next_turn: int


@dataclass
class _SessionRecord:
    session_id: UUID
    next_turn: int = 1
    responses: OrderedDict[UUID, dict[str, object]] = field(default_factory=OrderedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    def __init__(
        self,
        agent: AgentProtocol,
        catalog: CatalogProtocol,
        *,
        top_k: int,
        usage_recorder: UsageRecorder | None = None,
        usage_context: dict[str, object] | None = None,
        agent_lock: asyncio.Lock | None = None,
    ) -> None:
        self._agent = agent
        self._catalog = catalog
        self._top_k = top_k
        self._agent_lock = agent_lock or asyncio.Lock()
        self._sessions: dict[UUID, _SessionRecord] = {}
        self._usage_recorder = usage_recorder
        self._usage_context = usage_context or {}

    def fresh(self) -> "SessionManager":
        """Create an empty session boundary around the same configured engine."""
        return SessionManager(
            self._agent,
            self._catalog,
            top_k=self._top_k,
            usage_recorder=self._usage_recorder,
            usage_context=dict(self._usage_context),
            agent_lock=self._agent_lock,
        )

    @property
    def capability_profile(self) -> object | None:
        """Expose the non-secret startup probe snapshot for runtime status reporting."""
        return getattr(self._agent, "profile", None)

    async def create_session(self) -> SessionSnapshot:
        session_id = uuid4()
        async with self._agent_lock:
            _, cancelled = await self._wait_for_worker(
                asyncio.create_task(asyncio.to_thread(self._agent.reset, str(session_id), {}))
            )
        self._sessions[session_id] = _SessionRecord(session_id=session_id)
        if cancelled:
            raise asyncio.CancelledError
        return SessionSnapshot(session_id, 1)

    def get_session(self, session_id: UUID) -> SessionSnapshot | None:
        record = self._sessions.get(session_id)
        return None if record is None else SessionSnapshot(record.session_id, record.next_turn)

    async def send_message(
        self, session_id: UUID, message_id: UUID, message: str
    ) -> dict[str, object]:
        if not message.strip() or len(message) > 4_000:
            raise InvalidMessage("message must contain 1 to 4000 non-whitespace characters")
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFound("session not found")
        async with record.lock:
            cached = record.responses.get(message_id)
            if cached is not None:
                return copy.deepcopy(cached)
            turn = record.next_turn
            started_at = time.monotonic()
            async with self._agent_lock:
                raw, cancelled = await self._wait_for_worker(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self._agent.respond, str(session_id), message, turn, self._top_k
                        )
                    )
                )
                usage_sources = copy.deepcopy(
                    getattr(self._agent, "last_usage_sources", None)
                )
            latency_ms = (time.monotonic() - started_at) * 1000.0
            if not isinstance(raw, dict):
                raise TypeError("Agent.respond() must return a dictionary")
            agent_response = copy.deepcopy(raw)
            usage_metadata = self._record_usage(
                str(session_id), turn, raw, latency_ms, usage_sources
            )
            response_usage = agent_response.get("usage")
            if usage_metadata is not None and isinstance(response_usage, dict):
                response_usage.update(usage_metadata)
            envelope = {
                "session_id": str(session_id),
                "message_id": str(message_id),
                "turn": turn,
                "agent_response": agent_response,
                "products": {},
            }
            record.responses[message_id] = copy.deepcopy(envelope)
            while len(record.responses) > 128:
                record.responses.popitem(last=False)
            record.next_turn += 1
            record.last_accessed_at = datetime.now(timezone.utc)
            if cancelled:
                raise asyncio.CancelledError
            try:
                asins = [
                    str(item.get("parent_asin", ""))
                    for item in agent_response.get("recommendations", [])
                    if isinstance(item, dict) and item.get("parent_asin")
                ]
                products = await asyncio.to_thread(self._catalog.summaries, asins)
            except Exception:
                logger.exception("catalog presentation enrichment failed")
            else:
                envelope["products"] = products
            record.responses[message_id] = copy.deepcopy(envelope)
            return envelope

    def _record_usage(
        self,
        session_id: str,
        turn: int,
        raw: dict,
        latency_ms: float,
        usage_sources: object = None,
    ) -> dict[str, object] | None:
        """Best-effort per-turn usage/cost recording for the dashboard (never raises)."""
        try:
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            sources = self._normalize_usage_sources(usage_sources)
            if sources:
                provider = sources[0]["provider"] if len(sources) == 1 else "mixed"
                model = sources[0]["model"] if len(sources) == 1 else "multiple"
                cost_usd = sum(float(source["cost_usd"]) for source in sources)
                online = any(bool(source["online"]) for source in sources)
            else:
                provider = str(self._usage_context.get("provider") or "none")
                model = str(self._usage_context.get("model") or "")
                cost_usd = estimate_cost_usd(
                    provider, model, prompt_tokens, completion_tokens
                )
                online = (prompt_tokens + completion_tokens) > 0
            event = {
                "session_id": session_id,
                "turn": turn,
                "provider": provider,
                "model": model,
                "retrieval_backend": str(
                    self._usage_context.get("retrieval_backend") or "auto"
                ),
                "rerank_backend": str(self._usage_context.get("rerank_backend") or "none"),
                "output_strategy": str(
                    self._usage_context.get("output_strategy") or "holdback"
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "online": online,
                "latency_ms": round(latency_ms, 3),
            }
            if sources:
                event["usage_sources"] = sources
            if self._usage_recorder is not None:
                self._usage_recorder.record(event)
            return {
                "estimated_cost_usd": cost_usd,
                "sources": sources,
            } if sources else None
        except Exception:
            logger.warning("usage recording failed", exc_info=True)
            return None

    @staticmethod
    def _normalize_usage_sources(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        sources: list[dict[str, object]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or "none")
            model = str(item.get("model") or "")
            prompt_tokens = max(0, int(item.get("prompt_tokens") or 0))
            completion_tokens = max(0, int(item.get("completion_tokens") or 0))
            sources.append(
                {
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": estimate_cost_usd(
                        provider, model, prompt_tokens, completion_tokens
                    ),
                    "online": bool(item.get("online"))
                    or (prompt_tokens + completion_tokens) > 0,
                }
            )
        return sources

    @staticmethod
    async def _wait_for_worker(worker: asyncio.Task[object]) -> tuple[object, bool]:
        cancelled = False
        while True:
            try:
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                if worker.cancelled():
                    raise
                cancelled = True
