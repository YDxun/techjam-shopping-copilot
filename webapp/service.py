from __future__ import annotations

import asyncio
import copy
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID, uuid4

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
    def __init__(self, agent: AgentProtocol, catalog: CatalogProtocol, *, top_k: int) -> None:
        self._agent = agent
        self._catalog = catalog
        self._top_k = top_k
        self._agent_lock = asyncio.Lock()
        self._sessions: dict[UUID, _SessionRecord] = {}

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
            async with self._agent_lock:
                raw, cancelled = await self._wait_for_worker(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self._agent.respond, str(session_id), message, turn, self._top_k
                        )
                    )
                )
            if not isinstance(raw, dict):
                raise TypeError("Agent.respond() must return a dictionary")
            agent_response = copy.deepcopy(raw)
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

    @staticmethod
    async def _wait_for_worker(worker: asyncio.Task[object]) -> tuple[object, bool]:
        cancelled = False
        while True:
            try:
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                cancelled = True
