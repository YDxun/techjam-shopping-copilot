import asyncio
import copy
import threading
import time
from uuid import uuid4

import pytest

from webapp.catalog import CatalogError
from webapp.service import InvalidMessage, SessionManager, SessionNotFound


class FakeCatalog:
    def summaries(self, asins):
        return {asin: {"parent_asin": asin, "title": f"Product {asin}"} for asin in asins}


class FakeAgent:
    def __init__(self) -> None:
        self.resets = []
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.resets.append((session_id, copy.deepcopy(user_profile)))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        response = {
            "message": f"reply {turn}",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": "A1"}],
            "usage": {"prompt_tokens": turn, "completion_tokens": 1},
        }
        self.calls.append((session_id, user_message, turn, top_k))
        with self.guard:
            self.active -= 1
        return response


def test_session_uses_empty_profile_and_allows_turn_eleven() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        for turn in range(1, 12):
            result = await manager.send_message(session.session_id, uuid4(), f"message {turn}")
            assert result["turn"] == turn
        assert agent.resets == [(str(session.session_id), {})]
        assert [call[2] for call in agent.calls] == list(range(1, 12))

    asyncio.run(scenario())


def test_duplicate_message_id_returns_cached_exact_agent_response() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        message_id = uuid4()
        first = await manager.send_message(session.session_id, message_id, "cotton")
        second = await manager.send_message(session.session_id, message_id, "ignored retry body")
        assert second == first
        assert len(agent.calls) == 1
        assert first["agent_response"] == {
            "message": "reply 1",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": "A1"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    asyncio.run(scenario())


def test_agent_calls_are_global_serialized_across_sessions() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        left = await manager.create_session()
        right = await manager.create_session()
        await asyncio.gather(
            manager.send_message(left.session_id, uuid4(), "left"),
            manager.send_message(right.session_id, uuid4(), "right"),
        )
        assert agent.max_active == 1
        assert sorted(call[2] for call in agent.calls) == [1, 1]

    asyncio.run(scenario())


def test_agent_failure_does_not_advance_turn_and_valid_text_is_unchanged() -> None:
    class FailsOnceAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            if self.fail:
                self.fail = False
                raise RuntimeError("temporary failure")
            return super().respond(session_id, user_message, turn, top_k)

    async def scenario() -> None:
        agent = FailsOnceAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        with pytest.raises(RuntimeError):
            await manager.send_message(session.session_id, uuid4(), " first ")
        result = await manager.send_message(session.session_id, uuid4(), " second ")
        assert result["turn"] == 1
        assert agent.calls[0][1] == " second "

    asyncio.run(scenario())


def test_invalid_messages_and_unknown_sessions_do_not_call_agent() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FakeCatalog(), top_k=10)
        session = await manager.create_session()
        with pytest.raises(InvalidMessage):
            await manager.send_message(session.session_id, uuid4(), "   ")
        with pytest.raises(InvalidMessage):
            await manager.send_message(session.session_id, uuid4(), "x" * 4001)
        with pytest.raises(SessionNotFound):
            await manager.send_message(uuid4(), uuid4(), "valid")
        assert agent.calls == []

    asyncio.run(scenario())


def test_catalog_failure_after_agent_response_is_cached_without_reinvocation() -> None:
    class FailingCatalog:
        def summaries(self, asins):
            raise CatalogError("presentation file changed")

    async def scenario() -> None:
        agent = FakeAgent()
        manager = SessionManager(agent, FailingCatalog(), top_k=10)
        session = await manager.create_session()
        message_id = uuid4()
        first = await manager.send_message(session.session_id, message_id, "cotton")
        retry = await manager.send_message(session.session_id, message_id, "cotton")
        assert first == retry
        assert first["products"] == {}
        assert manager.get_session(session.session_id).next_turn == 2
        assert len(agent.calls) == 1

    asyncio.run(scenario())
