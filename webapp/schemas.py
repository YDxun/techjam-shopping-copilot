from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    message_id: UUID
    message: str = Field(max_length=4_000)

    @field_validator("message")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class SessionResponse(BaseModel):
    session_id: UUID
    next_turn: int


class ChatResponse(BaseModel):
    session_id: UUID
    message_id: UUID
    turn: int
    agent_response: dict[str, Any]
    products: dict[str, dict[str, Any]]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
