from __future__ import annotations

from typing import Protocol

from agent.dialogue.models import RecognitionRequest, RecognitionResult


class IntentRecognizer(Protocol):
    def recognize(self, request: RecognitionRequest) -> RecognitionResult: ...
