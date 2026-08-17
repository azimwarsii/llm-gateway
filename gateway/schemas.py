"""OpenAI-compatible request/response models (the subset the gateway supports)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return " ".join(p.get("text", "") for p in self.content if isinstance(p, dict))
        return ""


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None
    # Anything else (tools, response_format, ...) is passed through untouched.
    model_config = {"extra": "allow"}

    def prompt_text(self) -> str:
        return "\n".join(m.text() for m in self.messages)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Usage()
    # Gateway-specific metadata, namespaced so OpenAI clients ignore it.
    gateway: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "llm-gateway"
    created: int = Field(default_factory=lambda: int(time.time()))


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
