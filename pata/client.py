"""Async client for an OpenAI-compatible chat endpoint (vLLM on the DigitalOcean droplet)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: float


class ModelClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        schema_hint: bool = False,
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.schema_hint = schema_hint
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)

    @classmethod
    def from_env(cls) -> ModelClient | None:
        """None when no endpoint is configured; the service then runs on rules alone."""
        url = os.environ.get("DO_VLLM_URL")
        if not url:
            return None
        return cls(
            base_url=url,
            api_key=os.environ.get("DO_VLLM_API_KEY"),
            model=os.environ.get("PATA_MODEL", "pata"),
            # The base model needs the schema spelled out; the tuned model does not.
            schema_hint=os.environ.get("PATA_SCHEMA_HINT", "0") == "1",
        )

    async def complete(self, messages: list[dict]) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            # Qwen3.5 thinks by default; triage needs the answer, not the reasoning.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        r = await self._http.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        body = r.json()
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return Completion(
            text=choice["message"]["content"] or "",
            model=body.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason") or "",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
