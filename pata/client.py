"""Async client for an OpenAI-compatible chat endpoint.

The tuned model is served by Tinker's OpenAI-compatible endpoint by default; a self-hosted
vLLM server (deploy/do_vllm.md) or any other OpenAI-compatible host works the same way.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx

TINKER_OAI_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Tinker's endpoint answers in about 1 s, but holds some requests for about 5 s more before
# answering (seen in Sentry, DECISIONS.md D18). A second identical request sent after this
# delay usually lands on the fast path, so we race the two and keep the first answer.
TINKER_HEDGE_MS = 1500


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: float
    hedged: bool = False  # a second request was sent
    hedge_won: bool = False  # and its answer was the one used


def provider(base_url: str) -> str:
    if "thinkingmachines" in base_url:
        return "tinker"
    if "googleapis.com" in base_url:
        return "gcp.gemini"
    return "vllm"


def thinking_off(base_url: str) -> dict:
    """Request fields that turn Qwen3.5 thinking off; each host spells it differently."""
    p = provider(base_url)
    if p == "tinker":
        return {"reasoning_effort": False}
    if p == "gcp.gemini":
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def default_api_key(base_url: str) -> str | None:
    """MODEL_API_KEY, else the provider key already in .env for the known hosts."""
    if key := os.environ.get("MODEL_API_KEY"):
        return key
    p = provider(base_url)
    if p == "tinker":
        return os.environ.get("TINKER_API_KEY")
    if p == "gcp.gemini":
        return os.environ.get("HOSTED_API_KEY")
    return None


class ModelClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        schema_hint: bool = False,
        max_tokens: int = 512,
        timeout: float = 30.0,
        hedge_ms: int | None = None,
    ) -> None:
        """base_url includes the version path, e.g. http://host:8000/v1.

        hedge_ms: send a second identical request if the first has not answered by then.
        None picks the default for the host; 0 turns hedging off.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.schema_hint = schema_hint
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.provider = provider(self.base_url)
        if hedge_ms is None:
            hedge_ms = TINKER_HEDGE_MS if self.provider == "tinker" else 0
        self.hedge_ms = hedge_ms
        self._extra = thinking_off(self.base_url)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)

    @classmethod
    def from_env(cls) -> ModelClient | None:
        """None when no endpoint is configured; the service then runs on rules alone."""
        url = os.environ.get("MODEL_URL")
        if not url:
            return None
        key = default_api_key(url)
        if not key and provider(url) == "tinker":
            return None  # a deploy without a Tinker key runs on the rules alone
        hedge = os.environ.get("PATA_HEDGE_MS")
        return cls(
            base_url=url,
            api_key=key,
            model=os.environ.get("PATA_MODEL", "pata"),
            # The base model needs the schema spelled out; the tuned model does not.
            schema_hint=os.environ.get("PATA_SCHEMA_HINT", "0") == "1",
            hedge_ms=int(hedge) if hedge else None,
        )

    async def _once(self, messages: list[dict]) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            **self._extra,
        }
        t0 = time.perf_counter()
        r = await self._http.post(f"{self.base_url}/chat/completions", json=payload)
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

    async def complete(self, messages: list[dict]) -> Completion:
        if not self.hedge_ms:
            return await self._once(messages)
        t0 = time.perf_counter()
        first = asyncio.create_task(self._once(messages))
        done, _ = await asyncio.wait({first}, timeout=self.hedge_ms / 1000)
        if done:
            return first.result()
        second = asyncio.create_task(self._once(messages))
        pending = {first, second}
        try:
            while True:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                ok = [t for t in done if t.exception() is None]
                if ok or not pending:
                    task = ok[0] if ok else done.pop()
                    c = task.result()  # raises only when both requests failed
                    c.hedged, c.hedge_won = True, task is second
                    c.latency_ms = (time.perf_counter() - t0) * 1000
                    return c
        finally:
            for task in pending:
                task.cancel()

    async def aclose(self) -> None:
        await self._http.aclose()
