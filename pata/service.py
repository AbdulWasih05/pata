"""The triage agent: model first, schema check, rules fallback. Traced with Sentry agent spans.

Span tree per request (names follow Sentry's gen_ai conventions):

    gen_ai.invoke_agent   invoke_agent pata
      gen_ai.execute_tool   execute_tool rules_prelabel
      gen_ai.chat           chat <model>
      schema.validate
      gen_ai.execute_tool   execute_tool rules_fallback   (only when the model answer fails)

Addresses are customer PII, so prompt and reply text are attached to spans only when
SENTRY_SEND_PROMPTS=1. By default spans carry lengths, decisions, and token counts.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import sentry_sdk

from pata import rules
from pata.client import ModelClient
from pata.prompt import build_messages, parse_output
from pata.schema import TriageInput, TriageOutput

AGENT = "pata"


@dataclass
class TriageResult:
    output: TriageOutput
    source: Literal["model", "rules"]
    fallback_reason: str | None
    latency_ms: float
    rules_decision: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "output": self.output.model_dump(mode="json"),
            "source": self.source,
            "fallback_reason": self.fallback_reason,
            "latency_ms": round(self.latency_ms, 1),
            "rules_decision": self.rules_decision,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _send_prompts() -> bool:
    return os.environ.get("SENTRY_SEND_PROMPTS", "0") == "1"


async def triage(inp: TriageInput, client: ModelClient | None) -> TriageResult:
    t0 = time.perf_counter()
    with sentry_sdk.start_span(op="gen_ai.invoke_agent", name=f"invoke_agent {AGENT}") as agent:
        agent.set_data("gen_ai.operation.name", "invoke_agent")
        agent.set_data("gen_ai.agent.name", AGENT)
        agent.set_data("pata.market", inp.market.value)
        agent.set_data("pata.payment", inp.payment.value)
        agent.set_data("pata.address_chars", len(inp.raw_address))

        with sentry_sdk.start_span(
            op="gen_ai.execute_tool", name="execute_tool rules_prelabel"
        ) as s:
            s.set_data("gen_ai.operation.name", "execute_tool")
            s.set_data("gen_ai.tool.name", "rules_prelabel")
            s.set_data("gen_ai.agent.name", AGENT)
            fallback = rules.triage(inp)
            s.set_data("gen_ai.tool.call.result", fallback.decision.value)

        out, source, reason, comp = fallback, "rules", "no_model_configured", None
        if client is not None:
            messages = build_messages(inp, schema_hint=client.schema_hint)
            try:
                with sentry_sdk.start_span(op="gen_ai.chat", name=f"chat {client.model}") as s:
                    s.set_data("gen_ai.operation.name", "chat")
                    s.set_data("gen_ai.agent.name", AGENT)
                    s.set_data("gen_ai.provider.name", "vllm")
                    s.set_data("gen_ai.request.model", client.model)
                    s.set_data("gen_ai.request.temperature", 0.0)
                    s.set_data("gen_ai.request.max_tokens", client.max_tokens)
                    if _send_prompts():
                        msgs = json.dumps(messages, ensure_ascii=False)
                        s.set_data("gen_ai.input.messages", msgs)
                    comp = await client.complete(messages)
                    s.set_data("gen_ai.response.model", comp.model)
                    s.set_data("gen_ai.usage.input_tokens", comp.input_tokens)
                    s.set_data("gen_ai.usage.output_tokens", comp.output_tokens)
                    s.set_data("gen_ai.usage.total_tokens", comp.input_tokens + comp.output_tokens)
                    s.set_data("gen_ai.response.finish_reasons", [comp.finish_reason])
                    if _send_prompts():
                        s.set_data("gen_ai.output.messages", comp.text)
            except Exception as e:  # network, timeout, HTTP error: rules take over
                reason = f"model_error:{type(e).__name__}"
                sentry_sdk.capture_exception(e)
            if comp is not None:
                with sentry_sdk.start_span(op="schema.validate", name="schema.validate") as s:
                    try:
                        out, source, reason = parse_output(comp.text), "model", None
                        s.set_data("valid", True)
                    except ValueError as e:
                        reason = f"schema_invalid:{type(e).__name__}"
                        s.set_data("valid", False)
                        s.set_data("error", str(e)[:300])
                        s.set_data("finish_reason", comp.finish_reason)

        if source == "rules" and client is not None:
            with sentry_sdk.start_span(
                op="gen_ai.execute_tool", name="execute_tool rules_fallback"
            ) as s:
                s.set_data("gen_ai.operation.name", "execute_tool")
                s.set_data("gen_ai.tool.name", "rules_fallback")
                s.set_data("gen_ai.agent.name", AGENT)
                s.set_data("fallback_reason", reason)

        agent.set_data("pata.decision", out.decision.value)
        agent.set_data("pata.confidence", out.confidence)
        agent.set_data("pata.source", source)
        agent.set_data("pata.fallback_taken", source == "rules" and client is not None)
        agent.set_data("pata.agrees_with_rules", out.decision == fallback.decision)
        if comp is not None:
            agent.set_data("gen_ai.usage.input_tokens", comp.input_tokens)
            agent.set_data("gen_ai.usage.output_tokens", comp.output_tokens)

    return TriageResult(
        output=out,
        source=source,
        fallback_reason=reason,
        latency_ms=(time.perf_counter() - t0) * 1000,
        rules_decision=fallback.decision.value,
        model=comp.model if comp else None,
        input_tokens=comp.input_tokens if comp else 0,
        output_tokens=comp.output_tokens if comp else 0,
    )


async def triage_batch(
    inputs: list[TriageInput], client: ModelClient | None, concurrency: int = 16
) -> list[TriageResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(inp: TriageInput) -> TriageResult:
        async with sem:
            return await triage(inp, client)

    return await asyncio.gather(*(one(i) for i in inputs))
