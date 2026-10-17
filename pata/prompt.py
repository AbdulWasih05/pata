"""System prompt, chat formatting and output parsing. Shared by training, eval and serving."""

from __future__ import annotations

import json
import re

from pata.schema import Decision, ReasonCode, TriageInput, TriageOutput

SYSTEM_PROMPT = (
    "You are Pata, a courier dispatch triage model. Read one order's address and context and "
    "decide before dispatch: DELIVERABLE (driver can find it), CALL_FIRST (call the customer "
    "first), or LIKELY_FAIL (hold the parcel). Reply with one JSON object only, no prose."
)

# Appended for zero-shot base-model runs only; the tuned model learns the shape from data.
SCHEMA_HINT = (
    "\n\nJSON keys, in this order: decision ("
    + " | ".join(d.value for d in Decision)
    + "), confidence (0 to 1), reason_codes (list from: "
    + ", ".join(c.value for c in ReasonCode)
    + "; OK only when DELIVERABLE), normalised {country, admin1, city, area, street, building, "
    "unit, landmark, postcode (Indian PIN or null), makani (UAE 10 digits or null), po_box "
    "(or null)}, driver_note (one line, max 120 chars), call_script (one line if CALL_FIRST, "
    "else null). Copy field values from the address text as written; use \"\" when absent."
)


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def user_content(inp: TriageInput) -> str:
    return _dumps(inp.model_dump(mode="json"))


def build_messages(inp: TriageInput, schema_hint: bool = False) -> list[dict]:
    system = SYSTEM_PROMPT + (SCHEMA_HINT if schema_hint else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content(inp)},
    ]


def target_text(out: TriageOutput) -> str:
    return _dumps(out.model_dump(mode="json"))


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply. Raises ValueError if there is none."""
    t = _FENCE.sub("", _THINK.sub("", text).strip())
    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def parse_output(text: str) -> TriageOutput:
    """Parse and validate a model reply against the contract. Raises on any failure."""
    return TriageOutput.model_validate(extract_json(text))
