"""Regenerate docs/examples_out.json: the example outputs quoted in the README and the post.

    uv run --env-file .env python -m eval.examples

Runs each input in docs/examples_in.json through pata.service.triage with the serving
checkpoint and records which checkpoint answered, so every quoted output traces to a run.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from eval.latency_bench import CHECKPOINT
from pata.client import TINKER_OAI_URL, ModelClient, default_api_key
from pata.schema import TriageInput
from pata.service import triage

IN = Path("docs/examples_in.json")
OUT = Path("docs/examples_out.json")


async def main() -> None:
    client = ModelClient(TINKER_OAI_URL, default_api_key(TINKER_OAI_URL), CHECKPOINT)
    try:
        examples = []
        for raw in json.loads(IN.read_text("utf-8")):
            res = await triage(TriageInput.model_validate(raw), client)
            examples.append({"input": raw, "output": res.output.model_dump(mode="json"),
                             "source": res.source})
    finally:
        await client.aclose()
    OUT.write_text(json.dumps({
        "checkpoint": CHECKPOINT,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "examples": examples,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for e in examples:
        print(e["source"], e["output"]["decision"], "|", e["input"]["raw_address"])


if __name__ == "__main__":
    asyncio.run(main())
