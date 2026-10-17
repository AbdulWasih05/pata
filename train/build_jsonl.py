"""Turn labelled splits into chat conversations for SFT.

Writes train/out/{train,dev}_conversations.jsonl in the tinker_cookbook conversation format:
{"messages": [system, user, assistant]}. The assistant turn is the output JSON only.

    uv run python -m train.build_jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pata.prompt import build_messages, target_text
from pata.schema import TriageInput, TriageOutput

SPLITS = Path("data/splits")
OUT = Path("train/out")


def to_conversation(row: dict) -> dict:
    inp = TriageInput.model_validate(row["input"])
    out = TriageOutput.model_validate(row["output"])
    msgs = build_messages(inp)
    msgs.append({"role": "assistant", "content": target_text(out)})
    return {"id": row["id"], "market": row["market"], "messages": msgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        rows = [json.loads(x) for x in (SPLITS / f"{split}.jsonl").read_text("utf-8").splitlines()]
        path = args.out / f"{split}_conversations.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(to_conversation(r), ensure_ascii=False) + "\n")
        chars = sum(len(json.dumps(to_conversation(r), ensure_ascii=False)) for r in rows)
        print(f"{split}: {len(rows)} conversations, ~{chars // 4} tokens (chars/4) -> {path}")


if __name__ == "__main__":
    main()
