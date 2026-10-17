"""pata triage --market AE --in orders.csv --out triaged.csv

Uses the model endpoint in DO_VLLM_URL when set, otherwise the rules alone.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from pata.client import ModelClient
from pata.csvio import read_inputs, write_results
from pata.schema import Market
from pata.service import triage_batch


async def _run(args: argparse.Namespace) -> None:
    rows, inputs = read_inputs(Path(args.inp).read_text(encoding="utf-8"), Market(args.market))
    client = None if args.rules_only else ModelClient.from_env()
    try:
        results = await triage_batch(inputs, client)
    finally:
        if client:
            await client.aclose()
    Path(args.out).write_text(write_results(rows, results), encoding="utf-8", newline="")
    print(f"{len(results)} orders -> {args.out}")
    print(dict(Counter(r.output.decision.value for r in results)))
    print("source:", dict(Counter(r.source for r in results)))


def main() -> None:
    ap = argparse.ArgumentParser(prog="pata")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage", help="triage a CSV of orders")
    t.add_argument("--market", choices=[m.value for m in Market], required=True)
    t.add_argument("--in", dest="inp", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
