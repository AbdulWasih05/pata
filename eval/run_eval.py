"""Evaluate a triage backend per market and write eval/results.md.

    uv run python -m eval.run_eval --checkpoint rules
    uv run python -m eval.run_eval --checkpoint base                  # zero-shot Qwen3.5-4B (Linux)
    uv run python -m eval.run_eval --checkpoint tinker://<run>/...    # tuned, via Tinker (Linux)
    uv run python -m eval.run_eval --checkpoint https://<host>/v1    # OpenAI-compatible endpoint

Test set: the independently reviewed gold rows (data/gold/*_handcheck.csv) when
any are filled; otherwise the silver test split, and the table says so.

Every run is appended to eval/results.jsonl; eval/results.md is regenerated from it, so numbers
in the post always trace back to a logged run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

from pata.client import ModelClient, default_api_key
from pata.prompt import build_messages, parse_output
from pata.rules import triage as rules_triage
from pata.schema import Decision, TriageInput, TriageOutput

TEST = Path("data/splits/test.jsonl")
GOLD = Path("data/gold")
OUT = Path("eval/out")
RESULTS_JSONL = Path("eval/results.jsonl")
RESULTS_MD = Path("eval/results.md")
BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5_disable_thinking"
FIELDS = ("building", "area", "postcode")
DECISIONS = [d.value for d in Decision]


# ---------------------------------------------------------------- test sets


def load_test(split: str) -> tuple[list[dict], str]:
    rows = {json.loads(x)["id"]: json.loads(x) for x in TEST.read_text("utf-8").splitlines()}
    gold = []
    for f in sorted(GOLD.glob("*_handcheck.csv")):
        with f.open(encoding="utf-8") as fh:
            for g in csv.DictReader(fh):
                d = (g.get("gold_decision") or "").strip().upper()
                if d in DECISIONS and g["id"] in rows:
                    r = json.loads(json.dumps(rows[g["id"]]))
                    r["gold_decision"] = d
                    gold.append(r)
    if split == "gold" or (split == "auto" and gold):
        if not gold:
            raise SystemExit("no reviewed gold rows yet; use --split silver")
        return gold, "gold (blind review, not rules)"
    for r in rows.values():
        r["gold_decision"] = r["output"]["decision"]
    return list(rows.values()), "silver (rules-labelled)"


# ---------------------------------------------------------------- backends


class Backend:
    name = ""

    async def predict(self, inp: TriageInput) -> tuple[str, float]:
        """Return (raw reply text, latency ms)."""
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class RulesBackend(Backend):
    name = "rules"

    async def predict(self, inp):
        t0 = time.perf_counter()
        out = rules_triage(inp)
        return out.model_dump_json(), (time.perf_counter() - t0) * 1000


class HttpBackend(Backend):
    def __init__(self, url: str, schema_hint: bool):
        # Generous limits: a hosted baseline may reason before it answers.
        self.client = ModelClient(url, default_api_key(url), os.environ.get("PATA_MODEL", "pata"),
                                  schema_hint=schema_hint, max_tokens=4096, timeout=180.0)
        self.name = f"http:{self.client.model}"

    async def predict(self, inp):
        msgs = build_messages(inp, schema_hint=self.client.schema_hint)
        # Retry rate limits and provider overloads, so they are not scored as model errors.
        for wait in (5, 15, 45, 90, None):
            try:
                c = await self.client.complete(msgs)
                return c.text, c.latency_ms
            except httpx.HTTPStatusError as e:
                if wait is None or e.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                await asyncio.sleep(wait)

    async def aclose(self):
        await self.client.aclose()


class TinkerBackend(Backend):
    """Samples through Tinker with the same renderer used in training. Linux only."""

    def __init__(self, checkpoint: str, schema_hint: bool, max_tokens: int = 512):
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.tinker = tinker
        self.renderers = renderers
        self.renderer = renderers.get_renderer(RENDERER, get_tokenizer(BASE_MODEL))
        sc = tinker.ServiceClient()
        if checkpoint == "base":
            self.sampler = sc.create_sampling_client(base_model=BASE_MODEL)
            self.name = f"base:{BASE_MODEL}"
        else:
            self.sampler = sc.create_sampling_client(model_path=checkpoint)
            self.name = f"tuned:{checkpoint}"
        self.schema_hint = schema_hint
        self.params = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=0.0, stop=self.renderer.get_stop_sequences()
        )

    async def predict(self, inp):
        prompt = self.renderer.build_generation_prompt(build_messages(inp, self.schema_hint))
        t0 = time.perf_counter()
        res = await self.sampler.sample_async(prompt=prompt, num_samples=1,
                                              sampling_params=self.params)
        ms = (time.perf_counter() - t0) * 1000
        msg, _ = self.renderer.parse_response(res.sequences[0].tokens)
        return self.renderers.get_text_content(msg), ms


def make_backend(checkpoint: str, schema_hint: bool | None) -> Backend:
    if checkpoint == "rules":
        return RulesBackend()
    if checkpoint.startswith(("http://", "https://")):
        return HttpBackend(checkpoint, bool(schema_hint))
    # The base model gets the schema spelled out; a tuned checkpoint learned it.
    hint = (checkpoint == "base") if schema_hint is None else schema_hint
    return TinkerBackend(checkpoint, hint)


# ---------------------------------------------------------------- metrics


def _norm(v) -> str:
    return " ".join(str(v or "").lower().split())


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    f1s = []
    for c in DECISIONS:
        tp = sum(p == c and g == c for g, p in pairs)
        fp = sum(p == c and g != c for g, p in pairs)
        fn = sum(p != c and g == c for g, p in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def score(preds: list[dict]) -> dict:
    n = len(preds)
    valid = [p for p in preds if p["valid"]]
    pairs = [(p["gold"], p["pred"] or "INVALID") for p in preds]
    pairs_fb = [(p["gold"], p["pred"] or p["rules"]) for p in preds]  # serve-time behaviour
    not_deliv = [p for p in preds if p["gold"] != "DELIVERABLE"]
    fd = sum((p["pred"] or p["rules"]) == "DELIVERABLE" for p in not_deliv)
    fields = {
        f: (sum(_norm(p["fields"].get(f)) == _norm(p["truth"].get(f)) for p in valid) / len(valid)
            if valid else 0.0)
        for f in FIELDS
    }
    lat = [p["latency_ms"] for p in preds]
    return {
        "n": n,
        "schema_valid": len(valid) / n,
        "accuracy": sum(g == p for g, p in pairs) / n,
        "accuracy_with_fallback": sum(g == p for g, p in pairs_fb) / n,
        "macro_f1": macro_f1(pairs),
        "false_deliverable": fd / len(not_deliv) if not_deliv else 0.0,
        "false_deliverable_n": f"{fd}/{len(not_deliv)}",
        **{f"field_{k}": v for k, v in fields.items()},
        "p50_latency_ms": statistics.median(lat) if lat else 0.0,
        "confusion": dict(Counter(f"{g}->{p}" for g, p in pairs)),
    }


# ---------------------------------------------------------------- run


async def run(backend: Backend, rows: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(r: dict) -> dict:
        nonlocal done
        inp = TriageInput.model_validate(r["input"])
        rules_dec = rules_triage(inp).decision.value
        async with sem:
            try:
                text, ms = await backend.predict(inp)
                err = ""
            except Exception as e:  # count a failed call as an invalid answer
                text, ms, err = "", 0.0, f"{type(e).__name__}: {e}"[:200]
        out: TriageOutput | None = None
        try:
            out = parse_output(text)
        except ValueError as e:
            err = err or str(e)[:200]
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(rows)}", flush=True)
        return {
            "id": r["id"],
            "market": r["market"],
            "gold": r["gold_decision"],
            "rules": rules_dec,
            "pred": out.decision.value if out else None,
            "valid": out is not None,
            "fields": out.normalised.model_dump(mode="json") if out else {},
            "truth": r["output"]["normalised"],
            "latency_ms": ms,
            "error": err,
            "reply": text if out is None else "",
        }

    return await asyncio.gather(*(one(r) for r in rows))


def write_markdown() -> None:
    runs = [json.loads(x) for x in RESULTS_JSONL.read_text("utf-8").splitlines() if x.strip()]
    latest: dict[tuple, dict] = {}
    for r in runs:  # keep the latest run per (label, test set, market)
        for market, m in r["metrics"].items():
            latest[(r["label"], r["test_set"], market)] = {**r, "m": m, "market": market}
    lines = [
        "# Eval results",
        "",
        "Generated by `eval/run_eval.py` from `eval/results.jsonl`. Do not edit by hand.",
        "",
        "False-DELIVERABLE rate: share of parcels that should not ship as-is (gold CALL_FIRST or "
        "LIKELY_FAIL) that the system would send out (model answer, or rules when the answer is "
        "invalid). This is the mistake that costs money. Accuracy counts an invalid answer as "
        "wrong; accuracy w/ fallback is what the gateway serves.",
        "",
    ]
    for test_set in sorted({k[1] for k in latest}):
        lines += [f"## Test set: {test_set}", ""]
        for market in ("IN", "AE"):
            rows = [v for k, v in latest.items() if k[1] == test_set and k[2] == market]
            if not rows:
                continue
            lines += [
                f"### {market}",
                "",
                "| System | n | Schema-valid | Accuracy | Acc. w/ fallback | Macro-F1 | "
                "False-DELIVERABLE | Building | Area | Postcode | p50 latency |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for v in sorted(rows, key=lambda x: x["date"]):
                m = v["m"]
                lines.append(
                    f"| {v['label']} | {m['n']} | {m['schema_valid']:.1%} | {m['accuracy']:.1%} | "
                    f"{m['accuracy_with_fallback']:.1%} | {m['macro_f1']:.3f} | "
                    f"{m['false_deliverable']:.1%} ({m['false_deliverable_n']}) | "
                    f"{m['field_building']:.1%} | {m['field_area']:.1%} | "
                    f"{m['field_postcode']:.1%} | {m['p50_latency_ms']:.0f} ms |"
                )
            lines.append("")
    lines += ["## Runs", ""]
    for r in runs:
        lines.append(f"* {r['date']} `{r['label']}`: {r['backend']} on {r['test_set']}"
                     f"{' ' + r['note'] if r.get('note') else ''}")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rescore(path: Path, rows: list[dict]) -> list[dict]:
    """Saved predictions re-scored against the current labels, without sampling again."""
    by_id = {r["id"]: r for r in rows}
    preds = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    return [
        {**p, "gold": by_id[p["id"]]["gold_decision"],
         "truth": by_id[p["id"]]["output"]["normalised"]}
        for p in preds
        if p["id"] in by_id
    ]


async def amain(args: argparse.Namespace) -> None:
    rows, test_set = load_test(args.split)
    if args.market:
        rows = [r for r in rows if r["market"] == args.market]
    if args.limit:
        rows = rows[: args.limit]
    if args.rescore:
        preds = rescore(args.rescore, rows)
        backend_name = f"rescored {args.rescore.name}"
        print(f"{backend_name}: {len(preds)} rows of {test_set}")
    else:
        backend = make_backend(args.checkpoint, args.schema_hint)
        backend_name = backend.name
        print(f"{backend.name} on {len(rows)} rows of {test_set}")
        try:
            preds = await run(backend, rows, args.concurrency)
        finally:
            await backend.aclose()

    label = args.label or backend_name
    OUT.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    if test_set.startswith("gold"):
        safe += "__gold"  # silver and gold runs of the same system keep separate files
    if not args.rescore:  # keep the original predictions file untouched
        with (OUT / f"{safe}.jsonl").open("w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics = {m: score([p for p in preds if p["market"] == m])
               for m in ("IN", "AE") if any(p["market"] == m for p in preds)}
    record = {"date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "label": label,
              "backend": backend_name, "test_set": test_set, "note": args.note,
              "metrics": metrics}
    if not args.no_log:
        with RESULTS_JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_markdown()
    for m, s in metrics.items():
        print(f"{m}: " + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in s.items() if k != "confusion"))
        print(f"    confusion: {s['confusion']}")
    bad = Counter(p["error"][:60] for p in preds if not p["valid"])
    if bad:
        print("invalid answers:", bad.most_common(5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="rules | base | tinker://... | http(s)://endpoint")
    ap.add_argument("--rescore", type=Path,
                    help="re-score a saved eval/out/*.jsonl against the current labels")
    ap.add_argument("--label", help="row name in results.md (default: backend name)")
    ap.add_argument("--split", choices=["auto", "gold", "silver"], default="auto")
    ap.add_argument("--market", choices=["IN", "AE"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--schema-hint", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true", help="do not append to results")
    args = ap.parse_args()
    if not (args.checkpoint or args.rescore):
        ap.error("--checkpoint or --rescore is required")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
