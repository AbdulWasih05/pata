"""Silver-label both markets with pata.rules and write train / dev / test splits.

Decision, reason codes and confidence come from the rules. Normalised fields come from the
source truth (gold spans for India, the generator for UAE). Rows where the two disagree on
whether a building is identified are dropped, so a target never says NO_BUILDING_ID next to
a filled-in building field. For UAE, rows where the rules disagree with the generator's
intended decision are dropped too. Drop counts are printed and written to data/splits/STATS.md.

    uv run python -m data.label
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from data.augment import augment
from pata.rules import call_script, confidence, decide, driver_note, extract_features
from pata.schema import Decision, Normalised, TriageInput, TriageOutput

IN_RAW = Path("data/in/in_raw.jsonl")
AE_RAW = Path("data/ae/ae_raw.jsonl")
SPLITS = Path("data/splits")
GOLD = Path("data/gold")

TRAIN_PER_MARKET = 3000
DEV_PER_MARKET = 200
HANDCHECK_PER_MARKET = 150


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def label(rec: dict) -> tuple[dict | None, str]:
    """Return (labelled row, "") or (None, drop reason)."""
    inp = TriageInput.model_validate(rec["input"])
    truth = Normalised.model_validate(rec["truth"])
    f = extract_features(inp)
    decision, codes = decide(inp, f)

    rules_has = bool(f.building or f.makani or (f.unit and f.street))
    truth_has = bool(truth.building or truth.unit or truth.makani)
    if rules_has != truth_has:
        return None, "building_disagree"
    intended = rec.get("intended_decision")
    if intended and intended != decision.value:
        return None, "intent_disagree"

    out = TriageOutput(
        decision=decision,
        confidence=confidence(decision, codes, f),
        reason_codes=codes,
        normalised=truth,
        driver_note=driver_note(decision, codes, truth),
        call_script=call_script(codes) if decision == Decision.CALL_FIRST else None,
    )
    row = {
        "id": rec["id"],
        "market": inp.market.value,
        "pool": rec["pool"],
        "pattern": rec.get("pattern", ""),
        "augment": rec.get("augment", []),
        "input": inp.model_dump(mode="json"),
        "output": out.model_dump(mode="json"),
    }
    return row, ""


def _stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["output"]["decision"], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    picked: list[dict] = []
    while len(picked) < n and any(by.values()):
        for v in by.values():
            if v and len(picked) < n:
                picked.append(v.pop())
    return picked


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_handcheck(market: str, rows: list[dict]) -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    path = GOLD / f"{market.lower()}_handcheck.csv"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            if any((r.get("gold_decision") or "").strip() for r in csv.DictReader(fh)):
                # Human work is never overwritten; a fresh sheet goes next to it instead.
                path = path.with_suffix(".new.csv")
                print(f"{market}: hand-check sheet has human labels, writing {path} instead")
    cols = ["id", "reviewer", "raw_address", "phone_present", "payment", "declared_pincode",
            "city_hint", "silver_decision", "silver_reasons", "gold_decision", "notes"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            inp, out = r["input"], r["output"]
            w.writerow({
                "id": r["id"],
                "reviewer": "Wasih" if i % 2 == 0 else "Usman",
                "raw_address": inp["raw_address"],
                "phone_present": inp["phone_present"],
                "payment": inp["payment"],
                "declared_pincode": inp["declared_pincode"] or "",
                "city_hint": inp["city_hint"] or "",
                "silver_decision": out["decision"],
                "silver_reasons": " ".join(out["reason_codes"]),
                "gold_decision": "",
                "notes": "",
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sources = {
        "IN": [augment(r, args.seed) for r in _read(IN_RAW)],
        "AE": _read(AE_RAW),
    }
    train, dev, test, stats = [], [], [], []
    for market, recs in sources.items():
        drops: Counter = Counter()
        rows = []
        for rec in recs:
            row, why = label(rec)
            if row:
                rows.append(row)
            else:
                drops[why] += 1
        pool_train = [r for r in rows if r["pool"] == "train"]
        pool_test = [r for r in rows if r["pool"] == "test"]
        rng.shuffle(pool_train)
        m_dev = _stratified(pool_train, DEV_PER_MARKET, rng)
        dev_ids = {r["id"] for r in m_dev}
        m_train = [r for r in pool_train if r["id"] not in dev_ids][:TRAIN_PER_MARKET]
        hand = _stratified(pool_test, HANDCHECK_PER_MARKET, rng)
        _write_handcheck(market, hand)
        train += m_train
        dev += m_dev
        test += pool_test

        def dist(rs):
            return dict(Counter(r["output"]["decision"] for r in rs))

        stats.append(
            f"## {market}\n\n"
            f"* source records: {len(recs)}, labelled: {len(rows)}, dropped: {dict(drops)}\n"
            f"* train: {len(m_train)} {dist(m_train)}\n"
            f"* dev: {len(m_dev)} {dist(m_dev)}\n"
            f"* test (silver): {len(pool_test)} {dist(pool_test)}\n"
            f"* hand-check sheet: {len(hand)} {dist(hand)}\n"
        )

    rng.shuffle(train)
    _write_jsonl(SPLITS / "train.jsonl", train)
    _write_jsonl(SPLITS / "dev.jsonl", dev)
    _write_jsonl(SPLITS / "test.jsonl", test)
    header = "# Split statistics\n\nGenerated by `uv run python -m data.label`.\n\n"
    text = header + "\n".join(stats)
    (SPLITS / "STATS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
