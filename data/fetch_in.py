"""Build the India record pool from public Hugging Face datasets (all Apache 2.0).

Sources (see DECISIONS.md D11):
  * gagan1985/indian-addresses-gold, MCA rows only: real company-registry addresses with
    span-labelled fields. Bank rows are KYC customer data and are excluded.
  * gagan1985/indian-addresses-raw, MCA rows in QUARANTINED state (no pincode found).
  * abr-6024/indian-address-parsing-synthetic: synthetic customer-typed addresses with truth.

Writes data/in/in_raw.jsonl in the same record format as data/ae_generator.py.

    uv run python -m data.fetch_in --n 4000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from pata.markets import IN_PIN_PREFIX
from pata.rules import extract_features, normalise
from pata.schema import Market, Normalised, TriageInput

RAW_DIR = Path("data/raw")
OUT = Path("data/in/in_raw.jsonl")

STATE_CODES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CG": "Chhattisgarh", "CT": "Chhattisgarh",
    "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala",
    "LA": "Ladakh", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
    "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OR": "Odisha", "OD": "Odisha",
    "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TG": "Telangana", "TS": "Telangana", "TR": "Tripura", "UP": "Uttar Pradesh",
    "UK": "Uttarakhand", "UT": "Uttarakhand", "WB": "West Bengal",
}

# ~10% of PIN regions (first three digits) are held out for test.
TEST_PIN_FRACTION = 0.10


def _fetch(repo: str, filename: str) -> Path:
    return Path(
        hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW_DIR / repo.split("/")[1])
    )


def clean_registry_text(s: str) -> str:
    """Drop registry artefacts a customer would never type."""
    s = re.sub(r"\bUnclassified\b", " ", s)
    s = re.sub(r"(?<!\d)0{6}(?!\d)", " ", s)  # placeholder pincode
    s = re.sub(r",\s*-1\s*,", ", ", s)
    s = re.sub(r"\s+,", ",", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _state(v) -> str:
    if not isinstance(v, str) or not v.strip():
        return ""
    v = v.strip()
    return STATE_CODES.get(v.upper(), v.title())


def _s(v) -> str:
    """Strip, and collapse a value the gold set repeats from two spans ("RANI RANI")."""
    if not isinstance(v, str):
        return ""
    v = re.sub(r"\s+", " ", v).strip()
    half = len(v) // 2
    if len(v) % 2 == 1 and v[:half].lower() == v[half + 1 :].lower():
        return v[:half]
    return v


def _first(*vals) -> str:
    return next((x for x in map(_s, vals) if x), "")


def _test_regions(pins: list[str], seed: int) -> set[str]:
    """Pick whole PIN regions until they cover ~10% of rows; skip regions over 3% each."""
    counts = Counter(p[:3] for p in pins if p)
    regions = sorted(counts)
    random.Random(seed).shuffle(regions)
    chosen, covered, target = set(), 0, TEST_PIN_FRACTION * len(pins)
    for reg in regions:
        if covered >= target:
            break
        if counts[reg] <= 0.03 * len(pins):
            chosen.add(reg)
            covered += counts[reg]
    return chosen


def _record(rid, source, raw, truth: Normalised, pool, pin=None) -> dict:
    inp = TriageInput(market=Market.IN, raw_address=raw[:1000])
    return {
        "id": rid,
        "pattern": source,
        "pool": pool,
        "input": inp.model_dump(mode="json"),
        "truth": truth.model_dump(mode="json"),
        "pin_region": (pin or "")[:3],
    }


def gold_records(test_regions: set[str]) -> list[dict]:
    g = pd.read_parquet(_fetch("gagan1985/indian-addresses-gold", "data/gold_addresses.parquet"))
    g = g[g.source_type == "mca"]
    out = []
    for r in g.itertuples():
        pin = re.sub(r"\D", "", _s(r.pincode))
        house_name, house_no = _s(r.houseName), _s(r.houseNumber)
        poi = _s(r.poi)
        if poi.lower().startswith(("c/o", "c/o.", "co ")):
            poi = ""  # care-of names are not landmarks
        truth = Normalised(
            country=Market.IN,
            admin1=_state(r.state),
            city=_first(r.city, r.district).title(),
            area=_first(r.subLocality, r.locality, r.village, r.subsubLocality),
            street=_s(r.street),
            building=house_name or house_no,
            unit=house_no if house_name else "",
            landmark=poi,
            postcode=pin if len(pin) == 6 else None,
        )
        pool = "test" if pin[:3] in test_regions else "train"
        out.append(_record(f"in-gold-{r.record_id}", "in_gold", clean_registry_text(r.raw_address),
                           truth, pool, pin))
    return out


def quarantined_records(n: int, seed: int) -> list[dict]:
    """MCA rows with no pincode. Truth comes from the rules (no gold exists for these)."""
    path = _fetch("gagan1985/indian-addresses-raw", "data/raw_addresses.parquet")
    df = pd.read_parquet(path, columns=["record_id", "source_type", "raw_address",
                                        "lifecycle_state"])
    q = df[(df.source_type == "mca") & (df.lifecycle_state == "QUARANTINED")]
    q = q.sample(min(n, len(q)), random_state=seed)
    out = []
    for i, r in enumerate(q.itertuples()):
        raw = clean_registry_text(r.raw_address)
        if len(raw) < 4:
            continue
        inp = TriageInput(market=Market.IN, raw_address=raw[:1000])
        truth = normalise(inp, extract_features(inp))
        out.append(_record(f"in-quar-{r.record_id}", "in_quarantined", raw, truth,
                           "test" if i % 10 == 0 else "train"))
    return out


def synthetic_records(n_train: int) -> list[dict]:
    out = []
    for split, pool, limit in (("train", "train", n_train), ("test", "test", None)):
        path = _fetch("abr-6024/indian-address-parsing-synthetic", f"{split}.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for i, row in enumerate(rows[:limit]):
            t = row["truth"]
            pin = re.sub(r"\D", "", t.get("pincode", ""))
            truth = Normalised(
                country=Market.IN,
                admin1=t.get("state", ""),
                city=t.get("city", ""),
                area=t.get("area", ""),
                street=t.get("street", ""),
                building=t.get("building", ""),
                unit=t.get("flat", ""),
                landmark=t.get("landmark", ""),
                postcode=pin or None,
            )
            raw = row["prompt"][-1]["content"]
            out.append(_record(f"in-synth-{split}-{i:05d}", "in_synth", raw, truth, pool, pin))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="target train-pool size")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    g = pd.read_parquet(_fetch("gagan1985/indian-addresses-gold", "data/gold_addresses.parquet"))
    pins = [re.sub(r"\D", "", _s(p)) for p in g[g.source_type == "mca"].pincode]
    test_regions = _test_regions([p for p in pins if p[:2] in IN_PIN_PREFIX], args.seed)

    gold = gold_records(test_regions)
    quar = quarantined_records(400, args.seed)
    n_gold_train = sum(r["pool"] == "train" for r in gold)
    n_synth = max(0, args.n - n_gold_train - sum(r["pool"] == "train" for r in quar))
    synth = synthetic_records(n_synth)
    records = gold + quar + synth

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {args.out}")
    print(Counter((r["pattern"], r["pool"]) for r in records))


if __name__ == "__main__":
    main()
