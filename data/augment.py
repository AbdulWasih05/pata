"""Customer-style noise and order context for India records.

Registry addresses are formal; customers are not. Each transform edits the raw text and keeps
the truth fields in step with it, so the target never mentions something the input lacks.
Pure functions driven by a seeded Random; the same record id always gives the same variant.
"""

from __future__ import annotations

import copy
import random
import re

# Two disjoint landmark pools so test-pool landmarks never appear in train.
LANDMARKS = {
    "train": [
        "Hanuman Mandir", "SBI Bank", "HP petrol pump", "Govt School", "Shiv Temple",
        "Bus Stand", "Big Bazaar", "Railway Crossing", "Water Tank", "Post Office",
        "Apollo Pharmacy", "Masjid", "Gurudwara", "Panchayat Office", "Reliance Fresh",
        "Axis Bank ATM", "Sai Baba Mandir", "Primary Health Centre", "Indian Oil pump",
        "Durga Mandir",
    ],
    "test": [
        "Kali Bari", "Canara Bank", "Bharat Petrol Pump", "Kendriya Vidyalaya", "Jain Mandir",
        "Auto Stand", "D-Mart", "Old Bridge",
    ],
}
LANDMARK_CUES = ["near", "nr", "opp", "opposite", "behind", "beside", "next to"]
CALL_PHRASES = [
    "call before coming", "call me on arrival", "pls call before delivery",
    "aane se pehle call karna", "call kar lena", "will share location on whatsapp",
    "ask watchman", "call me, gate is locked",
]
ABBREV = [
    (r"\bnear\b", "nr"), (r"\bopposite\b", "opp"), (r"\broad\b", "rd"), (r"\bstreet\b", "st"),
    (r"\bhouse no\.?", "h no"), (r"\bapartments?\b", "apts"), (r"\bbuilding\b", "bldg"),
    (r"\bsector\b", "sec"), (r"\bnagar\b", "ngr"),
]


def _phone(rng: random.Random) -> str:
    return str(rng.choice("6789")) + "".join(str(rng.randrange(10)) for _ in range(9))


def _drop_pin(text: str, pin: str) -> str:
    pat = re.escape(pin[:3]) + r"\s?" + re.escape(pin[3:])
    text = re.sub(r"[-,\s]*(?:pin(?:code)?[:\s-]*)?" + pat, "", text, flags=re.I)
    return text.strip(" ,-")


def _drop_segment(text: str, value: str) -> str | None:
    """Remove the comma segment containing value; None if value is not cleanly present."""
    if not value:
        return None
    parts = [p for p in text.split(",")]
    keep = [p for p in parts if value.lower() not in p.lower()]
    if len(keep) == len(parts) or not keep:
        return None
    return ",".join(keep).strip(" ,")


def _typo(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 4]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    words[i] = w[:j] + w[j + 1 :] if rng.random() < 0.5 else w[:j] + w[j] + w[j:]
    return " ".join(words)


TEXT_FIELDS = ("building", "unit", "street", "area", "landmark")


def _sync_truth(truth: dict, tags: list[str], applied: list[tuple[str, str]]) -> None:
    """Apply the same surface edits to truth fields so they still copy from the text."""
    for k in TEXT_FIELDS:
        v = truth.get(k) or ""
        if not v:
            continue
        for pat, rep in applied:
            v = re.sub(pat, rep, v, flags=re.I)
        if "lower" in tags:
            v = v.lower()
        if "no_commas" in tags:
            v = re.sub(r"\s*,\s*", " ", v)
        truth[k] = v


def augment(rec: dict, seed: int = 0) -> dict:
    """Return a new record with customer noise and order context. Truth tracks the text."""
    rng = random.Random(f"{seed}:{rec['id']}")
    out = copy.deepcopy(rec)
    inp, truth = out["input"], out["truth"]
    text = inp["raw_address"]
    tags: list[str] = []
    pin = truth.get("postcode") or ""
    protected = {truth.get("building", ""), truth.get("unit", "")}

    # Structural edits first (they change what the truth should say).
    if truth.get("building") and rng.random() < 0.08:
        dropped = _drop_segment(text, truth["building"])
        if dropped:
            text, truth["building"], truth["unit"] = dropped, "", ""
            protected = set()
            tags.append("drop_building")
    if pin and rng.random() < 0.10 and pin[:3] in text.replace(" ", ""):
        text = _drop_pin(text, pin)
        truth["postcode"] = None
        tags.append("drop_pin")
    if rng.random() < 0.20 and not truth.get("landmark"):
        lm = rng.choice(LANDMARKS[rec.get("pool", "train")])
        if rng.random() < 0.25:
            phrase = f"{lm} ke paas"
        else:
            phrase = f"{rng.choice(LANDMARK_CUES)} {lm}"
        # Landmark goes before the locality tail, like customers write it.
        parts = text.split(",")
        at = rng.randrange(1, len(parts) + 1) if len(parts) > 1 else 1
        parts.insert(at, " " + phrase)
        text = ",".join(parts)
        truth["landmark"] = phrase
        tags.append("add_landmark")
    if rng.random() < 0.12:
        text = f"{text}, {rng.choice(CALL_PHRASES)}"
        tags.append("call_phrase")
    if rng.random() < 0.15:
        text = f"{text} {rng.choice(['ph', 'mob', 'mob:', 'contact'])} {_phone(rng)}"
        tags.append("phone_in_text")
    if rng.random() < 0.03 and re.search(r"\bgali\b", text, re.I):
        text = re.sub(r"\bgali\b", "गली", text, count=1, flags=re.I)
        tags.append("devanagari")

    # Surface noise. Abbreviations and typos skip the building and unit so truth still matches.
    applied: list[tuple[str, str]] = []
    if rng.random() < 0.25:
        applied = [
            (pat, rep) for pat, rep in ABBREV
            if not any(re.search(pat, p, re.I) for p in protected if p)
        ]
        for pat, rep in applied:
            text = re.sub(pat, rep, text, flags=re.I)
        tags.append("abbrev")
    if rng.random() < 0.10:
        candidate = _typo(text, rng)
        keep = [truth.get(k) or "" for k in TEXT_FIELDS]
        if all(p.lower() in candidate.lower() for p in keep if p.lower() in text.lower()):
            text = candidate
            tags.append("typo")
    if rng.random() < 0.30:
        text = text.lower()
        tags.append("lower")
    if rng.random() < 0.15:
        text = re.sub(r"\s*,\s*", " ", text)
        tags.append("no_commas")
    inp["raw_address"] = re.sub(r"\s+", " ", text).strip(" ,")
    _sync_truth(truth, tags, applied)

    # Order context.
    inp["payment"] = "COD" if rng.random() < 0.6 else "PREPAID"
    inp["phone_present"] = rng.random() < 0.93
    r = rng.random()
    if truth.get("postcode") and r < 0.30:
        inp["declared_pincode"] = truth["postcode"]
    elif truth.get("postcode") and r < 0.35:
        wrong = str((int(truth["postcode"]) + rng.randrange(1000, 90000)) % 900000 + 100000)
        inp["declared_pincode"] = wrong
        tags.append("pin_mismatch")
    if truth.get("city") and rng.random() < 0.3:
        inp["city_hint"] = truth["city"]

    out["augment"] = tags
    return out
