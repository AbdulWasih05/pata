"""Synthetic UAE address generator. Pure stdlib, deterministic for a given seed.

No real customer addresses. Patterns come from Gulf last-mile experience (BRIEF.md section 4).
Each row carries the generator's intended decision and the ground-truth normalised fields
as they appear in the raw text, so rules and models can be checked against intent.

    uv run python -m data.ae_generator --n 3500 --seed 7 --out data/ae/ae_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pata.markets import AE_AMBIGUOUS, AE_AREAS
from pata.schema import Decision, Normalised, TriageInput

D, C, F = Decision.DELIVERABLE, Decision.CALL_FIRST, Decision.LIKELY_FAIL

# ---------------------------------------------------------------- reference pools

_EXCLUDED_AREAS = set(AE_AMBIGUOUS) | {
    "al rashidiya ajman", "ajman industrial", "hamdan street", "electra street",
}
_BUILDING_WORDS = re.compile(
    r"\b(?:towers?|residences?|heights|building|complex|court|plaza|centre|center|mall|block"
    r"|apartments?)\b"
)
# Residential communities: unambiguous, ASCII, and with no building word in the name
# (otherwise "Jumeirah Lake Towers" reads as the customer's building).
AREAS = sorted(
    k for k in AE_AREAS
    if k.isascii() and k not in _EXCLUDED_AREAS and not _BUILDING_WORDS.search(k)
)
_DISPLAY = {
    "jlt": "JLT", "jbr": "JBR", "jvc": "JVC", "jvt": "JVT", "dip": "DIP", "tecom": "TECOM",
    "mbz city": "MBZ City",
}
# (display text, gazetteer key) for industrial zones with a resolvable sub-area.
INDUSTRIAL = [
    ("Al Quoz Industrial {n}", "al quoz", 4),
    ("Al Qusais Industrial {n}", "al qusais", 5),
    ("Industrial Area {n}", "industrial area", 18),
    ("Ajman Industrial {n}", "ajman industrial", 2),
    ("Musaffah M-{n}", "musaffah", 40),
    ("Jebel Ali Industrial {n}", "jebel ali", 3),
]
CAMP_AREAS = ["sonapur", "al muhaisnah", "jebel ali", "musaffah", "dubai investments park"]
AR_AREAS = ["البرشاء", "ديرة", "الكرامة", "القصيص", "المجاز", "التعاون", "مصفح"]
AR_EMIRATE = {"Dubai": "دبي", "Sharjah": "الشارقة", "Abu Dhabi": "أبوظبي"}

EMIRATE_ABBR = {
    "Dubai": ["Dxb", "DXB", "dxb"],
    "Sharjah": ["Shj", "SHJ"],
    "Abu Dhabi": ["AUH", "Auh"],
}

# Name pools. Test-pool names never appear in train-pool rows (BRIEF.md section 4, splits).
POOLS = {
    "train": {
        "building": [
            "Al Fardan", "Marina Pinnacle", "Sapphire", "Golden Sands", "Silicon Gates",
            "Lake Shore", "Al Waleed", "Crystal", "Emerald", "Al Manara", "Palm View",
            "Blue Wave", "Al Safeer", "Pearl", "Al Mansoor", "Ocean View", "Sky Gardens",
            "Al Zahra", "Desert Bloom", "Al Hilal", "Green Lake", "Al Attar", "Gulf Pearl",
        ],
        "ar_building": ["الفردوس", "النور", "الوليد", "الصفا", "المنارة"],
        "road": [
            "Al Wasl Road", "Hessa Street", "Sheikh Zayed Road", "Al Khail Road",
            "Al Ittihad Road", "Damascus Street", "Beirut Street", "Al Maktoum Road",
        ],
        "street_nums": range(1, 61),
        "landmark": [
            "Lulu Hypermarket", "Carrefour", "ADCB", "Emirates NBD", "Al Noor Mosque",
            "Al Rashid Mosque", "the big mosque", "ADNOC petrol station", "ENOC pump",
            "EPPCO station", "Union metro station", "Indian High School",
            "Al Ittihad Private School", "Nesto Hypermarket",
        ],
        "ar_landmark": ["مسجد النور", "لولو هايبرماركت", "محطة المترو"],
        "arabizi_landmark": ["carrefour", "el masjed", "lulu", "el metro"],
        "company": [
            "Gulf Star Contracting", "Al Bawadi Cleaning", "Blue Line Security",
            "Desert Falcon Construction",
        ],
    },
    "test": {
        "building": [
            "Al Shams", "Coral Bay", "Moonlight", "Al Yasmeen", "Cedar", "Falcon Crest",
            "Al Bustan", "Horizon", "Opal", "Al Reef",
        ],
        "ar_building": ["الياسمين", "البستان", "الشمس"],
        "road": ["Al Mankhool Road", "Cairo Street", "Tripoli Street"],
        "street_nums": range(61, 100),
        "landmark": [
            "Union Coop", "Spinneys", "Mashreq ATM", "Al Farooq Mosque",
            "Emarat petrol station", "Al Qiyadah metro station", "Delhi Private School",
            "Choithrams",
        ],
        "ar_landmark": ["مسجد الفاروق", "الجمعية التعاونية"],
        "arabizi_landmark": ["spinneys", "el ba2ala", "union coop"],
        "company": ["Sand Dune Facilities", "Oasis Builders"],
    },
}
BUILDING_SUFFIX = ["Tower", "Building", "Residence", "Heights", "Tower", "Building"]
LANDMARK_CUES = ["behind", "near", "opposite", "next to", "beside", "in front of"]
AR_CUES = ["قرب", "خلف", "مقابل", "بجانب"]

CALL_PHRASES = [
    "call before coming", "call me when you reach", "pls call on arrival",
    "call first, gate is locked", "call me before delivery",
]
WHATSAPP_PHRASES = [
    "will share location on WhatsApp", "I'll send location on whatsapp",
    "location pin on WhatsApp", "send location pls", "whatsapp me for location",
]
WATCHMAN_PHRASES = [
    "ask watchman", "ask the security", "ask security at gate", "ask the haris",
    "ask the watchman for the key",
]

# Pattern -> (intended decision before phone escalation, share of rows).
PATTERNS: dict[str, tuple[Decision, float]] = {
    "villa_street": (D, 0.11),
    "flat_building": (D, 0.11),
    "makani": (D, 0.04),
    "po_box_door": (D, 0.03),
    "industrial_plot": (D, 0.04),
    "labour_camp": (D, 0.03),
    "arabic_full": (D, 0.04),
    "landmark_only": (C, 0.07),
    "arabic_landmark": (C, 0.02),
    "mixed_script": (C, 0.04),
    "arabizi": (C, 0.04),
    "call_before": (C, 0.04),
    "whatsapp_location": (C, 0.03),
    "ask_watchman": (C, 0.02),
    "ambiguous_area": (C, 0.05),
    "missing_unit": (C, 0.04),
    "staff_accommodation": (D, 0.02),  # DECISIONS.md D12
    "po_box_only": (F, 0.06),
    "no_identifier": (F, 0.11),
    "too_short": (F, 0.06),
}

# ---------------------------------------------------------------- row drafting


@dataclass
class Draft:
    """An address as a list of (truth field, text) parts, joined at the end."""

    pattern: str
    intended: Decision
    parts: list[tuple[str | None, str]]
    emirate: str  # canonical emirate for admin1/city
    names: set[str] = field(default_factory=set)
    makani: str | None = None
    po_box: str | None = None
    arabic: bool = False  # no Latin-style noise
    hint_allowed: bool = True
    sep: str = ", "


class Gen:
    def __init__(self, rng: random.Random, pool: str):
        self.r = rng
        self.pool = pool
        self.p = POOLS[pool]

    # -- pickers

    def area(self) -> tuple[str, str]:
        key = self.r.choice(AREAS)
        return _DISPLAY.get(key, key.title()), AE_AREAS[key]

    def industrial(self) -> tuple[str, str]:
        text, key, hi = self.r.choice(INDUSTRIAL)
        return text.format(n=self.r.randint(1, hi)), AE_AREAS[key]

    def building_name(self, names: set[str]) -> str:
        base = self.r.choice(self.p["building"])
        names.add(f"building:{base}")
        return f"{base} {self.r.choice(BUILDING_SUFFIX)}"

    def street(self, names: set[str]) -> str:
        if self.r.random() < 0.25:
            road = self.r.choice(self.p["road"])
            names.add(f"street:{road}")
            return road
        n = self.r.choice(self.p["street_nums"])
        letter = self.r.choice(["", "", "", "a", "b"])
        names.add(f"street:{n}{letter}")
        if self.r.random() < 0.2 and not letter:
            return f"{n}{_ordinal(n)} Street"
        return f"Street {n}{letter}"

    def landmark(self, names: set[str]) -> str:
        name = self.r.choice(self.p["landmark"])
        names.add(f"landmark:{name}")
        return f"{self.r.choice(LANDMARK_CUES)} {name}"

    def emirate_part(self, emirate: str) -> tuple[str, str]:
        return ("emirate", emirate)

    # -- patterns (each returns a Draft)

    def villa_street(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        parts = [
            ("building", f"Villa {self.r.randint(1, 120)}"),
            ("street", self.street(names)),
            ("area", area),
        ]
        return Draft("villa_street", D, parts, em, names)

    def flat_building(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        unit = self.r.choice(["Flat", "Flat", "Apt", "Apartment", "Office"])
        parts = [
            ("unit", f"{unit} {self.r.randint(1, 45)}{self.r.randint(1, 20):02d}"),
            ("building", self.building_name(names)),
        ]
        if self.r.random() < 0.3:
            parts.append(("street", self.street(names)))
        parts.append(("area", area))
        return Draft("flat_building", D, parts, em, names)

    def makani(self) -> Draft:
        names: set[str] = set()
        digits = str(self.r.randint(1, 9)) + "".join(str(self.r.randint(0, 9)) for _ in range(9))
        fmt = self.r.choice(["Makani {a} {b}", "Makani: {a}{b}", "Makani No. {a} {b}"])
        mk = fmt.format(a=digits[:5], b=digits[5:])
        if self.r.random() < 0.5:
            area, em = self.area()
            parts = [
                ("building", f"Villa {self.r.randint(1, 120)}"),
                ("street", self.street(names)),
                ("area", area),
                (None, mk),
            ]
        else:
            area, em = self.industrial()
            parts = [(None, mk), ("area", area)]
        d = Draft("makani", D, parts, em, names, makani=digits)
        return d

    def po_box_door(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        box = str(self.r.randint(100, 99999))
        parts = [
            ("unit", f"Office {self.r.randint(101, 2405)}"),
            ("building", self.building_name(names)),
            ("area", area),
            ("emirate", em),
            (None, f"{self.r.choice(['PO Box', 'P.O. Box', 'P.O Box'])} {box}"),
        ]
        return Draft("po_box_door", D, parts, em, names, po_box=box)

    def industrial_plot(self) -> Draft:
        names: set[str] = set()
        area, em = self.industrial()
        plot = f"Plot {self.r.randint(100, 999)}-{self.r.randint(100, 999)}"
        parts = [
            ("unit", f"{self.r.choice(['Warehouse', 'Shop', 'Unit'])} {self.r.randint(1, 60)}"),
            ("building", plot),
            ("street", self.street(names)),
            ("area", area),
        ]
        return Draft("industrial_plot", D, parts, em, names)

    def labour_camp(self) -> Draft:
        key = self.r.choice(CAMP_AREAS)
        parts = [
            ("unit", f"Room {self.r.randint(1, 240)}"),
            ("building", f"Camp {self.r.randint(1, 30)}"),
            ("area", key.title()),
            ("emirate", AE_AREAS[key]),
        ]
        return Draft("labour_camp", D, parts, AE_AREAS[key])

    def arabic_full(self) -> Draft:
        names: set[str] = set()
        area = self.r.choice(AR_AREAS)
        em = AE_AREAS[area]
        n = self.r.choice(self.p["street_nums"])
        names.add(f"street:{n}")
        if self.r.random() < 0.5:
            b = self.r.choice(self.p["ar_building"])
            names.add(f"building:{b}")
            parts = [("unit", f"شقة {self.r.randint(1, 40)}"), ("building", f"بناية {b}")]
        else:
            parts = [("building", f"فيلا {self.r.randint(1, 120)}")]
        parts += [("street", f"شارع {n}"), ("area", area), (None, AR_EMIRATE[em])]
        return Draft("arabic_full", D, parts, em, names, arabic=True, sep="، ")

    def landmark_only(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        parts = [("landmark", self.landmark(names)), ("area", area)]
        return Draft("landmark_only", C, parts, em, names)

    def arabic_landmark(self) -> Draft:
        names: set[str] = set()
        area = self.r.choice(AR_AREAS)
        em = AE_AREAS[area]
        lm = self.r.choice(self.p["ar_landmark"])
        names.add(f"landmark:{lm}")
        parts = [
            ("landmark", f"{self.r.choice(AR_CUES)} {lm}"),
            ("area", area),
            (None, AR_EMIRATE[em]),
        ]
        return Draft("arabic_landmark", C, parts, em, names, arabic=True, sep="، ")

    def mixed_script(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        b = self.r.choice(self.p["ar_building"])
        names.add(f"building:{b}")
        v = self.r.randint(0, 2)
        if v == 0:
            parts = [
                ("unit", f"شقة {self.r.randint(1, 40)}"),
                ("building", f"بناية {b}"),
                ("landmark", "near metro"),
                ("area", area),
                ("emirate", em),
            ]
            return Draft("mixed_script", C, parts, em, names, arabic=True, sep=" ")
        if v == 1:
            parts = [
                ("unit", f"Flat {self.r.randint(101, 1505)}"),
                ("building", f"بناية {b}"),
                ("area", area),
                ("emirate", em),
            ]
        else:
            n = self.r.choice(self.p["street_nums"])
            names.add(f"street:{n}")
            parts = [
                ("building", f"Villa {self.r.randint(1, 120)}"),
                ("street", f"شارع {n}"),
                ("area", area),
            ]
        return Draft("mixed_script", C, parts, em, names, arabic=True)

    def arabizi(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        n = self.r.choice(self.p["street_nums"])
        names.add(f"street:{n}")
        lm = self.r.choice(self.p["arabizi_landmark"])
        names.add(f"landmark:{lm}")
        cue = self.r.choice(["wara", "ganb", "ba3d"])
        if self.r.random() < 0.5:
            parts = [
                ("building", f"fela {self.r.randint(1, 120)}"),
                ("street", f"shar3 {n}"),
                ("landmark", f"{cue} {lm}"),
                ("area", area),
            ]
        else:
            parts = [
                ("building", f"3mara {self.r.randint(1, 60)}"),
                ("unit", f"sha2a {self.r.randint(1, 40)}"),
                ("landmark", f"{cue} {lm}"),
                ("area", area.lower()),
            ]
        return Draft("arabizi", C, parts, em, names)

    def _with_phrase(self, pattern: str, phrases: list[str]) -> Draft:
        base = self.villa_street() if self.r.random() < 0.5 else self.flat_building()
        base.parts.append((None, self.r.choice(phrases)))
        base.pattern, base.intended = pattern, C
        return base

    def call_before(self) -> Draft:
        return self._with_phrase("call_before", CALL_PHRASES)

    def whatsapp_location(self) -> Draft:
        return self._with_phrase("whatsapp_location", WHATSAPP_PHRASES)

    def ask_watchman(self) -> Draft:
        return self._with_phrase("ask_watchman", WATCHMAN_PHRASES)

    def ambiguous_area(self) -> Draft:
        names: set[str] = set()
        v = self.r.randint(0, 4)
        if v == 0:  # Al Nahda with no emirate anywhere
            parts = [
                ("building", f"Villa {self.r.randint(1, 80)}"),
                ("street", self.street(names)),
                ("area", "Al Nahda"),
            ]
            d = Draft("ambiguous_area", C, parts, "", names)
            d.hint_allowed = False
            return d
        if v == 1:
            parts = [
                ("building", f"Villa {self.r.randint(1, 80)}"),
                ("street", self.street(names)),
                ("area", "Khalifa City"),
                ("emirate", "Abu Dhabi"),
            ]
            return Draft("ambiguous_area", C, parts, "Abu Dhabi", names)
        if v == 2:
            parts = [
                ("unit", f"Warehouse {self.r.randint(1, 60)}"),
                ("building", f"Plot {self.r.randint(100, 999)}"),
                ("street", self.street(names)),
                ("area", "Al Quoz"),
                ("emirate", "Dubai"),
            ]
            return Draft("ambiguous_area", C, parts, "Dubai", names)
        if v == 3:
            parts = [
                ("unit", f"Flat {self.r.randint(101, 905)}"),
                ("building", self.building_name(names)),
                ("area", "Al Qusais"),
                ("emirate", "Dubai"),
            ]
            return Draft("ambiguous_area", C, parts, "Dubai", names)
        parts = [
            ("building", f"Plot {self.r.randint(1, 400)}"),
            ("street", self.street(names)),
            ("area", "Industrial Area"),
            ("emirate", "Sharjah"),
        ]
        return Draft("ambiguous_area", C, parts, "Sharjah", names)

    def missing_unit(self) -> Draft:
        names: set[str] = set()
        area, em = self.area()
        parts = [("building", self.building_name(names)), ("area", area)]
        return Draft("missing_unit", C, parts, em, names)

    def staff_accommodation(self) -> Draft:
        names: set[str] = set()
        company = self.r.choice(self.p["company"])
        names.add(f"company:{company}")
        area, em = self.industrial()
        parts = [
            ("unit", f"Room {self.r.randint(1, 240)}"),
            ("building", f"{company} Staff Accommodation"),
            ("area", area),
            ("emirate", em),
        ]
        return Draft("staff_accommodation", D, parts, em, names)

    def po_box_only(self) -> Draft:
        area, em = self.area()
        box = str(self.r.randint(100, 99999))
        label = self.r.choice(["PO Box", "P.O. Box", "POBox", "P.O Box"])
        parts = [(None, f"{label} {box}")]
        if self.r.random() < 0.7:
            parts.append(("area", area))
        parts.append(("emirate", em))
        return Draft("po_box_only", F, parts, em, po_box=box)

    def no_identifier(self) -> Draft:
        area, em = self.area()
        parts = [("area", area), ("emirate", em)]
        filler = self.r.choice(["", "", "deliver fast", "my home", "UAE", "pls deliver asap"])
        if filler:
            parts.append((None, filler))
        d = Draft("no_identifier", F, parts, em)
        return d

    def too_short(self) -> Draft:
        v = self.r.randint(0, 2)
        if v == 0:
            em = self.r.choice(["Dubai", "Sharjah", "Ajman", "Abu Dhabi"])
            return Draft("too_short", F, [("emirate", em)], em)
        if v == 1:
            key = self.r.choice(["jlt", "jbr", "jvc", "deira", "karama", "mirdif"])
            return Draft("too_short", F, [("area", _DISPLAY.get(key, key.title()))], AE_AREAS[key])
        em = self.r.choice(["Dubai", "Sharjah"])
        return Draft("too_short", F, [("emirate", em), (None, "UAE")], em)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ---------------------------------------------------------------- noise

_ABBR = {
    "street": [(r"\bStreet\b", "St"), (r"\bRoad\b", "Rd")],
    "building": [(r"\bBuilding\b", "Bldg")],
    "unit": [(r"\bFlat\b", "Flt"), (r"\bApartment\b", "Apt")],
    "landmark": [(r"^near\b", "nr"), (r"^opposite\b", "opp")],
}
_PROTECTED = {
    "tower", "building", "residence", "heights", "behind", "near", "opposite", "beside",
    "front", "mosque", "station", "school", "metro", "petrol", "next",
    # Structural keywords: typos go on names, so the intended label stays unambiguous.
    "villa", "flat", "room", "camp", "plot", "office", "warehouse", "street", "staff",
    "accommodation",
}


def _typo(text: str, r: random.Random) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()
           and w.lower() not in _PROTECTED]
    if not idx:
        return text
    i = r.choice(idx)
    w = words[i]
    j = r.randint(1, len(w) - 2)
    words[i] = w[:j] + w[j] + w[j:] if r.random() < 0.5 else w[:j] + w[j + 1 :]
    return " ".join(words)


def _phone(r: random.Random, arabic: bool) -> str:
    num = f"5{r.choice('024568')}{r.randint(1000000, 9999999)}"
    forms = [f"0{num}", f"+971 {num[:2]} {num[2:5]} {num[5:]}"]
    return r.choice(forms if arabic else [*forms, f"mob 0{num}"])


def _apply_noise(d: Draft, r: random.Random, phone_present: bool) -> None:
    parts = list(d.parts)
    if not d.arabic:
        if r.random() < 0.35:
            for i, (fld, text) in enumerate(parts):
                for pat, rep in _ABBR.get(fld or "", []):
                    # Flt is a real customer spelling; keep it rare so labels stay mostly clean.
                    p = 0.25 if rep == "Flt" else 0.6
                    if r.random() < p:
                        text = re.sub(pat, rep, text)
                if fld == "emirate" and text in EMIRATE_ABBR and r.random() < 0.5:
                    text = r.choice(EMIRATE_ABBR[text])
                parts[i] = (fld, text)
        if r.random() < 0.12:
            cand = [i for i, (fld, _) in enumerate(parts) if fld in ("building", "landmark")]
            if cand:
                i = r.choice(cand)
                parts[i] = (parts[i][0], _typo(parts[i][1], r))
        if r.random() < 0.30:
            parts = [(fld, text.lower()) for fld, text in parts]
        if r.random() < 0.25:
            d.sep = " "
        elif r.random() < 0.15:
            d.sep = r.choice([" ,  ", ",  ", " , "])
    if phone_present and r.random() < 0.12:
        parts.append((None, _phone(r, d.arabic and d.sep == "، ")))
    d.parts = parts


# ---------------------------------------------------------------- assembly


def _tokens(text: str) -> int:
    toks = re.findall(r"[\w']+", text.lower())
    return sum(1 for t in toks if len(t) >= 2 or t.isdigit())


def _finalise(i: int, d: Draft, pool: str, r: random.Random) -> dict:
    payment = "COD" if r.random() < 0.55 else "PREPAID"
    phone_present = r.random() < 0.92

    has_emirate = any(fld == "emirate" or fld is None and t in AR_EMIRATE.values()
                      for fld, t in d.parts)
    if not has_emirate and d.emirate and d.pattern not in ("too_short",) and d.hint_allowed:
        if r.random() < 0.7:
            d.parts.append(("emirate", AR_EMIRATE[d.emirate] if d.arabic and d.sep == "، "
                            and d.emirate in AR_EMIRATE else d.emirate))
    # A deliverable COD order needs a full address (DECISIONS.md D6); pad short ones the way
    # customers do, with the emirate and country.
    if d.intended == D and payment == "COD":
        if _tokens(" ".join(t for _, t in d.parts)) < 6:
            if not any(fld == "emirate" for fld, _ in d.parts):
                d.parts.append(("emirate", d.emirate))
            d.parts.append((None, "UAE"))

    _apply_noise(d, r, phone_present)
    raw = d.sep.join(t for _, t in d.parts)

    hint = None
    if d.hint_allowed and d.emirate and r.random() < 0.4:
        hint = r.choice([d.emirate, d.emirate.upper(), d.emirate.lower()]
                        + EMIRATE_ABBR.get(d.emirate, [])[:1])

    intended = d.intended
    if intended == C and not phone_present:
        intended = F  # DECISIONS.md D2

    truth = {"country": "AE", "admin1": d.emirate, "city": d.emirate, "postcode": None,
             "makani": d.makani, "po_box": d.po_box}
    for fld in ("area", "street", "building", "unit", "landmark"):
        truth[fld] = next((t for f_, t in d.parts if f_ == fld), "")
    inp = {"market": "AE", "raw_address": raw, "phone_present": phone_present,
           "payment": payment, "declared_pincode": None, "city_hint": hint}
    TriageInput.model_validate(inp)
    Normalised.model_validate(truth)
    return {
        "id": f"ae-{i:06d}", "pattern": d.pattern, "intended_decision": intended.value,
        "pool": pool, "input": inp, "truth": truth, "_names": sorted(d.names),
    }


def _quota(n: int) -> list[str]:
    counts = {p: max(1, round(n * w)) for p, (_, w) in PATTERNS.items()}
    biggest = max(counts, key=counts.get)
    counts[biggest] += n - sum(counts.values())
    return [p for p, c in counts.items() for _ in range(c)]


def generate(n: int, seed: int = 7, test_share: float = 0.10) -> list[dict]:
    """Rows with an extra `_names` key (held-out names used), stripped before writing."""
    r = random.Random(seed)
    plan = _quota(n)
    r.shuffle(plan)
    rows = []
    for i, pattern in enumerate(plan):
        pool = "test" if r.random() < test_share else "train"
        d = getattr(Gen(r, pool), pattern)()
        rows.append(_finalise(i, d, pool, r))
    return rows


# ---------------------------------------------------------------- report


def report(rows: list[dict]) -> None:
    from pata.rules import triage

    print("by pattern:", dict(Counter(x["pattern"] for x in rows).most_common()))
    print("by intended:", dict(Counter(x["intended_decision"] for x in rows)))
    print("by pool:", dict(Counter(x["pool"] for x in rows)))

    labels = [d.value for d in Decision]
    conf: Counter = Counter()
    bad = []
    for x in rows:
        got = triage(TriageInput.model_validate(x["input"]))
        conf[(x["intended_decision"], got.decision.value)] += 1
        if got.decision.value != x["intended_decision"]:
            bad.append((x, got))
    agree = sum(v for (a, b), v in conf.items() if a == b) / len(rows)
    print(f"rules vs intent agreement: {agree:.1%}")
    print("intended \\ rules".ljust(18) + "".join(lbl[:12].rjust(13) for lbl in labels))
    for a in labels:
        print(a.ljust(18) + "".join(str(conf[(a, b)]).rjust(13) for b in labels))
    print("disagreements by pattern:", dict(Counter(x["pattern"] for x, _ in bad).most_common()))
    for x, got in bad[:15]:
        print(f"  [{x['pattern']}] {x['intended_decision']} vs {got.decision.value} "
              f"{[c.value for c in got.reason_codes]} | {x['input']['raw_address']!r} "
              f"pay={x['input']['payment']} phone={x['input']['phone_present']} "
              f"hint={x['input']['city_hint']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/ae/ae_raw.jsonl")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps({k: v for k, v in x.items() if k != "_names"},
                                ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    if not args.no_report:
        report(rows)


if __name__ == "__main__":
    main()
