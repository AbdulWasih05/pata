"""Deterministic dispatch rules. Pure functions, no I/O.

Two jobs: produce silver labels for training data, and act as the serve-time fallback
when the model output fails schema validation. Rules follow BRIEF.md section 3; every
interpretation beyond the brief is recorded in DECISIONS.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from pata.markets import AE_AMBIGUOUS, AE_AREAS, EMIRATES, IN_CITIES, IN_STATES, pin_to_state
from pata.schema import (
    DRIVER_NOTE_MAX,
    REASON_ORDER,
    Decision,
    Market,
    Normalised,
    Payment,
    ReasonCode,
    TriageInput,
    TriageOutput,
)

R = ReasonCode

# ---------------------------------------------------------------- patterns

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ARABIC = re.compile(r"[؀-ۿ]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[a-z]")

# Phone numbers are stripped before PIN / building detection so they are not misread.
# Per market: an Indian mobile pattern would otherwise eat UAE Makani numbers.
_PHONE = {
    Market.IN: re.compile(r"(?:\+?91[\s-]?)?(?<!\d)[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    Market.AE: re.compile(r"(?:\+?971[\s-]?|(?<!\d)0)5\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
}

_PIN = re.compile(r"(?<![\d])([1-9]\d{2})\s?(\d{3})(?![\d])")
_MAKANI = re.compile(r"(?:makani|مكاني)\s*(?:no\.?|number|#|:)?\s*[:\-]?\s*(\d{5})\s?(\d{5})")
_PO_BOX = re.compile(r"\b(?:p\.?\s?o\.?\s*box|post\s+box|pob|ص\.?\s?ب)\s*[:#.]?\s*(\d{2,7})")

_NUM = r"\s*(?:no\.?|number|num|#)?\s*[:\-.]?\s*[a-z]?-?\d+[a-z]?"

# Numbered building / house / plot identifiers.
_BUILDING_NUMBERED = re.compile(
    r"\b(?:villa|vila|house|h\.?\s?no|hno|door\s*no|d\.?\s?no|plot|bldg|building|block|blk"
    r"|tower|kothi|bungalow|quarter|qtr|khasra|survey\s*no|s\.?\s?no|warehouse|wh|camp|gate)"
    + _NUM
    + r"|#\s*\d+"
    + r"|\bno\.?\s*\d+"
    + r"|(?:فيلا|بناية|مبنى|عمارة|برج|بيت|منزل)\s*(?:رقم)?\s*\d+"
    + r"|\bfela\s*\d+|\b(?:3mara|3imara|emara|imara|binaya|bnaya)\s*\d+"
)
_LEADING_NUMBER = re.compile(r"^\s*\d+[a-z]?\s*(?:[,/\-]|\s+[a-z])")
_SLASH_NUMBER = re.compile(r"\b\d+[a-z]?\s*/\s*\d+[a-z]?\b")
# Flat / housing-board codes: "B-204", "HIG-35", "MIG 12" needs the hyphen to avoid false hits.
_FLAT_CODE = re.compile(r"\b[a-z]{1,3}\s?-\s?\d+[a-z]?\b")

_MULTI_UNIT_WORDS = (
    r"towers?|building|bldg|residency|residences?|apartments?|apts?|heights|complex|society|chs"
    r"|court|enclave|mansion|plaza|centre|center|mall|arcade|chambers|bhavan|bhawan|block"
    r"|accommodation|camp"  # staff accommodation / labour camp: findable by company name
)
# Named single-occupancy buildings ("Kanwal House", "Shanti Niwas"): identify the door alone.
_SINGLE_UNIT_WORDS = r"house|niwas|nivas|sadan|kutir|bungalow|cottage|kunj|mahal"
# A named building: up to four words followed by a building word ("Marina Pinnacle Tower").
_NAMED_BUILDING = re.compile(
    r"\b([a-z'&.]+(?:\s+[a-z'&.]+){0,3})\s+(?:"
    + _MULTI_UNIT_WORDS
    + "|"
    + _SINGLE_UNIT_WORDS
    + r")\b"
    r"|(?:بناية|عمارة|برج|مبنى)\s+[؀-ۿ]+(?:\s+[؀-ۿ]+)?"
)
_MULTI_UNIT = re.compile(r"\b(?:" + _MULTI_UNIT_WORDS + r")\b|برج|بناية|عمارة|مبنى")
# Words ending in a multi-unit word that are really single units or not buildings.
_NOT_A_BUILDING = re.compile(
    r"\b(?:shopping|community|health|medical|service|city)\s+cent(?:re|er)\b"
)

_UNIT = re.compile(
    r"\b(?:flat|flt|apt|apartment|unit|room|rm|office|off|suite|ste|shop|door|fl|floor|flr)"
    + _NUM
    + r"|\b\d+(?:st|nd|rd|th)\s*(?:floor|flr|fl)\b"
    + r"|\b(?:ground|first|second|third|top)\s+floor\b|\bg/?f\b"
    + r"|(?:شقة|مكتب|محل|غرفة)\s*(?:رقم)?\s*\d+"
    + r"|\b(?:sha2a|shaqa|shi2a|shaqqa)\s*\d+"
)

_LANDMARK = re.compile(
    r"\b(?:near(?:by)?|nr|opp|opposite|behind|beside|besides|next\s+to|in\s*front\s+of"
    r"|adjacent(?:\s+to)?|adj|close\s+to|across\s+from|facing|ganb|gamb|jamb|janb|wara|gidam"
    r"|ke\s+paas|ke\s+samne|ke\s+peeche)\b"
    r"|(?:قرب|بجانب|خلف|مقابل|جنب|ورا|عند)"
)

_CALL = re.compile(
    r"\bcall\s+(?:me|before|on\s+arrival|when|first|us|him|her)\b|\bgive\s+(?:a\s+)?call"
    r"|\b(?:will|i'?ll)\s+(?:share|send)\s+(?:the\s+|my\s+)?(?:location|loc|pin)"
    r"|\b(?:share|send)\s+(?:the\s+|my\s+)?location\b|\blocation\s+pin\b|\bwhats\s?app\b"
    r"|\bask\s+(?:the\s+)?(?:watchman|security|guard|chowkidar|haris|natoor|bawab)\b"
    r"|\bcall\s+kar|\bphone\s+kar|\bring\s+me\b"
    r"|اتصل|كلمني|ارسل\s*الموقع|واتساب"
)

_VILLAGE = re.compile(r"\b(?:vill(?:age)?|vpo|v\.p\.o|gram|gaon|tehsil|taluka|teh)\b")
_STREET = re.compile(
    r"\b(?:street|st|road|rd|lane|ln|marg|path|avenue|ave|highway|hwy|cross|main|salai|gali"
    r"|sector|sec|nagar|colony|layout|extension|phase)\b|شارع|سكة"
)

_ARABIZI_WORDS = {
    "shar3", "share3", "7ay", "7ai", "3nd", "3and", "ganb", "gamb", "janb", "jenb", "wara",
    "gidam", "bait", "beit", "bayt", "3mara", "3imara", "emara", "imara", "binaya", "bnaya",
    "ma7al", "ma7atta", "ba3d", "fo2", "ta7t", "sha2a", "shi2a", "fela", "masjed", "jame3",
    "mat3am", "dukkan", "baqala", "baqqala", "bjanb",
}
_UNIT_PREFIXES = ("flat", "apt", "villa", "plot", "unit", "shop", "office", "room", "bldg")

_STOPWORDS = {"the", "of", "and", "in", "at", "to", "a", "an", "is", "pls", "please", "plz"}


# ---------------------------------------------------------------- features


@dataclass
class Features:
    text: str  # normalised lowercase text, phones removed
    tokens: list[str]
    building: str = ""
    building_is_named: bool = False
    multi_unit: bool = False
    unit: str = ""
    landmark: str = ""
    street: str = ""
    text_pins: list[str] = field(default_factory=list)
    makani: str | None = None
    po_box: str | None = None
    call_requested: bool = False
    mixed_script: bool = False
    village: bool = False
    ae_area: str = ""
    ae_emirate: str = ""
    ae_ambiguous_area: str = ""
    in_state: str = ""
    in_city: str = ""


def _base_text(raw: str) -> str:
    t = unicodedata.normalize("NFKC", raw).translate(_ARABIC_DIGITS)
    return re.sub(r"\s+", " ", t).strip()


def clean_text(raw: str) -> str:
    return _base_text(raw).lower()


def _blank(m: re.Match) -> str:
    """Replace a match with spaces so character offsets stay aligned."""
    return " " * len(m.group(0))


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[\w']+", text)
    return [t for t in toks if t not in _STOPWORDS and (len(t) >= 2 or t.isdigit())]


def _is_arabizi(tok: str) -> bool:
    if tok in _ARABIZI_WORDS:
        return True
    if not re.search(r"[a-z]", tok) or not re.search(r"[2379]", tok) or len(tok) < 3:
        return False
    if re.fullmatch(r"\d+(?:st|nd|rd|th)", tok):  # ordinals
        return False
    if re.fullmatch(r"[a-z]{0,2}\d+[a-z]?|\d+[a-z]{0,2}", tok):  # codes like b12, 12a
        return False
    return not tok.startswith(_UNIT_PREFIXES)


def _segment(text: str, cased: str, m: re.Match) -> str:
    """The comma-delimited segment containing a match, in original casing, trimmed."""
    start = text.rfind(",", 0, m.start()) + 1
    end = text.find(",", m.end())
    return re.sub(r"\s+", " ", cased[start : end if end != -1 else len(text)]).strip(" ,.-")


_POSTPOSITION = re.compile(r"\bke\s+(?:paas|samne|peeche)\b")


def _landmark_spans(text: str) -> list[tuple[int, int]]:
    """Character spans that describe a landmark: from the cue word to the next comma.

    Postpositional cues ("mandir ke paas") cover the whole comma segment instead.
    """
    spans = []
    for m in _LANDMARK.finditer(text):
        end = text.find(",", m.end())
        start = text.rfind(",", 0, m.start()) + 1 if _POSTPOSITION.match(m.group(0)) else m.start()
        spans.append((start, end if end != -1 else len(text)))
    return spans


def _inside(m: re.Match, spans: list[tuple[int, int]]) -> bool:
    return any(s <= m.start() < e for s, e in spans)


def _longest_key(text: str, keys) -> str:
    best = ""
    for k in keys:
        if len(k) > len(best) and re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", text):
            best = k
    return best


def extract_features(inp: TriageInput) -> Features:
    cased = _base_text(inp.raw_address)
    text = cased.lower()
    if len(text) != len(cased):  # rare Unicode case folding that changes length
        cased = text
    makani = _MAKANI.search(text)  # before phones, so a Makani is never mistaken for one
    # Blank out phones (and below, Makani and PO Box) with spaces so offsets stay aligned.
    text = _PHONE[inp.market].sub(_blank, text)
    f = Features(text=text, tokens=_tokens(text))

    def orig(m: re.Match) -> str:
        return re.sub(r"\s+", " ", cased[m.start() : m.end()]).strip(" ,/-")

    if makani:
        f.makani = makani.group(1) + makani.group(2)
        text = text[: makani.start()] + _blank(makani) + text[makani.end() :]
    if m := _PO_BOX.search(text):
        f.po_box = m.group(1)
    # Strip Makani and PO Box digits before looking for PINs and building numbers.
    scrubbed = _PO_BOX.sub(_blank, _MAKANI.sub(_blank, text))

    if inp.market == Market.IN:
        f.text_pins = sorted({a + b for a, b in _PIN.findall(scrubbed)})

    # "near Al Noor Building" names a landmark, not the customer's building.
    spans = _landmark_spans(scrubbed)

    def find(pattern: re.Pattern) -> re.Match | None:
        return next((m for m in pattern.finditer(scrubbed) if not _inside(m, spans)), None)

    named = None if _NOT_A_BUILDING.search(scrubbed) else find(_NAMED_BUILDING)
    f.building_is_named = bool(named)
    if m := find(_BUILDING_NUMBERED):
        f.building = orig(m)
    elif named:
        f.building = orig(named)
    elif m := _LEADING_NUMBER.search(scrubbed):
        f.building = orig(m)
    elif m := find(_SLASH_NUMBER):
        f.building = orig(m)
    elif inp.market == Market.IN and (m := find(_FLAT_CODE)):
        f.building = orig(m)
    f.multi_unit = bool(find(_MULTI_UNIT)) and not _NOT_A_BUILDING.search(scrubbed)

    if m := _UNIT.search(scrubbed):
        f.unit = orig(m)
    elif inp.market == Market.IN and (m := _FLAT_CODE.search(scrubbed)):
        f.unit = orig(m)
    elif f.multi_unit and (m := _LEADING_NUMBER.search(scrubbed) or find(_SLASH_NUMBER)):
        f.unit = orig(m)  # "205, Panorama Complex": the leading number is the flat

    if spans:
        start, end = spans[0]
        f.landmark = re.sub(r"\s+", " ", cased[start:end]).strip(" ,.-")
    if m := _STREET.search(scrubbed):
        f.street = _segment(scrubbed, cased, m)

    f.call_requested = bool(_CALL.search(text))
    f.village = inp.market == Market.IN and bool(_VILLAGE.search(text))

    has_latin = bool(_LATIN.search(text))
    f.mixed_script = (
        (has_latin and bool(_ARABIC.search(text)))
        or (has_latin and bool(_DEVANAGARI.search(text)))
        or (inp.market == Market.AE and any(_is_arabizi(t) for t in f.tokens))
    )

    hint = clean_text(inp.city_hint or "")
    context = f"{text} {hint}".strip()
    if inp.market == Market.AE:
        f.ae_area = _longest_key(text, AE_AREAS)
        emirate_key = _longest_key(context, EMIRATES)
        f.ae_emirate = EMIRATES.get(emirate_key, "") or AE_AREAS.get(f.ae_area, "")
        for area, resolver in AE_AMBIGUOUS.items():
            if re.search(r"(?<![\w])" + re.escape(area) + r"(?![\w])", text) and not re.search(
                resolver, context
            ):
                f.ae_ambiguous_area = area
                break
    else:
        f.in_state = _longest_key(context, IN_STATES)
        f.in_city = _longest_key(context, IN_CITIES)
    return f


# ---------------------------------------------------------------- decision


def _sorted(codes: set[ReasonCode]) -> list[ReasonCode]:
    return [c for c in REASON_ORDER if c in codes]


def decide(inp: TriageInput, f: Features) -> tuple[Decision, list[ReasonCode]]:
    fail: set[ReasonCode] = set()
    call: set[ReasonCode] = set()
    # A numbered unit with a street ("Flat 1 A, 32 First Main Road") is findable too.
    has_building = bool(f.building or f.makani or (f.unit and f.street))

    # LIKELY_FAIL conditions
    if len(f.tokens) < 3:
        fail.add(R.TOO_SHORT)
    if inp.market == Market.IN and inp.declared_pincode and f.text_pins:
        declared = re.sub(r"\D", "", inp.declared_pincode)
        if declared not in f.text_pins:
            fail.add(R.PINCODE_MISMATCH)
    if f.po_box and not has_building:
        fail.add(R.PO_BOX_ONLY)
    if not has_building:
        if f.landmark:
            call.add(R.LANDMARK_ONLY)
        elif f.village:
            call.add(R.NO_BUILDING_ID)  # village-level: findable by asking locally
        else:
            fail.add(R.NO_BUILDING_ID)

    # CALL_FIRST conditions
    if f.multi_unit and has_building and not f.unit:
        call.add(R.MISSING_UNIT)
    if f.call_requested:
        call.add(R.CALL_REQUESTED)
    if f.mixed_script:
        call.add(R.MIXED_SCRIPT)
    if inp.market == Market.AE:
        if f.ae_ambiguous_area or not (f.ae_area or f.ae_emirate or f.makani):
            call.add(R.AMBIGUOUS_AREA)
    elif not (f.text_pins or inp.declared_pincode):
        call.add(R.AMBIGUOUS_AREA)

    weak = len(f.tokens) < 6 or not (f.street or f.building_is_named or f.makani or f.unit)
    if inp.payment == Payment.COD and (call or (weak and not fail)):
        call.add(R.COD_HIGH_RISK)

    if fail:
        decision = Decision.LIKELY_FAIL
    elif call:
        decision = Decision.CALL_FIRST
        if not inp.phone_present:
            decision = Decision.LIKELY_FAIL  # cannot call, so escalate one level
            fail.add(R.NO_PHONE)
    else:
        decision = Decision.DELIVERABLE

    codes = fail | call
    return decision, (_sorted(codes) if codes else [R.OK])


def confidence(decision: Decision, codes: list[ReasonCode], f: Features) -> float:
    if decision == Decision.DELIVERABLE:
        c = 0.85 + 0.05 * bool(f.text_pins or f.makani) + 0.05 * bool(f.unit or f.street)
    elif decision == Decision.LIKELY_FAIL:
        c = 0.7 + 0.08 * len(codes)
    else:
        c = 0.6 + 0.08 * len(codes)
    return round(min(c, 0.95), 2)


# ---------------------------------------------------------------- normalisation


def normalise(inp: TriageInput, f: Features) -> Normalised:
    n = Normalised(country=inp.market)
    n.building = f.building
    n.unit = f.unit if f.unit != f.building else ""
    n.landmark = f.landmark
    n.street = f.street if f.street not in (f.landmark, f.building) else ""
    hint = (inp.city_hint or "").strip()
    if inp.market == Market.AE:
        n.makani = f.makani
        n.po_box = f.po_box
        n.area = f.ae_area.title() if f.ae_area.isascii() else f.ae_area
        n.admin1 = f.ae_emirate
        n.city = hint or ("Al Ain" if "al ain" in f.text else f.ae_emirate)
    else:
        declared = re.sub(r"\D", "", inp.declared_pincode or "")
        n.postcode = f.text_pins[0] if f.text_pins else (declared or None)
        n.admin1 = f.in_state.title() if f.in_state else pin_to_state(n.postcode or "")
        n.city = hint or f.in_city.title()
    return n


# ---------------------------------------------------------------- notes

_CODE_PHRASE = {
    R.NO_BUILDING_ID: "no house/building number",
    R.LANDMARK_ONLY: "landmark only",
    R.PINCODE_MISMATCH: "PIN does not match",
    R.MIXED_SCRIPT: "mixed script, check spelling",
    R.AMBIGUOUS_AREA: "area unclear",
    R.MISSING_UNIT: "no flat/unit number",
    R.COD_HIGH_RISK: "COD on weak address",
    R.CALL_REQUESTED: "customer asked for a call",
    R.NO_PHONE: "no phone to call",
    R.PO_BOX_ONLY: "PO Box only, no door address",
    R.TOO_SHORT: "address too short",
}


def driver_note(decision: Decision, codes: list[ReasonCode], n: Normalised) -> str:
    where = ", ".join(x for x in (n.unit, n.building, n.street, n.area or n.city) if x)
    if decision == Decision.DELIVERABLE:
        note = f"Go to {where}." if where else "Address looks complete."
        if n.landmark:
            note = note[:-1] + f" ({n.landmark})."
    else:
        why = "; ".join(_CODE_PHRASE[c] for c in codes if c in _CODE_PHRASE)
        prefix = "Call customer before dispatch" if decision == Decision.CALL_FIRST else "Hold"
        note = f"{prefix}: {why}."
    return note if len(note) <= DRIVER_NOTE_MAX else note[: DRIVER_NOTE_MAX - 3].rstrip() + "..."


def call_script(codes: list[ReasonCode]) -> str:
    asks = []
    if R.MISSING_UNIT in codes:
        asks.append("your flat or unit number")
    if R.LANDMARK_ONLY in codes or R.NO_BUILDING_ID in codes:
        asks.append("your building or house number")
    if R.AMBIGUOUS_AREA in codes:
        asks.append("which area and city exactly")
    if R.MIXED_SCRIPT in codes:
        asks.append("the building name spelling")
    if R.CALL_REQUESTED in codes or not asks:
        asks.append("a location pin on WhatsApp")
    s = "Hi, this is your courier. Could you confirm " + ", ".join(asks) + "?"
    if R.COD_HIGH_RISK in codes:
        s += " Payment is cash on delivery."
    return s


# ---------------------------------------------------------------- entry point


def triage(inp: TriageInput) -> TriageOutput:
    f = extract_features(inp)
    decision, codes = decide(inp, f)
    n = normalise(inp, f)
    return TriageOutput(
        decision=decision,
        confidence=confidence(decision, codes, f),
        reason_codes=codes,
        normalised=n,
        driver_note=driver_note(decision, codes, n),
        call_script=call_script(codes) if decision == Decision.CALL_FIRST else None,
    )
