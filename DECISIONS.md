# Decisions

Decisions made while building, so nobody has to ask twice. Newest at the bottom.

**D1. Brief file name.** CLAUDE.md refers to `BUILD_BRIEF.md`; the file in the repo is `BRIEF.md`. Same document. Kept the existing name.

**D2. Phone-absent escalation.** BRIEF section 3 says "a phone being absent escalates one level (DELIVERABLE to CALL_FIRST becomes LIKELY_FAIL since you cannot call)". Read as: anything that would be CALL_FIRST becomes LIKELY_FAIL when there is no phone, because the call cannot happen. A DELIVERABLE address stays DELIVERABLE without a phone; escalating it to CALL_FIRST would ask the driver to call a number that does not exist.

**D3. Extra reason codes.** The brief's list has no code for four of its own rules, so the model would have to explain them with a wrong code. Added: `CALL_REQUESTED` ("call me", "will share location"), `NO_PHONE` (the escalation in D2), `PO_BOX_ONLY` (PO Box with no door address), `TOO_SHORT` (fewer than 3 meaningful tokens). `reason_codes` are always emitted in the enum's fixed order so training targets are deterministic.

**D4. India addresses with no PIN at all** (neither in text nor declared) are `CALL_FIRST` with `AMBIGUOUS_AREA`. A courier cannot route an Indian parcel without a PIN. This mirrors the `QUARANTINED` lifecycle state in the raw dataset, which means "no pincode found".

**D5. Indian village-level addresses** ("Village X, Post Y, Tehsil Z") with no house number are `CALL_FIRST` with `NO_BUILDING_ID`, not `LIKELY_FAIL`. Rural deliveries routinely work this way by asking locally once in the village. Without this rule a large share of the MCA registry data would be labelled LIKELY_FAIL, which is wrong in practice.

**D6. "COD with a weak address"** is defined as: COD, and either another CALL_FIRST reason is present, or the address has fewer than 6 meaningful tokens, or it has a building id but no street, named building, unit, or Makani. Adds `COD_HIGH_RISK`.

**D7. Landmark-qualified buildings.** A building mentioned after a landmark cue ("near Al Noor Building") is treated as the landmark, not the customer's building. Postpositional Hindi cues ("mandir ke paas") cover the whole comma segment.

**D8. Makani counts as a building identifier** for AE: it geolocates the entrance. A UAE mobile number (05X...) is never read as a Makani; Makani requires the keyword ("Makani", "مكاني").

**D9. Ambiguous UAE areas** (Al Nahda, Khalifa City, Al Rashidiya, Al Quoz, Al Qusais, Industrial Area) need a qualifier in the text or `city_hint` (emirate, sector letter, or sub-area number), otherwise `AMBIGUOUS_AREA`. The list lives in `pata/markets.py`.

**D10. Normalised fields keep the customer's original casing** (taken from the input text by character offset), so the model learns to copy rather than rewrite.
