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

**D11. India sources.** Training uses only MCA (company-registry) rows from `gagan1985/indian-addresses-gold` and `-raw`, never the `bank` rows: those are KYC customer addresses, and this repo holds no real customer addresses. Gold MCA rows (2,508, span-labelled) give real normalised fields; a sample of 380 MCA `QUARANTINED` rows (no PIN) covers the no-PIN case, with fields from the rules; `abr-6024/indian-address-parsing-synthetic` adds customer-typed India addresses with truth fields. Registry artefacts (`Unclassified`, placeholder PIN `000000`) are stripped. India test rows are whole PIN regions held out from train (about 10% of gold rows), plus the abr-6024 test file.

**D12. UAE staff accommodation** ("Room 88, Blue Line Security Staff Accommodation, Al Qusais Industrial 3") is `DELIVERABLE`: room number, named compound and numbered area are findable. "accommodation" and "camp" are building words in the rules.

**D13. Label consistency filter.** Decisions come from the rules, normalised fields from source truth. Rows where they disagree on whether a building is identified are dropped (318 India rows), as are UAE rows where the rules disagree with the generator's intended decision (now 0). UAE agreement is 100% partly by construction, because the generator was checked against the rules. That is why the reported test numbers must come from the human hand-checked sets in `data/gold/`, not the silver test split.

**D14. Augmentation keeps truth in step with text.** Each India record gets one augmented variant (`data/augment.py`): landmark, call phrase, phone, dropped PIN or building, abbreviations, typos, lowercase, plus COD (60%), no phone (7%), declared PIN (30% correct, 5% wrong). Truth fields get the same edits so the model learns to copy, and typos never land inside a truth field. Test-pool rows draw landmarks from a separate pool.

**D15. A house number in front of a named building is the unit.** "h.no 323, Balaji Heights" and "#743, Gokul Bhavan" give the flat; the name gives the building. Also "2nd. floor" (with a full stop) counts as a floor. Found because the epoch-1 checkpoint of run 1 called these DELIVERABLE while the rules said MISSING_UNIT: on review the model was right. The fix changed the decision on 165 India training rows and 37 test rows (git diff of data/splits between 798be16 and 1a9237d). Run 2 trains on the corrected labels.

**D16. The gold test sets are a blind LLM review, not human hand-checks.** The humans did not have time to label 300 rows before the deadline. Each market's 150 rows were labelled by a separate Claude agent that saw only the address and order context (no rules output, no silver label, no code) and the decision definitions in the brief and this file. The `reviewer` column says `llm-blind`. It is still independent of `pata/rules.py`, which is the point of a gold set here: the reviewer agrees with the rules on 130/150 India and 127/150 UAE rows. The post and README say plainly that these labels come from an LLM.

**D17. No retraining on what the gold review found.** The review exposed rules that are stricter than a dispatcher would be: `MIXED_SCRIPT` sends every Arabic-script address to CALL_FIRST even when building, unit and area are clear, and D9 asks for a sub-area in Al Quoz or Khalifa City even when a plot and street are given. Changing the rules and retraining now would tune to the test set, so the findings are reported as limitations instead.

**D18. The tuned model is served by Tinker's OpenAI-compatible endpoint, with hedged requests.** No GPU Droplet was available, so the DigitalOcean category is dropped and Render calls Tinker directly with the run 2 sampler checkpoint (`reasoning_effort: false` turns thinking off there). Sentry `chat` spans showed two latency bands: about 1.1 s and about 6.1 s, the second being a server-side hold that streaming, fresh connections and concurrency did not avoid. The client now sends a second identical request when the first has not answered in 1.5 s and keeps the first answer (`PATA_HEDGE_MS`, on by default only for Tinker). A first manual run of 40 sequential requests through the gateway gave p90 6.16 s to 3.15 s. The reported numbers come from the scripted benchmark (`eval/latency_bench.py`, written to `eval/latency.md`): p90 6.19 s to 2.92 s, mean 2.37 s to 1.30 s, requests over 5 s 10 to 0 of 40. The cost is a second call on slow requests only: $1.62 per 10,000 parcels, at most $1.86 with hedging.

**D19. Partner categories entered: Tinker, Render, Sentry.** There is no .tech domain and no Droplet, so those two are dropped. The demo lives at the Render URL.

**D20. Keep the free Render instance warm.** A free instance sleeps after 15 idle minutes and the first request then takes 30 s or more. `.github/workflows/keepwarm.yml` calls `/healthz` every 10 minutes. GitHub may delay scheduled runs, so the README still warns about a slow first request.
