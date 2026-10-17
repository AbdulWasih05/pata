---
title: A $2 fine-tune that tells a courier which parcels will fail before the van leaves
published: false
tags: devchallenge, hf26challenge, opensource, ai
cover_image: [TODO: cover image URL]
---

<!--
DRAFT. Numbers come from eval/results.md, train/RUNS.md and DECISIONS.md (D16 to D18).
Lines marked [TODO] need a human. Delete this comment before publishing.
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge, Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12).*

## What I Built

A courier van leaves the depot with 120 parcels. Somewhere in that pile is "behind Lulu, call before coming", "Bandra West, Mumbai" with no house number, and a cash-on-delivery order whose PIN code points to a different city. Each one is a wasted stop: the driver's time, a return trip, a COD payment that never gets collected, and a customer who now trusts the shop less.

**Pata** reads the raw address the customer typed, plus the order context (market, cash on delivery or prepaid, whether there is a phone number), and returns a dispatch decision before the van leaves:

* `DELIVERABLE`: the driver can find it.
* `CALL_FIRST`: call the customer before loading it, with a one-line call script.
* `LIKELY_FAIL`: hold the parcel.

It also returns the address split into fields, the reason codes, and a one-line note for the driver. "Pata" is Hindi and Urdu for "address", and also "to find out".

Who pays: small couriers, cash-on-delivery sellers and 3PL operations teams in India and the Gulf, where a large share of orders are COD and addresses are written as directions rather than as postal records.

Three real outputs from the tuned model:

| Customer typed | Context | Decision | Driver note |
|---|---|---|---|
| `Flat 1203, Marina Pinnacle Tower, Dubai Marina` | UAE, COD | DELIVERABLE | Go to Flat 1203, Marina Pinnacle Tower, Dubai Marina. |
| `behind Lulu Hypermarket Al Barsha, call before coming` | UAE, COD | CALL_FIRST | Call customer before dispatch: landmark only; COD on weak address. |
| `H.No 45, MG Road, Bengaluru 560001` | India, declared PIN 110001 | LIKELY_FAIL | Hold: PIN does not match. |

For the second one it also writes the call script: *"Hi, this is your courier. Could you confirm your building or house number? Payment is cash on delivery."*

## Demo

[TODO: Render URL]

[TODO: 90 second video: paste three addresses, one per decision; upload a CSV; show a Sentry trace]

Paste one address, or upload a CSV of the morning's orders and download it back with a decision column. There is also a CLI: `pata triage --market AE --in orders.csv --out triaged.csv`.

Want your own copy? The repo has a **Deploy to Render** button: one free web service, it asks for a Tinker key (leave it empty to run on rules alone) and an optional Sentry DSN.

## Code

{% github AbdulWasih05/pata %}

## How I Built It

**Rules first.** Before any model, I wrote the decision rules as pure Python functions with tests (`pata/rules.py`, 82 tests across the repo). The rules label the training data and act as the fallback at serve time. Every judgement call is written down in `DECISIONS.md` (19 of them), for example: an Indian address with no PIN code at all is `CALL_FIRST`, because a courier cannot route it.

**Data.**

* India, real: company-registry (MCA) addresses from [gagan1985/indian-addresses-gold](https://huggingface.co/datasets/gagan1985/indian-addresses-gold) and [gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw), Apache 2.0. The same datasets also contain bank KYC addresses; I left those out, because they are real customers' homes.
* India, customer-style: [abr-6024/indian-address-parsing-synthetic](https://huggingface.co/datasets/abr-6024/indian-address-parsing-synthetic), Apache 2.0.
* UAE: **synthetic.** There is no public dataset of Gulf delivery addresses, so `data/ae_generator.py` generates them from common Gulf delivery-address patterns: villa and street numbers, Makani numbers, PO Boxes (which cannot be delivered to a door), "behind Lulu", Arabic script and Arabizi, "will share location on WhatsApp", Al Nahda that exists in both Dubai and Sharjah, labour camps. No real customer addresses are in the repo.
* Registry addresses are cleaner than what customers type, so `data/augment.py` adds customer noise (landmarks, "call me", Hinglish "mandir ke paas", typos, lowercase, missing PIN) and order context (60% COD, some orders without a phone, some with a wrong declared PIN). The target fields are edited along with the text, so the model learns to copy rather than invent.
* Splits: 3,000 India and 2,966 UAE training conversations, about 1.5M tokens per epoch. India test rows come from whole PIN regions held out of training, so test street and building names are not in the training set.

**Fine-tuning on Tinker.** LoRA SFT of `Qwen/Qwen3.5-4B` with the `tinker_cookbook` supervised recipe: rank 32, learning rate 4.9e-4 from the cookbook's `get_lr`, batch 64, thinking disabled (the `qwen3_5_disable_thinking` renderer), loss only on the assistant JSON. The system prompt is three sentences; the model learns the schema from the data.

| Run | Epochs | Tokens trained | Wall time | Cost |
|---|---|---|---|---|
| run1 | 3 | 4,528,137 | 19.4 min | $3.21 |
| run2 | 2 | 3,013,890 | 12.8 min | $2.14 |

Costs are from `tinker billing usage` at the published $0.737 per million training tokens. Everything on Tinker so far, including every eval, came to $6.60.

**The run that went wrong.** Run 1's first checkpoint scored worse than I expected on India: 45 parcels that should have been held or called came back `DELIVERABLE`. I read all 45. In 30 of them the model was right and my rules were wrong. "h.no 323, Balaji Heights" and "#743, Gokul Bhavan" were labelled "missing flat number", but the house number *is* the flat. By the end of training, run 1 had learned my bug: 38 genuinely deliverable test addresses came back `CALL_FIRST`. I fixed the rule (with regression tests), which moved 165 training labels, and trained run 2 on the corrected labels for 2 epochs instead of 3. A small model trained on rules can still be a useful check on those rules, but only if you read the disagreements.

**Serving.** The gateway calls the run 2 checkpoint through Tinker's OpenAI-compatible endpoint, so there is no GPU box to run. The same client talks to any vLLM server, and the repo has the merge-and-serve path (`train/export_hf.py`, `deploy/do_vllm.md`) for anyone who wants the weights in-house.

**Gateway.** A FastAPI service on Render (`render.yaml`) serves the page and the API. For every request it computes the rules answer, calls the model, validates the JSON against the pydantic schema, and falls back to the rules if the call fails or the JSON is invalid. The gateway records which one it served.

## Sentry: the 6-second requests

Every `/triage` request is a Sentry agent run: `invoke_agent pata` with child spans `execute_tool rules_prelabel`, `chat` (model, provider, input and output tokens), `schema.validate`, and `execute_tool rules_fallback` when taken. Customer addresses are personal data, so the spans carry lengths, decisions, token counts and flags, not the address text.

[TODO: screenshot 1, docs/sentry/: trace list or chat span durations for environment local-hedge-0, showing the two bands]

The first afternoon of traces had a pattern I would not have seen in logs: `chat` spans were either about 1.1 seconds or about 6.1 seconds, almost nothing in between. Token counts were the same in both groups, so it was not the model writing more. I tried streaming (the first token itself arrived late), fresh connections and parallel requests: the extra five seconds stayed. It is a hold on the server side, and it hit 12 of 40 single requests, which is exactly how a dispatcher pastes addresses.

The fix is an old one: a hedged request. If the first call has not answered in 1.5 seconds, the client sends the same request again and keeps whichever answer arrives first (`pata/client.py`, tests in `tests/test_client.py`). The `chat` span now records `pata.hedged` and `pata.hedge_won`, so the trade-off stays visible.

[TODO: screenshot 2, docs/sentry/: environment local-hedge-1500, same view, with pata.hedged on the chat span]

Forty sequential requests through the real gateway, before and after:

| | Hedging off | Hedging on |
|---|---|---|
| p50 | 1.09 s | 1.14 s |
| p90 | 6.16 s | 3.15 s |
| Mean | 2.63 s | 1.69 s |
| Requests over 5 s | 12 of 40 | 1 of 40 |

The cost is a second call on slow requests only, at about $0.0002 each.

## Eval

**What "gold" means here, honestly.** The rules label the training data, so testing against rule labels mostly checks that the model copied the rules. The test set needs labels from someone else. We did not have time to hand-label 300 rows, so each market's 150 test rows were labelled by a separate Claude agent that saw only the address and the order context, never the rules' answer, the silver label or the code. That is an LLM's judgement, not a dispatcher's, and I treat it as a second opinion rather than ground truth.

The column that costs money is **false-DELIVERABLE**: of the parcels that should have been held or called first, how many would have gone out on the van.

**India** (150 test addresses)

| System | Valid JSON | Accuracy | False-DELIVERABLE | Building field | Area field | p50 latency |
|---|---|---|---|---|---|---|
| Qwen3.5-4B base, zero-shot | 85.3% | 43.3% | 26.4% (28/106) | 52.3% | 76.6% | 1.07 s |
| Rules alone | n/a | 86.7% | 8.5% (9/106) | 47.3% | 14.0% | 0 s |
| **Pata, tuned 4B (run 2)** | **100%** | **86.0%** | 10.4% (11/106) | **89.3%** | **87.3%** | 1.15 s |

**UAE** (150 synthetic test addresses)

| System | Valid JSON | Accuracy | False-DELIVERABLE | Building field | p50 latency |
|---|---|---|---|---|---|
| Qwen3.5-4B base, zero-shot | 82.0% | 25.3% | 0.0% (0/77) | 74.0% | 1.07 s |
| Rules alone | n/a | 84.7% | 0.0% (0/77) | 92.0% | 0 s |
| **Pata, tuned 4B (run 2)** | **100%** | **85.3%** | 0.0% (0/77) | **100%** | 1.13 s |

What this says:

* **Fine-tuning doubled decision accuracy** over the same model zero-shot (43% to 86% India, 25% to 85% UAE), and took valid JSON from 85% (India) and 82% (UAE) to 100%.
* **On decisions, the tuned model matches the rules it learned from,** within a point. That is expected, and it is not the reason the model exists.
* **The reason is the fields.** On India, the rules find the building in 47% of addresses and the area in 14%; the tuned model gets 89% and 87%, because it learned field boundaries from human span labels in the registry data rather than from my regexes. A driver needs "Flat 14, Balaji Heights, Kondapur", not just "DELIVERABLE".
* The base model's 0% false-DELIVERABLE on UAE is not caution, it is a habit: it answers `CALL_FIRST` for almost everything, including 50 of 73 addresses that were fine.

**Where the reviewer disagreed with my rules.** Both the rules and the tuned model disagree with the blind reviewer on about 1 row in 7. The review found two places where my rules are stricter than a dispatcher would be. `MIXED_SCRIPT` sends every Arabic-script address to `CALL_FIRST`, even "Flat 328, بناية الشمس, Al Karama, Dubai", where building, flat and area are all clear. And I treat Al Quoz and Khalifa City as ambiguous without a sub-area, even when a warehouse and plot number are given. I did not change the rules and retrain after seeing this, because that would tune the model to its own test set. They are the first fixes for the next run.

## Why Does Open Innovation Matter?

1. **Per-parcel cost.** A triage call is about 130 input and 135 output tokens. At Tinker's published sampling rates ($0.33 per million prefill, $1.005 per million sampled) that is about $0.00018 a parcel, or **$1.80 for 10,000 parcels a month**, plus the occasional hedged second call. Training it cost $2.14.
2. **You cannot fine-tune a closed API on Gulf address quirks.** "Al Nahda" without an emirate, a Makani number, a PO Box that is not a door: the base model got 25% of the UAE test set right. One $2 LoRA run got it to 85% against an independent reviewer.
3. **Customer addresses can stay in-house.** An address plus a phone number plus "cash on delivery" is data you should not hand to a third party per parcel. The demo calls Tinker for convenience, but the weights are open: merge them (`train/export_hf.py`) and serve them with vLLM on a box the courier controls. The gateway does not change.
4. **No lock-in.** The rules, the data generator, the training script and the eval are all in the repo. A courier can retrain on its own 10,000 historical orders without asking anyone.

## My Agent Session

[TODO: Wasih, DevRelay session embed, or remove this section]

## Honest limitations

* **The gold labels are an LLM's, not a dispatcher's.** They are independent of my rules, which is what the test needs, but they are not field truth. Real delivery outcomes would be better than both.
* **The UAE data is synthetic.** It comes from real patterns, but it is generated, and the generator was checked against the same rules that label it.
* **The rules are too strict in two known places** (Arabic script, large areas with sub-areas). The model inherited both.
* **The demo depends on Tinker's OpenAI-compatible endpoint,** which Tinker describes as a beta for testing and low traffic. A busy courier should run the merged weights on vLLM.
* **No geocoding.** Pata never checks that "Marina Pinnacle Tower" exists in Dubai Marina. A gazetteer lookup is the obvious next tool.
* **What a real courier's data would change:** 10,000 historical orders with their actual delivery outcomes would replace rule labels with ground truth. That is the first thing I would ask a paying customer for.

## Prize Categories

* **Best Use of Tinker:** two LoRA SFT runs of Qwen3.5-4B ($3.21 and $2.14), base vs tuned on the same held-out set in the Eval section, the failed run and what it taught me, run IDs and costs in `train/RUNS.md`. The live demo is served from the Tinker checkpoint.
* **Best Use of Render:** the gateway and UI run on Render from a one-click `render.yaml` blueprint; anyone can deploy their own copy with the button in the README.
* **Best Use of Sentry Agent Tracing:** every request is traced as an agent run; the traces exposed the 6-second requests, and the hedging fix cut p90 latency from 6.16 s to 3.15 s.

Built by Abdul Wasih (@iamwasih).
