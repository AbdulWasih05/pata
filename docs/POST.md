---
title: I fine-tuned a 4B model to tell a courier which parcels will fail before the van leaves
published: false
tags: devchallenge, hf26challenge, opensource, ai
cover_image: [TODO: cover image URL]
---

<!--
DRAFT. Every number below comes from eval/results.md, train/RUNS.md or deploy/do_vllm.md.
Lines marked [TODO] are waiting on a measurement or a human. Delete this comment before publishing.
Title option with numbers, once gold results are in:
"A $[cost] fine-tune cut false 'deliverable' calls from [base]% to [tuned]% on Indian and Gulf addresses"
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge, Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12).*

## What I Built

A courier van leaves the depot with 120 parcels. Somewhere in that pile is "behind Lulu, call before coming", "Bandra West, Mumbai" with no house number, and a cash-on-delivery order whose PIN code points to a different city. Each one is a wasted stop: [TODO: Usman, failed-delivery cost per parcel from experience, stated as experience], plus a return trip, plus a customer who now trusts the shop less.

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

[TODO: https://<name>.tech]

[TODO: 90 second video: paste three addresses, one per decision; upload a CSV; show a Sentry trace]

You can paste one address or upload a CSV of orders and download it back with a decision column. There is also a CLI: `pata triage --market AE --in orders.csv --out triaged.csv`.

## Code

{% github AbdulWasih05/pata %}

## How I Built It

**Rules first.** Before any model, I wrote the decision rules as pure Python functions with tests (`pata/rules.py`, 72 tests across the repo). The rules label the training data and act as the fallback at serve time. Every judgement call beyond the obvious is written down in `DECISIONS.md` (15 so far), for example: an Indian address with no PIN code at all is `CALL_FIRST`, because a courier cannot route it.

**Data.**

* India, real: company-registry (MCA) addresses from [gagan1985/indian-addresses-gold](https://huggingface.co/datasets/gagan1985/indian-addresses-gold) and [gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw), Apache 2.0. The same datasets also contain bank KYC addresses; I left those out, because they are real customers' homes.
* India, customer-style: [abr-6024/indian-address-parsing-synthetic](https://huggingface.co/datasets/abr-6024/indian-address-parsing-synthetic), Apache 2.0.
* UAE: **synthetic.** There is no public dataset of Gulf delivery addresses, so `data/ae_generator.py` generates them from patterns Usman knows from [TODO: Usman, years] of Gulf logistics work: villa and street numbers, Makani numbers, PO Boxes (which cannot be delivered to a door), "behind Lulu", Arabic script and Arabizi, "will share location on WhatsApp", Al Nahda that exists in both Dubai and Sharjah, labour camps. No real customer addresses are in the repo.
* Registry addresses are cleaner than what customers type, so `data/augment.py` adds customer noise (landmarks, "call me", Hinglish "mandir ke paas", typos, lowercase, missing PIN) and order context (60% COD, some orders without a phone, some with a wrong declared PIN). The target fields are edited along with the text, so the model learns to copy rather than invent.
* Splits: 3,000 India and 2,966 UAE training conversations, about 1.5M tokens per epoch. India test rows come from whole PIN regions held out of training, so test street and building names are not in the training set.

**Fine-tuning on Tinker.** LoRA SFT of `Qwen/Qwen3.5-4B` with the `tinker_cookbook` supervised recipe: rank 32, learning rate 4.9e-4 from the cookbook's `get_lr`, batch 64, thinking disabled (the `qwen3_5_disable_thinking` renderer), loss only on the assistant JSON. The system prompt is three sentences; the model learns the schema from the data.

| Run | Epochs | Tokens trained | Wall time | Cost |
|---|---|---|---|---|
| run1 | 3 | 4,528,137 | 19.4 min | [TODO: tinker billing] |
| run2 | 2 | 3,013,890 | 12.8 min | [TODO: tinker billing] |

**The run that went wrong.** Run 1's first checkpoint scored worse than I expected on India: 45 parcels that should have been held or called came back `DELIVERABLE`. I read all 45. In 30 of them the model was right and my rules were wrong. "h.no 323, Balaji Heights" and "#743, Gokul Bhavan" were labelled "missing flat number", but the house number *is* the flat. By the end of training, run 1 had learned my bug: 38 genuinely deliverable test addresses came back `CALL_FIRST`. I fixed the rule (with regression tests), which moved 165 training labels, and trained run 2 on the corrected labels for 2 epochs instead of 3. Both runs are scored on the same corrected labels below. A small model trained on rules can still be a useful check on those rules, but only if you read the disagreements.

**Serving.** The LoRA adapter is merged into full Hugging Face weights with the cookbook's `build_hf_model` and served by vLLM on a DigitalOcean GPU Droplet ([TODO: GPU and size]) behind an API key. Exact commands are in `deploy/do_vllm.md`.

**Gateway.** A FastAPI service on Render (`render.yaml`) serves the page and the API. For every request it computes the rules answer, calls the model, validates the JSON against the pydantic schema, and falls back to the rules if the call fails or the JSON is invalid. The gateway records which one it served.

**Tracing.** Every `/triage` request is a Sentry agent run: `invoke_agent pata` with child spans `execute_tool rules_prelabel`, `chat pata` (model, input and output tokens), `schema.validate`, and `execute_tool rules_fallback` when taken. Customer addresses are personal data, so by default the spans carry lengths, decisions and token counts, not the address text.

[TODO: Sentry section: 2 to 3 trace screenshots from docs/sentry/, the real problem the traces exposed, the fix, before/after numbers]

## Eval

Held-out test set, base model zero-shot (same system prompt plus a JSON schema hint) against the tuned model. The column that costs money is **false-DELIVERABLE**: of the parcels that should have been held or called first, how many would have gone out on the van.

[TODO: replace with the gold (hand-checked) table from eval/results.md once data/gold is filled in. Current numbers are on the silver test set, labelled by the rules.]

**India** (590 test addresses, silver labels)

| System | Valid JSON | Accuracy | False-DELIVERABLE | Building field | Area field |
|---|---|---|---|---|---|
| Qwen3.5-4B base, zero-shot | 87.1% | 61.0% | 29.4% (86/293) | 51.9% | 78.0% |
| Pata, tuned 4B (run 2) | 100% | 94.9% | 4.4% (13/293) | 92.7% | 88.0% |

**Why a model, if rules make the labels?** Decisions are only half the output. The same rules that label decisions get 41.7% of India building fields and 10.5% of area fields right; the tuned model gets 92.7% and 88.0%, because it learned field boundaries from the human-labelled gold spans rather than from regexes. Rules also cannot read an address they have no pattern for, and the model generalised past one of my rule bugs (see the run that went wrong).

**UAE** (334 synthetic test addresses, silver labels)

| System | Valid JSON | Accuracy | False-DELIVERABLE | Building field |
|---|---|---|---|---|
| Qwen3.5-4B base, zero-shot | 79.3% | 33.2% | 1.5% (3/196) | 71.3% |
| Pata, tuned 4B (run 2) | 100% | 99.4% | 1.0% (2/196) | 100% |

The base model's low UAE false-DELIVERABLE rate is not a strength: it answers `CALL_FIRST` for almost everything, including 96 of 138 addresses that were fine. The tuned model's invalid-answer rate is zero on both sets, so the rules fallback never fired in this eval.

Serving latency on the Droplet: [TODO: p50 for a ~300 token prompt, tokens/sec from vLLM logs].

## Why Does Open Innovation Matter?

1. **Customer addresses stay on the courier's own machine.** An address plus a phone number plus "cash on delivery" is exactly the data you should not be sending to a third party per parcel. With open weights the model runs on a box the courier controls.
2. **Per-parcel cost.** [TODO: arithmetic for 10,000 parcels a month: Droplet hourly price x hours, divided by parcels, against a hosted API at the measured ~[input] + [output] tokens per parcel.]
3. **You cannot fine-tune a closed API on Gulf address quirks.** "Al Nahda" without an emirate, a Makani number, a PO Box that is not a door: the base model got 33% of the UAE test set right. Two short LoRA runs got it to 99% on our synthetic set.
4. **No pricing lock-in.** The weights, the data generator and the rules are all in the repo. A courier can retrain on its own 10,000 historical orders without asking anyone.

## My Agent Session

[TODO: Wasih, DevRelay session embed, or remove this section]

## My First User

[TODO: Usman, if a courier or seller runs 20 of their own addresses before Monday, write what they said and what they asked for next. Otherwise remove this section.]

## Honest limitations

* **The UAE data is synthetic.** It comes from real patterns, but it is generated, and the generator was checked against the same rules that label it. That is why the UAE silver numbers look nearly perfect and why the hand-checked set matters more. [TODO: gold UAE numbers]
* **Labels are silver.** They come from rules, and the rules had at least one real bug that I found only because the model disagreed. There may be others. [TODO: agreement rate between rules and human hand-checks]
* **No geocoding.** Pata never checks that "Marina Pinnacle Tower" exists in Dubai Marina. A real gazetteer lookup is the obvious next tool.
* **Field extraction is imperfect.** For "H.No 45, MG Road" the model put the house number in `unit` instead of `building`. Area extraction on India is [TODO: field_area from gold]%.
* **Arabic script.** [TODO: quantify from gold: accuracy on Arabic and Arabizi rows]
* **What a real courier's data would change:** 10,000 historical orders with their actual delivery outcomes would replace rule labels with ground truth. That is the first thing I would ask a paying customer for.

## Prize Categories

* **Best Use of Tinker:** two LoRA SFT runs of Qwen3.5-4B on Tinker, base vs tuned on the same held-out set in the Eval section, run IDs and cost in `train/RUNS.md`.
* **Best Use of DigitalOcean:** the tuned model is served with vLLM on a GPU Droplet; spec, launch command and throughput in `deploy/do_vllm.md`. [TODO]
* **Best Use of Render:** the gateway and UI run on Render from `render.yaml`. [TODO: live URL]
* **Best Use of Sentry Agent Tracing:** every request is traced as an agent run; the Sentry section shows the traces, the problem they exposed, and the fix. [TODO]
* **Best Use of a .tech Domain:** Pata lives at [TODO: domain].

Built with [TODO: Usman's DEV handle].
