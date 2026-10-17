# Pata

Dispatch triage for couriers. Pata reads a raw customer address plus order context and returns a decision before the van leaves: `DELIVERABLE`, `CALL_FIRST`, or `LIKELY_FAIL`, with normalised fields, a one-line driver note, and the reason.

Entry for the DEV Hacktoberfest Open-Source AI Challenge, Week 2: Side Hustle. Built by Abdul Wasih and Usman.

Status: in progress (17 October 2026).

## How it works

1. `pata/rules.py`: deterministic rules (pure functions, tested). They label the training data and are the fallback at serve time.
2. A 4B open-weight model (`Qwen/Qwen3.5-4B`) fine-tuned with LoRA on Tinker produces the same JSON contract (`pata/schema.py`).
3. The tuned model is served with vLLM on a DigitalOcean GPU Droplet.
4. A FastAPI gateway on Render validates every model answer against the schema and falls back to the rules if it fails. Every request is traced in Sentry.

## Run

```
uv sync
uv run pytest
uv run ruff check .
```

## Data provenance

* India: public Hugging Face datasets, Apache 2.0 (details to follow).
* UAE: **synthetic**, generated from real-world address patterns. No real customer addresses are in this repo.

## Decisions

See [DECISIONS.md](DECISIONS.md).

## Post-deadline changes

None.
