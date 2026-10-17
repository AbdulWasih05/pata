# Training runs

Every Tinker run, including failed ones. Eval numbers live in `eval/results.md`.

## run1 (completed)

* started: 2026-10-17 06:18 UTC, wall time 19.4 min
* base: `Qwen/Qwen3.5-4B`, renderer `qwen3_5_disable_thinking`
* LoRA rank 32, lr 4.91e-04 (linear), batch 64, epochs 3
* data: `train/out/train_conversations.jsonl`, 1,509,379 tokens per epoch, 4,528,137 tokens trained
* cost: $3.21 (4,353,196 billed training tokens at $0.737 per million, `tinker billing usage`, rate-card estimate before invoice reconciliation)
* final checkpoint: `tinker://edae8b5c-4afe-5a61-8322-f6439c9b17f1:train:0/sampler_weights/final`
* state: `tinker://edae8b5c-4afe-5a61-8322-f6439c9b17f1:train:0/weights/final`
* last metrics: {'train_mean_nll': 0.0074}
* note: first run, cookbook default LR, rank 32, 3 epochs

## run2 (completed)

* started: 2026-10-17 06:23 UTC, wall time 12.8 min
* base: `Qwen/Qwen3.5-4B`, renderer `qwen3_5_disable_thinking`
* LoRA rank 32, lr 4.91e-04 (linear), batch 64, epochs 2
* data: `train/out/train_conversations.jsonl`, 1,506,945 tokens per epoch, 3,013,890 tokens trained
* cost: $2.14 (2,898,191 billed training tokens at $0.737 per million, same source)
* final checkpoint: `tinker://66ebff6a-2896-584d-9189-64032bad9b83:train:0/sampler_weights/final`
* state: `tinker://66ebff6a-2896-584d-9189-64032bad9b83:train:0/weights/final`
* last metrics: {'train_mean_nll': 0.0105}
* note: labels fixed per D15; 2 epochs since run1 dev NLL rose early in epoch 2

## What went wrong in run 1

Run 1 trained on labels with a rules bug: a house number in front of a named building ("h.no 323, Balaji Heights") was labelled MISSING_UNIT. At epoch 1 the model already disagreed with those labels on held-out data and was right; by epoch 2 it had learned the bug, and 38 deliverable India addresses in test came back as CALL_FIRST. The rules fix (DECISIONS.md D15) moved 165 India training labels. Run 2 trained on the corrected labels for 2 epochs instead of 3 and is the deployed candidate. Both runs are scored on the same corrected test labels in `eval/results.md`.

## Cost summary

From `tinker -f json billing usage 2026-10-16T00:00:00Z 2026-10-18T00:00:00Z`. Billed token counts are slightly lower than our rendered counts, which include the 200 held-out conversations.

| Item | Cost |
|---|---|
| run1 training | $3.21 |
| run2 training | $2.14 |
| eval and example sampling to 17 Oct (prefill $0.33/M, sample $1.005/M) | $1.24 |
| storage | $0.01 |
| **Total to end of 17 Oct UTC** | **$6.60** |
