"""LoRA SFT of Qwen3.5-4B on Tinker with the tinker_cookbook supervised recipe.

Runs on Linux only (tinker_cookbook needs tml-renderers). On the Windows dev box use WSL:

    uv run python -m train.build_jsonl
    uv run --env-file .env python -m train.train_tinker --run-name run1

Every run, including failed ones, is appended to train/RUNS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from tinker_cookbook import hyperparam_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5_disable_thinking"  # thinking off: fast, deterministic JSON
TRAIN_FILE = Path("train/out/train_conversations.jsonl")
RUNS_DIR = Path("train/out/runs")
RUNS_MD = Path("train/RUNS.md")
DEV_HOLDOUT = 200  # NLL is tracked on this many training conversations held out by the recipe


def count_tokens(path: Path, renderer_name: str, model: str) -> int:
    """Exact token count of the rendered conversations, for the cost line in RUNS.md."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = renderers.get_renderer(renderer_name, get_tokenizer(model))
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        model_input, _ = renderer.build_supervised_example(json.loads(line)["messages"])
        total += model_input.length
    return total


def build_config(args: argparse.Namespace, log_path: Path) -> train.Config:
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=RENDERER,
        max_length=2048,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common, file_path=str(TRAIN_FILE), test_size=DEV_HOLDOUT, shuffle_seed=0
    )
    n = sum(1 for _ in TRAIN_FILE.open(encoding="utf-8")) - DEV_HOLDOUT
    steps_per_epoch = max(1, n // args.batch_size)
    return train.Config(
        log_path=str(log_path),
        recipe_name="pata_triage_sft",
        model_name=args.model,
        renderer_name=RENDERER,
        dataset_builder=dataset,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.rank,
        save_every=steps_per_epoch,  # one checkpoint per epoch
        eval_every=max(1, steps_per_epoch // 4),
    )


def _last_checkpoint(log_path: Path) -> dict:
    f = log_path / "checkpoints.jsonl"
    if not f.exists():
        return {}
    lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[-1] if lines else {}


def _last_metrics(log_path: Path) -> dict:
    f = log_path / "metrics.jsonl"
    if not f.exists():
        return {}
    lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[-1] if lines else {}


def log_run(args, log_path: Path, tokens: int, status: str, minutes: float, note: str) -> None:
    ckpt = _last_checkpoint(log_path)
    m = _last_metrics(log_path)
    nll = {k: round(v, 4) for k, v in m.items() if "nll" in k and isinstance(v, float)}
    train_tokens = tokens * args.epochs
    if not RUNS_MD.exists():
        RUNS_MD.write_text(
            "# Training runs\n\nEvery Tinker run, including failed ones. Eval numbers live in "
            "`eval/results.md`.\n",
            encoding="utf-8",
        )
    with RUNS_MD.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## {args.run_name} ({status})\n\n"
            f"* started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}, "
            f"wall time {minutes:.1f} min\n"
            f"* base: `{args.model}`, renderer `{RENDERER}`\n"
            f"* LoRA rank {args.rank}, lr {args.lr:.2e} (linear), batch {args.batch_size}, "
            f"epochs {args.epochs}\n"
            f"* data: `{TRAIN_FILE}`, {tokens:,} tokens per epoch, "
            f"{train_tokens:,} tokens trained\n"
            f"* final checkpoint: `{ckpt.get('sampler_path', 'none')}`\n"
            f"* state: `{ckpt.get('state_path', 'none')}`\n"
            f"* last metrics: {nll or 'none'}\n"
            f"* note: {note}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="default: cookbook get_lr()")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true", help="count tokens and print config only")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = hyperparam_utils.get_lr(args.model, is_lora=True)

    log_path = RUNS_DIR / args.run_name
    if log_path.exists() and not args.dry_run:
        raise SystemExit(f"{log_path} exists; pick a new --run-name (runs are never overwritten)")
    tokens = count_tokens(TRAIN_FILE, RENDERER, args.model)
    est_cost = tokens * args.epochs / 1e6
    print(f"{tokens:,} tokens/epoch; {tokens * args.epochs:,} total; lr {args.lr:.2e}")
    print(f"price-per-M-tokens x {est_cost:.2f} = estimated cost (check `tinker` billing)")
    config = build_config(args, log_path)
    steps = math.ceil((sum(1 for _ in TRAIN_FILE.open()) - DEV_HOLDOUT) / args.batch_size)
    print(f"~{steps} steps/epoch, checkpoint every {config.save_every} steps")
    if args.dry_run:
        return

    t0 = time.time()
    try:
        asyncio.run(train.main(config))
    except BaseException as e:
        log_run(args, log_path, tokens, "failed", (time.time() - t0) / 60,
                f"{type(e).__name__}: {e}"[:300] + (f" {args.note}" if args.note else ""))
        raise
    log_run(args, log_path, tokens, "completed", (time.time() - t0) / 60, args.note or "-")
    print(f"done; checkpoint: {_last_checkpoint(log_path).get('sampler_path')}")


if __name__ == "__main__":
    main()
