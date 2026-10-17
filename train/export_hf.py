"""Download a Tinker LoRA checkpoint and merge it into full Hugging Face weights for vLLM.

Run on the GPU droplet (needs ~20 GB disk and ~16 GB RAM for the 4B bf16 merge):

    uv run --env-file .env python -m train.export_hf \
        --checkpoint tinker://<run>/sampler_weights/final --out /models/pata-4b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tinker_cookbook import weights

BASE_MODEL = "Qwen/Qwen3.5-4B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="tinker:// sampler checkpoint path")
    ap.add_argument("--out", type=Path, required=True, help="merged model dir (must not exist)")
    ap.add_argument("--adapter-dir", type=Path, default=Path("train/out/adapter"))
    args = ap.parse_args()

    adapter = weights.download(tinker_path=args.checkpoint, output_dir=str(args.adapter_dir))
    print(f"adapter downloaded to {adapter}")
    weights.build_hf_model(base_model=BASE_MODEL, adapter_path=adapter, output_path=str(args.out))
    print(f"merged model written to {args.out}")


if __name__ == "__main__":
    main()
