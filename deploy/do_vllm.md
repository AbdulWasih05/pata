# Serving Pata on a DigitalOcean GPU Droplet with vLLM

The tuned model is a merged Qwen3.5-4B (bf16, about 8 GB of weights). Any single-GPU Droplet
with 24 GB or more of VRAM serves it with room for a large KV cache.

Fill in the bracketed values as they are recorded; the post quotes this file.

## 1. Droplet

* Image: DigitalOcean AI/ML Ready (Ubuntu, NVIDIA drivers and CUDA preinstalled)
* GPU: [size slug and GPU model, e.g. gpu-l40sx1-48gb]
* Region: [region]
* Firewall: allow 22 from your IP only, and 8000 from anywhere (vLLM checks the API key on every
  request). Render has no static outbound IP on the starter plan, so an IP allow-list for 8000 is
  not possible; the API key is the lock.

## 2. Install

```bash
apt-get update && apt-get install -y git
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/AbdulWasih05/pata.git && cd pata
# .env with TINKER_API_KEY (for the export step only); never commit it
uv venv --python 3.12 && uv pip install vllm
```

## 3. Base model first (end-to-end check before tuned weights exist)

```bash
export VLLM_API_KEY=$(openssl rand -hex 24)   # paste into Render as DO_VLLM_API_KEY
uv run vllm serve Qwen/Qwen3.5-4B --served-model-name pata \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.85
```

Set `PATA_SCHEMA_HINT=1` on Render while the base model is serving.

## 4. Tuned weights

```bash
uv sync --group train
uv run --env-file .env python -m train.export_hf \
  --checkpoint [tinker:// sampler path from train/RUNS.md] --out /models/pata-4b
uv run vllm serve /models/pata-4b --served-model-name pata \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.85
```

Set `PATA_SCHEMA_HINT=0` on Render. Run it under `tmux` or a systemd unit so it survives logout.

## 5. Smoke test

```bash
curl -s http://[droplet-ip]:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"pata","temperature":0,"chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"{\"market\":\"AE\",\"raw_address\":\"Villa 12, Street 7, Al Barsha 2\"}"}]}'
```

## 6. Numbers for the post

* tokens/sec observed: [from vLLM logs, "Avg generation throughput"]
* p50 latency for a ~300 token prompt: [from `eval/results.md`, served-model row]

## 7. Before and after judging

* Snapshot the Droplet before the deadline.
* Destroy it after judging. The merged weights can be rebuilt from the Tinker checkpoint.
