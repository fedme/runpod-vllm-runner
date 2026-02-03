# RunPod vLLM Worker (MedGemma Fork)

Fork of the official [RunPod vLLM Worker](https://github.com/runpod-workers/worker-vllm) with fixes for [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it) and Gemma 3 model support on vLLM v0.15.0.

## What changed

- Removed deprecated vLLM fields from engine args (including `rope_scaling` which broke Gemma 3 models)
- Fixed `engine_args.py` dumping all env vars into engine config
- Added `transformers` upgrade and `timm` for MedGemma's vision backbone
- Defaults `dtype` to `bfloat16` for Gemma 3 / MedGemma models
- Added `HF_OVERRIDES` env var support

## Recommended environment variables

```
MODEL_NAME=google/medgemma-1.5-4b-it
DTYPE=bfloat16
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.95
LIMIT_MM_PER_PROMPT=image=1
```

## Links

- [Official RunPod vLLM Worker](https://github.com/runpod-workers/worker-vllm)
- [RunPod vLLM Worker docs](https://docs.runpod.io/workers/vllm/overview)
- [MedGemma 1.5 on HuggingFace](https://huggingface.co/google/medgemma-1.5-4b-it)
