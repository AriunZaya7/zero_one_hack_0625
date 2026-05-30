"""Pre-download HF model weights on the Leonardo LOGIN node (which has internet),
so compute nodes can run with HF_HUB_OFFLINE=1.

Run once on the login node:
    python scripts/prefetch_models.py
Then in jobs:  export HF_HUB_OFFLINE=1

Weights land in $HF_HOME (set it to a $WORK/$SCRATCH path with quota, e.g.
    export HF_HOME=$WORK/hf_cache
"""
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    # "Qwen/Qwen2.5-7B",   # uncomment if you have the quota / time
]

if __name__ == "__main__":
    for m in MODELS:
        print(f"downloading {m} ...", flush=True)
        snapshot_download(repo_id=m)
    print("done. set HF_HUB_OFFLINE=1 in your job scripts.")
