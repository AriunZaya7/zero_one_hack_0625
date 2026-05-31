"""Pre-download HF model weights on the Leonardo LOGIN node (which has internet),
so compute nodes can run with HF_HUB_OFFLINE=1.

Run once on the login node:
    export HF_HOME=$SCRATCH/hf_cache
    python scripts/prefetch_models.py

Then in jobs:  export HF_HUB_OFFLINE=1
               export HF_HOME=$SCRATCH/hf_cache
"""
import os
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    # "Qwen/Qwen2.5-7B",   # uncomment if you have the quota / time
]

if __name__ == "__main__":
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        scratch = os.environ.get("SCRATCH", "")
        if scratch:
            hf_home = os.path.join(scratch, "hf_cache")
            os.environ["HF_HOME"] = hf_home
            print(f"HF_HOME not set — defaulting to $SCRATCH/hf_cache: {hf_home}")
        else:
            print("WARNING: neither HF_HOME nor SCRATCH is set. "
                  "Weights will land in the default HF cache (~/.cache/huggingface).")
    else:
        print(f"Using HF_HOME={hf_home}")

    for m in MODELS:
        print(f"\nDownloading {m} ...", flush=True)
        snapshot_download(repo_id=m)
        print(f"  ✓ done: {m}", flush=True)

    print("\nAll models staged. Set HF_HUB_OFFLINE=1 in your job scripts.")
