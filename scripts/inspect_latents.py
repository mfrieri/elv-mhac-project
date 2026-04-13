"""
Inspect a saved latents .npz file produced by save_latents.py.

Usage:
    python scripts/inspect_latents.py latents/fourrooms_mhac_k_latents.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="Path to latents .npz file")
    args = p.parse_args()

    path = Path(args.npz)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = np.load(path)
    print(f"\n── {path.name} ──────────────────────────────────────")
    for key in data.files:
        arr = data[key]
        print(f"  {key:<20s}  shape={str(arr.shape):<20s}  dtype={arr.dtype}")

    latents = data["latents"]
    rewards = data["rewards"]
    ep_ret  = data["episode_returns"]
    ep_len  = data["episode_lengths"]
    ep_ids  = data["episode_ids"]

    n_episodes = ep_ret.shape[0]
    n_steps    = latents.shape[0]

    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Episodes           : {n_episodes}")
    print(f"  Total steps        : {n_steps}")
    print(f"  Latent dim         : {latents.shape[1]}")
    print(f"  Mean return        : {ep_ret.mean():.3f} ± {ep_ret.std():.3f}")
    print(f"  Success rate       : {(ep_ret > 0).mean():.1%}")
    print(f"  Mean episode len   : {ep_len.mean():.1f} ± {ep_len.std():.1f}")
    print(f"  Latent mean (abs)  : {np.abs(latents).mean():.4f}")
    print(f"  Latent std         : {latents.std():.4f}")

    # Check companion JSON
    meta_path = path.with_name(path.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n── Metadata ({meta_path.name}) ──────────────────────")
        for k, v in meta.items():
            print(f"  {k:<20s}: {v}")

    print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()