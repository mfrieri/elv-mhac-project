"""
Analyze saved latents across conditions/seeds.

Runs two probes per NPZ:
  1. Effective rank of the latent space (entropy of normalized singular values).
  2. Linear probes for state variables that the encoder *should* capture:
       - agent_x, agent_y, goal_dist   (linear regression, 5-fold CV R²)
       - agent_dir                     (logistic regression, 5-fold CV accuracy)

Aggregates per-seed results by condition (parsed from filename) and prints two
markdown tables: per-run rows and condition-level mean±std.

Usage:
    python scripts/analyze_latents.py latents/fourrooms_*.npz
    python scripts/analyze_latents.py latents/probe/*.npz --md results/probes.md
"""

import argparse
import glob
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cosmetic numerical warnings from BLAS when many latent dims are
# ReLU-dead; results remain correct (verified via reasonable R² / acc).
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def effective_rank(z: np.ndarray, eps: float = 1e-12) -> float:
    """
    exp(H(p)) where p_i = σ_i² / Σ σ_j².  Measures how many latent directions
    carry meaningful variance.  Equal to the latent_dim when variance is
    uniform, collapses toward 1 under full representation collapse.
    """
    z_c = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(z_c, compute_uv=False)
    p = s ** 2
    p = p / max(p.sum(), eps)
    p = p[p > eps]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def _make_preprocessor():
    # Drop ReLU-dead / near-constant dims, then standardize. Keeps BLAS
    # numerically stable and avoids StandardScaler blowing up small std's.
    return [
        ("vt", VarianceThreshold(threshold=1e-8)),
        ("scale", StandardScaler()),
    ]


def _probe_regression(z, y, n_splits=5):
    pipe = Pipeline(_make_preprocessor() + [("reg", Ridge(alpha=1.0))])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, z, y, cv=cv, scoring="r2")
    return float(scores.mean())


def _probe_classification(z, y, n_splits=5):
    pipe = Pipeline(_make_preprocessor() + [("clf", LogisticRegression(max_iter=2000))])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, z, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def analyze_one(npz_path: str) -> dict:
    data = np.load(npz_path)
    z = data["latents"]
    ax = data["agent_x"]
    ay = data["agent_y"]
    ad = data["agent_dir"]
    gd = data["goal_dist"]

    # Drop steps where goal wasn't found (goal_dist = -1)
    valid = gd >= 0
    z_v, ax_v, ay_v, ad_v, gd_v = z[valid], ax[valid], ay[valid], ad[valid], gd[valid]

    return {
        "n_steps": int(valid.sum()),
        "eff_rank": effective_rank(z_v),
        "r2_x":    _probe_regression(z_v, ax_v),
        "r2_y":    _probe_regression(z_v, ay_v),
        "acc_dir": _probe_classification(z_v, ad_v),
        "r2_gd":   _probe_regression(z_v, gd_v),
    }


# ---------------------------------------------------------------------------
# Name parsing + aggregation
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(
    r"(?P<env>[a-z]+)_(?P<cond>.+?)_seed(?P<seed>\d+)_latents"
)


def parse_run_name(path: str):
    """
    Extract (env, condition, seed) from an NPZ filename like
    'fourrooms_mhac_k_seed42_latents.npz'.  Returns (None, stem, None) on miss.
    """
    stem = Path(path).stem
    m = _NAME_RE.match(stem)
    if m:
        return m.group("env"), m.group("cond"), int(m.group("seed"))
    return None, stem, None


def _fmt_table(header, rows):
    sep = "|" + "|".join(["-" * max(3, len(h.strip())) for h in header.strip("|").split("|")]) + "|"
    return "\n".join([header, sep, *rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="NPZ files or glob patterns")
    ap.add_argument("--md", help="Also write the markdown output to this file")
    args = ap.parse_args()

    # Expand any globs that weren't expanded by the shell
    files = []
    for p in args.paths:
        if any(c in p for c in "*?["):
            files.extend(glob.glob(p))
        else:
            files.append(p)
    files = sorted(set(files))
    if not files:
        print("No NPZ files matched.")
        return

    rows = []
    for f in files:
        env, cond, seed = parse_run_name(f)
        print(f"Analyzing {f}  (cond={cond}, seed={seed})")
        metrics = analyze_one(f)
        rows.append({"env": env, "cond": cond, "seed": seed, "path": f, **metrics})

    # ---------- Per-run table ----------
    per_run_hdr = "| condition | seed |   T  | eff_rank |  R² x  |  R² y  | acc dir | R² goal_dist |"
    per_run = []
    for r in sorted(rows, key=lambda r: (r["cond"] or "", r["seed"] or 0)):
        per_run.append(
            f"| {r['cond']} | {r['seed']} | {r['n_steps']} | "
            f"{r['eff_rank']:7.2f} | {r['r2_x']:6.3f} | {r['r2_y']:6.3f} | "
            f"{r['acc_dir']:7.3f} | {r['r2_gd']:10.3f} |"
        )
    per_run_md = _fmt_table(per_run_hdr, per_run)

    # ---------- Aggregated table ----------
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["cond"]].append(r)

    def mstd(runs, key):
        vs = np.array([r[key] for r in runs], dtype=float)
        return vs.mean(), vs.std(ddof=0)

    agg_hdr = "| condition | n | eff_rank        | R² x            | R² y            | acc dir         | R² goal_dist   |"
    agg = []
    for cond in sorted(by_cond):
        runs = by_cond[cond]
        er_m, er_s = mstd(runs, "eff_rank")
        x_m,  x_s  = mstd(runs, "r2_x")
        y_m,  y_s  = mstd(runs, "r2_y")
        d_m,  d_s  = mstd(runs, "acc_dir")
        g_m,  g_s  = mstd(runs, "r2_gd")
        agg.append(
            f"| {cond} | {len(runs)} | "
            f"{er_m:5.2f} ± {er_s:4.2f} | "
            f"{x_m:5.3f} ± {x_s:4.3f} | "
            f"{y_m:5.3f} ± {y_s:4.3f} | "
            f"{d_m:5.3f} ± {d_s:4.3f} | "
            f"{g_m:5.3f} ± {g_s:4.3f} |"
        )
    agg_md = _fmt_table(agg_hdr, agg)

    out = (
        "\n## Per-run results\n\n" + per_run_md +
        "\n\n## Aggregated by condition (mean ± std across seeds)\n\n" + agg_md + "\n"
    )
    print(out)

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.md, "w") as fh:
            fh.write(out)
        print(f"\nWrote → {args.md}")


if __name__ == "__main__":
    main()
