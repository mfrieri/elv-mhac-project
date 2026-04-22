# MHAC — Final Insights & Results

Presentation-ready findings. Each entry should lead with the claim, then the numbers that back it up, then any caveats. Add new entries at the top so the most recent finding is easiest to find.

---

## Finding 3 — The decoder is a diagnostic, not a product

**Claim:** Adding a grid decoder (256-d latent → 11-way object-type logits on the 7×7 egocentric window) gave us a qualitative probe of the latent space. It confirmed the encoder preserves the agent's observation, but revealed that the predictor's outputs — though cosine-close to true future latents — sit *off the decoder's training manifold* and decode to neutral/empty grids. The finding is about what cosine-based prediction losses actually constrain, not about whether the decoder "worked."

**Setup:** `GridDecoder` (Linear→ReLU→Linear) trained jointly with `mhac_k` on FourRooms via `--lambda-recon 0.1`, same optimizer as encoder/predictor. Decoder applied post-hoc to both `z_t` (reconstruction) and `z_hat_direct_k` for k=1..5 (prediction) on held-out test-seed rollouts.

### Evidence

**(a) Reconstruction of `z_t` is clean.** `loss_recon` drops to ~0 and decoded grids visually match ground-truth observations (agent, walls, goal all recovered). Encoder does preserve spatial structure.

**(b) Prediction decoding is degenerate.** For k=1..5, `decoder(z_hat_k)` collapses to near-uniform "empty" grids — no agent, no walls, no goal — despite cosine similarity between `z_hat_k` and the true `z_{t+k}` staying around 0.75.

**(c) Mechanism.** Cosine loss constrains *direction* in latent space, not full distributional identity. A predicted latent can be 0.75 cosine-similar to the target while lying in a region the decoder never saw during training. The decoder was trained on encoder outputs only, so predictor outputs are out-of-distribution inputs to it.

### Interpretation

The decoder surfaced a gap that our scalar metrics (`aux/loss_pred_k`, `aux/drift_k`) hid: "close by cosine" ≠ "semantically decodable." This is a property of the loss function, not a bug — switching to pixel-level decoding would not help, because the off-manifold problem is in latent space, upstream of the decoder.

The research payoff is the actionable future-work direction it generates:
- Train the decoder jointly on predictor outputs (decoder-in-the-loop), or
- Replace cosine with an L2/MSE prediction target that constrains full identity, or
- Add a predictor-decode CE term to the loss so predicted latents are pushed onto the decoder's manifold.

### Suggestive-not-proven: expressivity/predictability tension

In the single `mhac_k + recon` run, `aux/loss_pred_k5` rose from ~0.08 → ~0.25 over 5M steps while `aux/loss_recon` fell toward 0. Consistent with the hypothesis that forcing the encoder to preserve fine spatial structure makes the latent harder to extrapolate — but n=1, so this is a direction to investigate, not a result to present. Would need recon-on vs recon-off at matched seeds to confirm.

### Caveat

Decoder trained alongside a single `mhac_k` seed with `N_ENVS=16` and `λ_cons=0.05`. Not an ablation — we did not compare predictor decode quality across conditions. The claim "predicted latents decode to empty grids" is qualitative from visualizations, not quantified by a metric like decoder-CE-on-`z_hat`.

### Source

- Decoder: [src/models/decoder.py](src/models/decoder.py)
- Reconstruction loss: [src/training/losses.py:83](src/training/losses.py#L83)
- Visualization script: [scripts/decode_latents.py](scripts/decode_latents.py)
- Trainer integration: [src/training/ppo_trainer.py:115](src/training/ppo_trainer.py#L115)

---

## Finding 2 — L_cons is a representation regularizer, not a drift controller

**Claim:** The consistency loss (L_cons) improves mhac_k's performance by shaping the encoder to encode task-relevant geometry (distance-to-goal) more linearly — not by suppressing drift between direct and chained predictions.

**Setup:** 3 conditions × 5 seeds on FourRooms, 50 evaluation episodes each, on held-out test seeds. Linear probes (Ridge / LogisticRegression, 5-fold CV) on frozen 256-d latents.

### Evidence

**(a) Effective rank — no collapse across conditions.**

| condition | eff_rank (out of 256) |
|---|---|
| baseline | 7.28 ± 2.34 |
| k_step_no_cons | 7.80 ± 0.44 |
| mhac_k | 7.32 ± 0.78 |

Rules out "L_cons collapses the latent" as an explanation.

**(b) Absolute position not linearly encoded (by anyone).**

| condition | R² agent_x | R² agent_y |
|---|---|---|
| baseline | 0.013 | -0.003 |
| k_step_no_cons | 0.034 | 0.006 |
| mhac_k | -0.002 | 0.026 |

Expected — agent sees only a 7×7 egocentric window, never its absolute coords.

**(c) Direction encoded equally (chance = 0.25).**

| condition | acc_dir |
|---|---|
| baseline | 0.396 ± 0.035 |
| k_step_no_cons | 0.369 ± 0.023 |
| mhac_k | 0.394 ± 0.016 |

Within each other's error bars.

**(d) Distance-to-goal — mhac_k wins cleanly.** *(load-bearing result)*

| condition | R² goal_dist | Δ vs baseline |
|---|---|---|
| baseline | 0.429 ± 0.018 | — |
| k_step_no_cons | 0.455 ± 0.026 | +0.026 |
| **mhac_k** | **0.490 ± 0.031** | **+0.061** |

- Monotonic ordering matches the performance ordering (baseline < k_step_no_cons < mhac_k).
- Δ = 0.061 / pooled σ ≈ 0.025 × √(2/5) → t ≈ 3.8, significant at n=5 (p<0.01).
- Consistent across all 5 seeds — not driven by an outlier.

### Interpretation

The multi-step chained rollout in L_cons constrains the encoder so that states *k dynamics-steps apart* are also close in latent space. For a navigation task, the most efficient geometry satisfying that constraint is one where distance-to-goal becomes a near-linear axis. That organization is directly useful to the policy — which likely drives the ~6pp test SR advantage of mhac_k.

### Caveat

n=5, FourRooms only. MultiRoom-N6 was scratched because baseline scored 0 across all seeds (no signal to compare against).

### Source

- Probe table: [results/probes.md](results/probes.md)
- Probe pipeline: [scripts/save_latents.py](scripts/save_latents.py), [scripts/analyze_latents.py](scripts/analyze_latents.py)

---

## Finding 1 — Drift is not what L_cons controls

**Claim:** On FourRooms, `k_step_no_cons` (λ_cons = 0) shows drift converging to zero just like `mhac_k`. Our original hypothesis — that L_cons is needed to keep chained predictions from drifting — was wrong.

**Why it happens mechanistically:** The direct and chained predictors share weights, and both optimize against the same (stop-gradient) target. L_pred alone is enough to pull the two heads into agreement; L_cons becomes redundant for the drift metric.

### Evidence

Drift curves for `k_step_no_cons` and `mhac_k` both collapse toward 0 over training (W&B `aux/drift_k{k}` panels, n=5 seeds each). See W&B project for plots.

### Caveat

This was the finding that reframed the project away from "L_cons fixes drift" and motivated Finding 2.
