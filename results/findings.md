# MHAC — Final Insights & Results

Presentation-ready findings. Each entry should lead with the claim, then the numbers that back it up, then any caveats. Add new entries at the top so the most recent finding is easiest to find.

---

## Finding 4 — The chained rollout in L_cons provides no measurable benefit over scaling L_pred

**Claim:** A pure-prediction ablation (`k_step_double_pred`, λ_pred = 0.2, λ_cons = 0) matches `mhac_k` across every outcome we measured — representation quality, test success rate, generalization gap, and k=5 prediction loss. The chained-rollout machinery in L_cons does not produce a measurable advantage over simply scaling the direct prediction loss.

**Setup:** Same 5-seed FourRooms sweep as Finding 2, with `k_step_double_pred` added as a confounder check. Probes from post-hoc 50-episode rollouts on held-out test seeds (from `save_latents`). SR / gen_gap / `loss_pred_k5` read from the W&B callback at the 5M-step final checkpoint.

### Evidence

**(a) Representation quality — tied on goal_dist R².** *(from Finding 2, reproduced)*

| condition | goal_dist R² | Δ vs baseline |
|---|---|---|
| baseline | 0.429 ± 0.018 | — |
| k_step_no_cons (λ_pred = 0.1) | 0.455 ± 0.026 | +0.026 |
| k_step_double_pred (λ_pred = 0.2) | 0.483 ± 0.022 | +0.054 |
| mhac_k (λ_pred = 0.1, λ_cons = 0.05) | 0.490 ± 0.031 | +0.061 |

Gap between `k_step_double_pred` and `mhac_k` is 0.007 — well inside one σ.

**(b) Test success rate — tied at the final checkpoint (W&B).**

| condition | test SR (mean ± std, n=5) |
|---|---|
| k_step_double_pred | 78.8% ± 7.2% |
| mhac_k | 77.2% ± 7.4% |

Paired across matched seeds: mean diff = +0.016 in favor of `k_step_double_pred`, paired t ≈ 0.40. Not distinguishable from zero.

**(c) Generalization gap — tied, both near zero.**

| condition | gen_gap (train − test) |
|---|---|
| k_step_double_pred | -0.028 ± 0.065 |
| mhac_k | -0.012 ± 0.069 |

Both average slightly negative (test SR ≳ train SR at measurement time, within noise at n=5 × 50 eval episodes). Neither condition overfits; neither has a visible generalization edge.

**(d) Prediction loss at k=5 — `mhac_k` is actually slightly *better*, despite half the λ_pred weight.**

| condition | loss_pred_k5 at 5M steps |
|---|---|
| k_step_double_pred | 0.260 ± 0.020 |
| mhac_k | 0.253 ± 0.010 |

Small and within noise, but in the opposite direction from naive expectation. Doubling λ_pred does not drive its own loss lower — if anything the chained rollout in `mhac_k` provides additional gradient signal into the shared predictor that slightly helps long-horizon prediction. That help does not translate into a policy advantage (see (b)).

### Interpretation

This ablation is the cleanest test of "does the chained rollout matter, or is it just more prediction signal?" Across four metrics the answer is *it doesn't measurably matter.* L_cons is a more complex way to spend compute on outcomes that pure L_pred at higher weight already achieves.

This is a *simpler-is-better* result and the cleanest paper takeaway:

- L_cons was added to control drift → wrong (Finding 1).
- L_cons was then hypothesized to uniquely shape representation geometry → wrong (revised Finding 2).
- L_cons provides no measurable advantage over pure L_pred at higher weight (this finding).

### What changed from the earlier draft

An earlier draft claimed `k_step_double_pred` beat `mhac_k` by ~8pp on test SR (91.6% vs 83.2%). That was based on the single 50-episode `save_latents` evaluation per seed and was within the noise floor for that sample size. The W&B callback's eval at the 5M-step final checkpoint — also 50 episodes but measured at the canonical step with the same callback RNG across runs — shows the two conditions tied. Deferring to the W&B numbers as the authoritative SR measurement.

### Caveat

- n=5, FourRooms only. λ_pred dose-response still not mapped — a sweep over `{0.05, 0.1, 0.2, 0.4}` would clarify whether the effect saturates, peaks, or keeps improving.
- "Tied" reflects the limit of n=5 × 50-episode eval resolution. A true null cannot be proved; we can only say no difference is observable above noise.

### Source

- Probe table: [results/probes.md](results/probes.md)
- SR / gen_gap / loss_pred_k5: W&B eval at the 5M-step final checkpoint for each seed

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

## Finding 2 — Prediction auxiliary losses regularize task-relevant geometry

**Claim:** Adding *any* prediction-based auxiliary loss (L_pred alone, L_pred + L_cons, or doubled L_pred) makes the encoder's latent space encode task-relevant geometry — distance-to-goal — more linearly. The regularization effect is a property of the prediction signal in general, not of the chained consistency rollout specifically.

**Setup:** 4 conditions × 5 seeds on FourRooms, 50 evaluation episodes each, on held-out test seeds. Linear probes (Ridge / LogisticRegression, 5-fold CV) on frozen 256-d latents from the final checkpoint.

### Evidence

**(a) Effective rank — no collapse across conditions.**

| condition | eff_rank (out of 256) |
|---|---|
| baseline | 7.28 ± 2.34 |
| k_step_no_cons | 7.80 ± 0.44 |
| k_step_double_pred | 7.51 ± 0.16 |
| mhac_k | 7.32 ± 0.78 |

All conditions use ~7–8 directions. Rules out representation collapse as an explanation for any observed differences.

**(b) Absolute position not linearly encoded (by anyone).**

| condition | R² agent_x | R² agent_y |
|---|---|---|
| baseline | 0.013 | -0.003 |
| k_step_no_cons | 0.034 | 0.006 |
| k_step_double_pred | -0.049 | 0.014 |
| mhac_k | -0.002 | 0.026 |

Expected — agent sees only a 7×7 egocentric window, never its absolute coords.

**(c) Direction encoded equally (chance = 0.25).**

| condition | acc_dir |
|---|---|
| baseline | 0.396 ± 0.035 |
| k_step_no_cons | 0.369 ± 0.023 |
| k_step_double_pred | 0.389 ± 0.028 |
| mhac_k | 0.394 ± 0.016 |

Within each other's error bars.

**(d) Distance-to-goal — all prediction-based conditions beat baseline; the chained-rollout variant is not unique.**

| condition | R² goal_dist | Δ vs baseline |
|---|---|---|
| baseline | 0.429 ± 0.018 | — |
| k_step_no_cons (λ_pred = 0.1) | 0.455 ± 0.026 | +0.026 |
| k_step_double_pred (λ_pred = 0.2) | 0.483 ± 0.022 | +0.054 |
| mhac_k (λ_pred = 0.1, λ_cons = 0.05) | 0.490 ± 0.031 | +0.061 |

Key observations:
- Every prediction-based condition beats baseline on goal-distance decodability.
- `k_step_double_pred` and `mhac_k` are statistically indistinguishable (Δ = 0.007, <<1 σ). The chained rollout does *not* provide a representation-quality advantage over simply scaling L_pred.
- More prediction signal → better linear decodability, whether the extra signal comes from L_cons's chained rollout or from a doubled direct loss weight.

### Interpretation

Prediction auxiliary losses constrain the encoder such that states reachable from each other in a small number of dynamics steps are near each other in latent space. For a navigation task, the most efficient geometry satisfying that constraint is one where distance-to-goal becomes a near-linear axis — which is what the probes find. Direction of the prediction signal (L_pred direct vs L_cons chained) is a secondary choice; magnitude matters more than form.

This overturns our earlier hypothesis that the chained rollout in L_cons was the specific mechanism driving representation quality. See Finding 4 for the performance consequences: pure L_pred at higher weight matches mhac_k on *every* measurable axis — representation, task performance, generalization — with none of the chained-rollout machinery.

### Caveat

n=5, FourRooms only. MultiRoom-N6 was scratched because baseline scored 0 across all seeds (no signal to compare against). The λ_pred dose-response isn't yet characterized — `{0.1, 0.2}` are two points, not a curve.

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
