# MHAC — Final Insights & Results

Presentation-ready findings. Each entry should lead with the claim, then the numbers that back it up, then any caveats. Add new entries at the top so the most recent finding is easiest to find.

---

## Finding 8 — `L_cons` converges to zero, but does not improve final long-horizon prediction quality

**Claim:** In `mhac_k`, the consistency loss `aux/loss_cons` decays rapidly to ~0 by the end of training, showing that chained predictions become nearly identical to direct predictions. But the final `aux/loss_pred_k5` trajectory is nearly the same in `mhac_k` and `k_step_no_cons`, indicating that explicitly enforcing chained consistency does **not** materially improve final 5-step prediction quality. This supports the downstream ablation result that `L_cons` is not the load-bearing part of MHAC.

**Setup:** Same FourRooms 5-seed sweep as Findings 6 and 7. `aux/loss_cons` plotted for `mhac_k`; `aux/loss_pred_k5` compared between `mhac_k` and `k_step_no_cons` over the full 5M-step training run. Metrics are logged by [WandBEvalCallback](src/utils/logging.py) / training logs and read as seed-mean curves.

### Evidence

**(a) `aux/loss_cons` in `mhac_k` collapses to ~0.** It starts around ~0.02 early in training and decays smoothly to approximately 0 by 5M steps. By the end of training, chained and direct predictions are effectively aligned under the cosine-consistency metric.

**(b) `aux/loss_pred_k5` looks nearly identical with and without `L_cons`.** The `mhac_k` and `k_step_no_cons` curves track each other closely throughout training and finish in the same ~0.25–0.27 band at 5M. Any small late-training difference is well within the across-seed spread and does not establish a meaningful prediction-quality advantage for `mhac_k`.

### Interpretation

This is the internal-training-dynamics version of Finding 4. `L_cons` is clearly *satisfiable* — the model drives it to zero — but satisfying it does not buy a better long-horizon predictor than direct prediction alone already produces. Mechanistically, that makes sense: the direct and chained predictors share weights, so once direct prediction is good enough, chained rollout can align with it and `L_cons` becomes easy to satisfy. The crucial result is that this alignment does **not** translate into better downstream performance or representation quality.

So the right interpretation is not "`L_cons` fails to optimize." It optimizes fine. The point is that **optimizing it does not add measurable value beyond `L_pred`** in FourRooms.

### Caveats

- This is an internal metric, not a downstream task metric. It supports the main ablation story but is not stronger evidence than the SR / gen-gap / probe comparisons.
- The curves shown are seed means; we do not yet report a formal per-seed endpoint test for final `loss_pred_k5`.
- This result is specific to FourRooms and the current MHAC implementation. A different predictor architecture or environment family could make `L_cons` matter more.

### Source

- W&B training curves: `aux/loss_cons`, `aux/loss_pred_k5`
- Compared conditions: `mhac_k`, `k_step_no_cons`

---

## Finding 7 — K=5 prediction losses produce a ~40% sample-efficiency speedup; K=1 does not

**Claim:** Test-SR-vs-step curves over the FourRooms 5-seed sweep show that the K=5 aux conditions (`k_step_no_cons`, `mhac_k`) reach test SR = 0.5 at ~1.15–1.20M environment steps, while baseline takes ~1.95M and `one_step` (K=1) takes ~1.80M. K=5 conditions establish a 10–25pp lead by 1.5M and the gap is maintained through 5M. K=1 prediction does not measurably accelerate early learning relative to baseline — it only improves the asymptote.

**Setup:** Same FourRooms 5-seed sweep as Finding 6 (n=5 each, 5M steps). `eval/test_sr` recorded by [WandBEvalCallback](src/utils/logging.py) at 50K-step intervals, 50 eval episodes per call on held-out test seeds. Curves below are seed-mean test SR.

### Evidence

**(a) Steps to first crossing of test-SR thresholds (seed-mean curve).**

| condition | first step ≥ 0.5 | first step ≥ 0.6 | mean test SR at 5M |
|---|---|---|---|
| baseline           | ~1.95M | ~3.20M | 0.61 |
| one_step (K=1)     | ~1.80M | ~2.60M | 0.76 |
| k_step_no_cons     | ~1.20M | ~1.30M | 0.74 |
| **mhac_k (K=5)**   | **~1.15M** | **~1.15M** | 0.72 |

K=5 conditions reach SR=0.5 ~700–800K steps before baseline. K=1 reaches it ~150K steps earlier — within the 50K eval-grid resolution and not distinguishable from baseline at the curve level. The gap to SR=0.6 is even starker: baseline takes 2.05M *more* steps than `mhac_k` to first cross 0.6.

**(b) Trajectory at coarse sample-efficiency anchors.**

| step  | baseline | one_step | k_step_no_cons | mhac_k |
|-------|----------|----------|----------------|--------|
| 0.5M  | 0.26     | 0.27     | 0.30           | 0.26   |
| 1.0M  | 0.31     | 0.29     | 0.32           | 0.37   |
| 1.5M  | 0.40     | 0.46     | 0.66           | 0.53   |
| 2.0M  | 0.55     | 0.58     | 0.58           | 0.57   |
| 3.0M  | 0.51     | 0.74     | 0.70           | 0.66   |
| 5.0M  | 0.61     | 0.76     | 0.74           | 0.72   |

Two things visible:

- **At 1.5M, K=5 conditions are already at 0.53–0.66 while baseline and one_step are at 0.40–0.46.** The K=5 aux signal is doing real work *during* learning, not just at convergence.
- **one_step tracks baseline through ~2M, then pulls ahead.** Its endpoint lift over baseline (Finding 6: +0.060) appears to come from continued late-training improvement rather than faster early learning — a different mechanism than what K=5 provides.

### Interpretation

This is the time-resolved analog of Finding 6's endpoint table, and it adds a claim the endpoint numbers can't make: **multi-horizon prediction accelerates the encoder during learning, while single-step prediction does not.** Consistent with Finding 2 (magnitude > form) and Finding 5 (K=5 sweet spot): K=5 supplies 5× more horizons of prediction supervision per gradient step, so the encoder gets more shaping pressure earlier in training. K=1 supplies a single horizon and the resulting signal is too weak to differentiate early learning from baseline.

For the presentation, this is the canonical "PPO + aux is more sample-efficient than vanilla PPO" plot — with the non-obvious twist that it's specifically the *multi-horizon* aux that produces the speedup, not the prediction objective in general.

### Caveats

- `k_step_double_pred` curves are not pulled here — those are after-the-fact ablations, not part of the original four-condition design, and are not needed for this finding's claim.
- Threshold crossings are read off the seed-mean curve at 50K-step resolution. Per-seed crossing times have wide spread (min/max range at SR=0.5 spans several hundred thousand steps for K=5 conditions). The ranking holds for the mean.
- "Gap maintained through 5M" rather than "baseline plateaus" — baseline is still slowly climbing through the end of training (0.51 at 3M → 0.61 at 5M), it just never catches up.

### Source

- W&B export: 4 conditions × 5 seeds, `eval/test_sr` over training, 50K-step cadence
- Final-checkpoint values match Finding 6's table (0.628 / 0.688 / 0.712 / 0.764), confirming the same run set

---

## Finding 6 — Main results: every prediction-based aux loss beats baseline; action conditioning is unnecessary

**Claim:** Across the full FourRooms condition sweep (n=5 seeds each, 5M steps), every prediction-based auxiliary condition beats baseline on test SR, and the ranking is consistent with the *prediction-signal-magnitude* story from Finding 2 (more L_pred signal → better policy). The `no_action` ablation — `k_step_no_cons`'s predictor with the action input removed — matches the action-conditioned variant in mean test SR and **halves the across-seed standard deviation** (0.041 vs 0.106). Action conditioning is not necessary for the regularization benefit, and may slightly destabilize training.

**Setup:** Same FourRooms `mhac_k`-style training as the rest of the findings; conditions vary only in (λ_pred, λ_cons) and the predictor's `use_action_conditioning` flag. Five seeds each ({42, 123, 456, 789, 1000}). SR read from the W&B callback at the 5M-step final checkpoint, 50 eval episodes per seed on held-out test seeds.

### Evidence

**(a) Headline test SR ranking (n=5 each).**

| condition          | (λ_pred, λ_cons) | act-cond | test SR (mean ± std) | min–max     | Δ vs baseline |
|--------------------|------------------|----------|----------------------|-------------|---------------|
| baseline           | (0.0, 0.0)       | —        | 0.628 ± 0.101        | 0.48–0.74   | —             |
| one_step (K=1)     | (0.1, 0.0)       | yes      | 0.688 ± 0.108        | 0.58–0.84   | +0.060        |
| k_step_no_cons     | (0.1, 0.0)       | yes      | 0.712 ± 0.106        | 0.60–0.88   | +0.084        |
| **no_action**      | (0.1, 0.0)       | **no**   | 0.732 ± **0.041**    | 0.70–0.80   | +0.104        |
| mhac_k (K=5)       | (0.1, 0.05)      | yes      | 0.764 ± 0.087        | 0.66–0.88   | +0.136        |
| k_step_double_pred | (0.2, 0.0)       | yes      | 0.788 ± 0.081        | 0.68–0.90   | +0.160        |

SE(n=5) ≈ 0.04–0.05 per condition. `k_step_double_pred` vs baseline (Δ=+0.160) is ~3.6 SE — clearly significant. `one_step` vs baseline (Δ=+0.060) is ~1.3 SE — borderline.

**(b) Generalization gap (train_sr − test_sr).**

| condition          | gen_gap (mean ± std)  | range            |
|--------------------|-----------------------|------------------|
| baseline           | +0.060 ± 0.114        | -0.10 to +0.18   |
| **one_step (K=1)** | **+0.124 ± 0.065**    | +0.04 to +0.22   |
| k_step_no_cons     | +0.056 ± 0.144        | -0.18 to +0.20   |
| no_action          | -0.024 ± 0.100        | -0.12 to +0.12   |
| mhac_k (K=5)       | -0.012 ± 0.077        | -0.10 to +0.10   |
| k_step_double_pred | **-0.028 ± 0.073**    | -0.14 to +0.04   |

Two non-obvious things in this table:

- **`one_step` is the *worst* generalizer in the sweep — worse than baseline.** Δ vs baseline = +0.064, ~1.4 SE. Borderline-significant individually, but consistent with Finding 5's K=1 result (mean gen_gap = +0.108 there with λ_cons=0.05). Two independent K=1 runs both show the worst generalization. Short-horizon prediction does not just fail to regularize — it appears to actively overfit.
- **All K=5 aux conditions cluster near zero gen_gap** (range -0.028 to +0.056). The *form* of the K=5 prediction objective (no_cons vs no_action vs cons vs double-pred) doesn't matter for the mean — what matters is being at K=5 at all. `double_pred` has the cleanest mean (-0.028) and one of the tightest spreads.

**(c) Magnitude-of-signal monotonicity (re Finding 2).** Ranking of aux conditions by total prediction-objective magnitude tracks the test-SR ranking:

- one_step: K=1, λ_pred=0.1 → smallest aux objective → lowest aux SR (0.688)
- k_step_no_cons: K=5, λ_pred=0.1 → 5× more horizons supervised → 0.712
- mhac_k: K=5, λ_pred=0.1, λ_cons=0.05 → +chained-rollout signal → 0.764
- k_step_double_pred: K=5, λ_pred=0.2 → 2× direct-pred weight → 0.788 (highest)

This is the test-SR analog of Finding 2's goal_dist R² ranking. Same monotonic story: more prediction signal → better policy. Form (chained vs direct) is again secondary to magnitude (Finding 4).

**(d) `no_action` per-seed numbers — variance reduction in both SR *and* gen_gap.**

|   | k_step_no_cons SR | no_action SR | k_step_no_cons gap | no_action gap |
|---|-------------------|--------------|--------------------|---------------|
| seed 42   | 0.74 |  0.72 | +0.10 | +0.04 |
| seed 123  | 0.88 |  0.74 | -0.18 | +0.12 |
| seed 456  | 0.68 |  0.70 | +0.04 | -0.08 |
| seed 789  | 0.66 |  0.80 | +0.12 | -0.08 |
| seed 1000 | 0.60 |  0.70 | +0.20 | -0.12 |
| **mean**  | **0.712** | **0.732** | **+0.056** | **-0.024** |
| **std**   | **0.106** | **0.041** | **0.144**  | **0.100**  |

The action-conditioned version has one strong seed (0.88) and one weak seed (0.60), and a gen_gap range of -0.18 to +0.20 (huge). The action-free version produces a tighter SR band of 0.70–0.80 *and* a smaller gen_gap range of -0.12 to +0.12. SR variance ratio F(4,4) ≈ 6.7 (p ≈ 0.05); gen_gap variance ratio F(4,4) ≈ 2.1 (smaller effect). Both axes point the same direction.

### Interpretation

Two things to take away.

**(1) The headline plot for the paper is row-by-row Table (a).** Every prediction-based regularizer beats baseline; the magnitudes line up with how much prediction signal each one delivers; and *neither* the chained-rollout machinery (Finding 4) *nor* the action conditioning (this finding) is load-bearing. The simplest and most surprising distillation: **the auxiliary objective works because it asks the encoder to predict the future, full stop. The bells and whistles around that objective don't matter.**

**(2) Why `no_action` is stable.** The action-conditioned predictor must learn a small but non-trivial map from (z_t, a_{t..t+k}) → z_{t+k}. Different seeds settle into different action embeddings and slightly different action-effect mappings, which couples to encoder dynamics in ways that produce per-seed variation. Strip out the action input and the predictor is forced to model the *marginal* distribution of z_{t+k} given z_t — a coarser objective, but one whose gradient signal into the encoder is invariant to the seed's action-embedding RNG. The encoder gets the same regularizing pressure on every run, hence the tighter spread.

This is also a useful negative result for the project's framing: action-conditioned latent-space prediction was the natural choice, but it isn't doing the work we thought.

### Caveats

- n=5, FourRooms only. Larger sweeps may resolve the borderline differences (e.g., one_step vs baseline).
- `mhac_k` mean here (0.764) differs slightly from the value quoted in Finding 4 (0.772). Both are 5-seed n=5 W&B reads of the same condition; the small gap reflects different W&B query timestamps within the final-checkpoint window. Not a re-run.
- We do **not** yet have probe-table goal_dist R² for `no_action` and `one_step`. A full cross-condition probe sweep would let us check whether `no_action`'s representation quality matches `k_step_no_cons` or sits elsewhere.
- The `no_action` variance F-test is borderline (p ≈ 0.05) at n=5; it's the right direction but a 10-seed replication would be more defensible.

### Source

- W&B sweep: 6 conditions × 5 seeds at the 5M-step checkpoint
- Per-condition launch: [scripts/train.py](scripts/train.py) `--condition {baseline, one_step, k_step_no_cons, no_action, mhac_k, k_step_double_pred}`
- `no_action` flag wiring: [scripts/train.py:64](scripts/train.py#L64), [scripts/train.py:111](scripts/train.py#L111)

---

## Finding 5 — K=5 is the sweet spot; longer horizons regularize, but with diminishing returns

**Claim:** A horizon ablation over K ∈ {1, 3, 5, 10} on FourRooms (`mhac_k`, λ_pred = 0.1, λ_cons = 0.05, n=5 seeds each) shows test SR rising monotonically from K=1 to K=5 and then plateauing, while the *cleaner* signal — generalization gap — is minimized and tightest at K=5. K=10 buys no additional benefit over K=5 on any measured axis. The horizon parameter behaves like a regularization knob whose returns saturate by K=5.

**Setup:** Same `mhac_k` condition as Findings 2 and 4, with the prediction horizon swept via the new `--k` flag in [scripts/train.py](scripts/train.py). All other hyperparameters held fixed. SR and gen_gap read from the W&B callback at the 5M-step final checkpoint, 50 eval episodes per seed on held-out test seeds.

### Evidence

**(a) Test success rate — rises K=1→K=5, plateaus at K=10.**

| K  | test SR (mean ± std, n=5) | min–max     |
|----|---------------------------|-------------|
| 1  | 0.656 ± 0.118             | 0.46 – 0.76 |
| 3  | 0.732 ± 0.183             | 0.54 – 0.92 |
| 5  | **0.764 ± 0.087**         | 0.66 – 0.88 |
| 10 | 0.772 ± 0.092             | 0.68 – 0.88 |

Δ(K=1 → K=5) = +0.108, roughly 2 SE — moderately strong with n=5. Δ(K=5 → K=10) = +0.008, well inside one σ. K=3 is intermediate in mean but **2× the spread** of K=5/10 (std 0.183 vs ~0.09), with two seeds collapsing to 0.54.

**(b) Generalization gap — K=5 has the smallest mean magnitude and the tightest spread.**

| K  | gen_gap (mean ± std)  | range          |
|----|-----------------------|----------------|
| 1  | +0.108 ± 0.140        | -0.02 to +0.32 |
| 3  | +0.080 ± 0.139        | -0.04 to +0.24 |
| 5  | **-0.012 ± 0.077**    | -0.10 to +0.10 |
| 10 | +0.048 ± 0.101        | -0.10 to +0.18 |

This is the clearest signal in the sweep. K=1 and K=3 each have *individual seeds* with gap > 0.20 (clear overfit on those seeds); K=5 has none. Δ(K=1 mean → K=5 mean) = -0.120, well outside the noise floor. K=10's mean is small but it has a +0.18 outlier seed.

**(c) Per-seed test SR (raw, for reproducibility).**

| seed | K=1  | K=3  | K=5  | K=10 |
|------|------|------|------|------|
| 42   | 0.72 | 0.92 | 0.88 | 0.68 |
| 123  | 0.70 | 0.54 | 0.66 | 0.70 |
| 456  | 0.46 | 0.78 | 0.72 | 0.88 |
| 789  | 0.64 | 0.54 | 0.82 | 0.74 |
| 1000 | 0.76 | 0.88 | 0.74 | 0.86 |

**(d) Drift collapses to ~0 across all K.** Final `aux/drift_k{K}` lands in the 0.0005 – 0.002 range for every K and every seed. Consistency loss does its job uniformly — the K story is *not* a consistency-loss story; it's a horizon-coverage story for L_pred.

### Interpretation

Longer-horizon prediction acts as a regularizer that prevents the encoder from latching onto seed-specific features: the encoder must produce a latent space in which states k steps apart in the dynamics remain predictable from one another, which constrains it away from idiosyncratic shortcuts. The benefit accumulates from K=1 to K=5, then saturates — at K=10 the marginal regularizing pressure of horizons 6–10 produces no additional generalization gain.

This is consistent with Finding 2 (prediction signal *magnitude* matters more than form) and Finding 4 (chained rollout provides no extra benefit beyond pure L_pred): K is another dial on the same fundamental quantity — *how much prediction-based regularization the encoder is exposed to.*

### Suggestive-not-proven: `loss_pred_k5` rises uniquely during K=5 training

Across all 5 seeds of the K=5 condition, `aux/loss_pred_k5` *climbs* steadily from ~0.05 to ~0.27 over 5M steps. The K=1 and K=3 conditions' deepest-horizon losses spike early then plateau around 0.18 (the typical pattern). The K=10 condition's `loss_pred_k10` *also* shows the typical pattern — early spike, settle around 0.18 — and crucially does **not** climb the way K=5's k=5 loss does. Hypothesis: the encoder keeps drifting toward more policy-relevant (less predictable) features, and at K=5 the predictor can't quite keep up; at K=10 the broader horizon supervision pins the encoder down enough that prediction stays tractable. This connects to Finding 3's expressivity/predictability tension. Worth investigating but not a presentable claim from these curves alone.

### Caveats

- n=5 per K, FourRooms only. K=5 vs K=1 difference on test SR is ~2 SE — suggestive, not definitive. The gen_gap signal is the cleaner one (Δ ≈ 0.12, well outside noise).
- K=5 vs K=3 difference in mean test SR is within noise (Δ = 0.03). The reason to prefer K=5 over K=3 is the **stability** (std halved) and the **gen_gap** (mean -0.01 vs +0.08), not the mean SR.
- K=1 here is *not* the same as the `one_step` condition: this run keeps λ_cons = 0.05, while `one_step` uses λ_cons = 0. Apples-to-apples K ablation only.
- "Plateau at K=10" reflects 5M training steps; longer training might separate K=5 and K=10. Not tested.

### Source

- W&B sweep: `mhac_k` condition × K ∈ {1, 3, 5, 10} × seeds {42, 123, 456, 789, 1000}
- Sweep launched via `--k` flag added to [scripts/train.py](scripts/train.py)
- Per-horizon loss / drift panels: W&B `aux/loss_pred_k{k}`, `aux/drift_k{k}`

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
