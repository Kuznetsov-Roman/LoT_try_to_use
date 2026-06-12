# Analysis report — improve5h PatchTST output-ensemble sweep — 2026-05-14T18:25Z

## TL;DR

- **Verdict: FAIL.** None of the 12 variants beats either reference. Best run reaches **70.29** student test acc (`outens_blend0800_warm05_seed3_60ep`, stars), which is **−0.16 vs cosine seed3 (70.45)** and **−0.41 vs previous best (70.70)**.
- **Root cause: the output ensemble has collapsed `policy_pred` to a near-constant bias.** Per-ensemble means are −0.264 (top5), −0.221 (top3), −0.094 (top7); per-run std over epochs 10–55 is only ≈0.003 — *three orders of magnitude* smaller than the absolute value. The "policy" is functionally equivalent to a fixed LR shrinkage of cosine.
- **The signal across the sweep is just "how much cosine is left after blending"**: Pearson r between `effective_lr / cosine_lr` and best acc is **+0.385 (n=12)**. All 12 runs are simply different multiplicative attenuations of cosine in the band 0.94–0.98.
- **Within-sweep spread is 0.76 pp (69.53 → 70.29)** while reported intra-seed σ ≈ 0.23 pp; the entire sweep is barely 3σ wide and cannot statistically separate variants from each other or from cosine with n=1 seed per cell.
- **Top recommendation:** spend the next 1–2 GPU-h on (a) reproducing prev_best 70.70 to bound seed noise and (b) using a single-checkpoint policy (no ensemble averaging) to restore state-dependent variance, before any further hyperparameter sweeps along this axis.

---

## Pre-registered hypotheses (paraphrased from `USER_CONSTRAINTS`)

- **H1**: at least one of the 12 improve5h variants beats **cosine seed3 = 70.45** (student test acc, same seed, same horizon). → **FAIL** (best = 70.29, 0/12 above 70.45).
- **H2**: at least one variant beats **previous_best PatchTST output-ensemble blend075 = 70.70**. → **FAIL** (best = 70.29, 0/12 above 70.70).
- **RQ1**: which knobs (`blend`, `warmup`, `cosine_after`, `n_members`) move the needle? → only `blend` matters, and only because higher blend means "more cosine" (see root cause).

---

## Main results table (sorted by best student test acc)

| Rank | Run | Server | n_mem | blend | warm | cos_after | Best Acc | @ep | Final Acc | last10 mean | Δ vs cos(70.45) | Δ vs prev(70.70) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | blend0800_warm05 | stars | 5 | 0.800 | 5 | 0 | **70.29** | 60 | 70.29 | 68.42 | **−0.16** | −0.41 |
| 2 | top7_blend0800_cosafter45 | industry | 7 | 0.800 | 10 | 45 | 70.11 | 60 | 70.11 | 68.59 | −0.34 | −0.59 |
| 3 | top7_blend0750_warm10 | industry | 7 | 0.750 | 10 | 0 | 70.04 | 60 | 70.04 | 68.13 | −0.41 | −0.66 |
| 4 | top5_blend0825_warm10 | industry | 5 | 0.825 | 10 | 0 | 69.99 | 58 | 69.86 | 68.61 | −0.46 | −0.71 |
| 5 | blend0725_warm10 | stars | 5 | 0.725 | 10 | 0 | 69.98 | 59 | 69.96 | 68.44 | −0.47 | −0.72 |
| 6 | blend0750_warm10_cosafter45 | stars | 5 | 0.750 | 10 | 45 | 69.86 | 59 | 69.78 | 68.22 | −0.59 | −0.84 |
| 6 | top5_blend0775_cosafter50 | industry | 5 | 0.775 | 10 | 50 | 69.86 | 60 | 69.86 | 68.48 | −0.59 | −0.84 |
| 8 | blend0775_warm10 | stars | 5 | 0.775 | 10 | 0 | 69.69 | 60 | 69.69 | 68.01 | −0.76 | −1.01 |
| 9 | blend0800_warm10_cosafter45 | stars | 5 | 0.800 | 10 | 45 | 69.62 | 60 | 69.62 | 67.85 | −0.83 | −1.08 |
| 10 | top3_blend0800_cosafter50 | industry | 3 | 0.800 | 10 | 50 | 69.60 | 60 | 69.60 | 67.77 | −0.85 | −1.10 |
| 11 | top3_blend0750_warm10 | industry | 3 | 0.750 | 10 | 0 | 69.57 | 60 | 69.57 | 67.95 | −0.88 | −1.13 |
| 12 | blend0750_warm05 | stars | 5 | 0.750 | 5 | 0 | 69.53 | 60 | 69.53 | 68.16 | −0.92 | −1.17 |

Notes:
- All 12 runs completed cleanly (`exit=0`), 60 epochs, ≈ 2 h 22 min each per side, n=1 seed (seed 3) per cell.
- Best epoch ≈ final epoch in 10 / 12 runs → no late overfitting; the policy is well-behaved, just unhelpful.
- The previously-best config (`top5, blend=0.75, warm=10, no cosafter`) was **not** in this sweep; the closest neighbour `blend0775_warm10` lands at 69.69 (Δ = −1.01 vs 70.70). With seed σ ≈ 0.23, the 70.70 datum is ≥ 3σ above this — suggestive that 70.70 was a lucky single-seed outlier (see Action 3 below).

---

## Diagnostic plots

All plots saved under `d:/lr-policy/results/improve5h_combined/analysis/plots/`.

- `plots/best_acc_bars.png` — best student acc per run, sorted, with cosine + prev-best reference lines. **What to look for:** the entire bar cloud sits below both reference lines, with the top variant grazing the cosine line.
- `plots/student_acc_curves.png` — full 60-epoch student-acc curves, faceted by server, with reference lines. **What to look for:** all 12 curves overlap heavily in the final 15 epochs and pass below the cosine reference.
- `plots/student_acc_last15.png` — zoomed last-15-epoch curves. **What to look for:** the rank ordering is mostly settled by epoch 50; small terminal differences are well within seed noise band.
- `plots/blend_vs_acc.png` — best acc vs cosine blend, coloured by member count. **What to look for:** weak positive trend with blend, no strong monotonicity along member count.
- `plots/eff_lr_vs_acc.png` — best acc vs effective `student_lr / cosine_lr` (state-averaged). **Pearson r = +0.385.** This is the cleanest summary plot: it shows the policy is just modulating an offset around cosine.
- `plots/policy_pred_diagnosis.png` — `policy_pred` trajectories and effective-LR multiplier vs epoch. **Key diagnosis plot.** The left panel shows that all curves are flat lines clustered by member count; the right panel shows effective LR sits in a tight band 0.94–0.98 × cosine across all 60 epochs.
- `plots/student_lr_trajectory.png` — actual student LR over training (log scale), 12 runs. **What to look for:** the curves are visually indistinguishable from cosine.
- `plots/generalisation_gap.png` — train − test gap at final epoch. **What to look for:** gap is uniform (~21 pp) — there is no overfitting/underfitting story; the result is genuinely about LR scaling not regularisation.

---

## Root cause

The **output ensemble has destroyed the state-dependent variance of `policy_pred`**, leaving only an ensemble-specific constant bias.

Per-run statistics on `policy_pred` over epochs 10–55 (the stable phase, post-warmup, pre-tail):

| Ensemble | Mean policy_pred | Std policy_pred | `exp(mean)` | Effective LR (state-avg) blend=0.75 / 0.80 |
|---|---:|---:|---:|---:|
| top3 | **−0.221** | 0.005 | 0.802 | 0.950 / 0.960 |
| top5 | **−0.264** | 0.003 | 0.768 | 0.942 / 0.954 |
| top7 | **−0.094** | 0.003 | 0.910 | 0.978 / 0.982 |

Three consequences:

1. **The policy adds no per-step adaptivity.** `policy_pred` std-over-epochs is **~0.003**, i.e. the per-epoch correction varies by ±0.3% around the constant. The learned policy_lr is effectively `cosine_lr × const`.
2. **The constant differs by ensemble membership in a non-monotonic way.** top5 has the *most negative* bias (−0.264), top7 has the *least* (−0.094) — adding two more checkpoints (seed3 + seed6) doesn't average smoothly with the existing 5, it shifts the bias. This is a strong signal that individual PatchTST seeds disagree on the bias and that "averaging" is dominated by which checkpoints landed in the subset, not by denoising.
3. **The entire 12-run cloud is just different attenuations of cosine.** Effective LR ratios span 0.94–0.98 and Pearson r(eff_ratio, best_acc) = +0.385. The other knobs (`warmup`, `cosine_after`) move the ratio second-order but cannot escape this band given the constant `policy_pred`.

Evidence pointers:
- `policy_pred` per-run stats: `d:/lr-policy/results/improve5h_combined/analysis/series.json` (key `policy_pred`) — values pulled in `correlate.py` lines 17–28.
- Per-run log lines, e.g. `outens_blend0800_warm05_seed3_60ep.log:241` `policy_pred=-0.263015`; `outens_top7_blend0750_warm10_seed3_60ep.log` middle section shows policy_pred ≈ −0.094 throughout.
- Policy → LR transform in code: `reviewer_artifacts/patchtst_output_ensemble_blend075/README.md:30-35` documents `policy_lr = cosine_lr * exp(clip(mean(policy_outputs), -1, 1))` then blend.

Statistical caveats:
- n=1 seed per cell → no within-cell variance estimate. Reported intra-seed σ ≈ 0.23 (cosine 60-ep, RESULTS_full_pipeline.md:53). The 0.76 pp spread of the sweep is barely 3σ wide.
- Best − cosine = −0.16 ≈ 0.7σ → not significant in any sense.
- Prev_best − best_of_sweep = +0.41 ≈ 1.8σ → not significant; consistent with seed luck.

---

## Ranked action list

All costs are A100-80GB, single-process, ≈ 40 min for 60 epochs (matches the actuals: 8487 s ÷ 3 parallel = 47 min/run wallclock; ≈ 40 min serial GPU time per run since they share the same A100 in the actual sweep).

### Action 1 — De-bias the ensemble or shrink it to size 1 *(highest information, 1.3 GPU-h)*

**Hypothesis.** The `policy_pred` collapse is caused by averaging many heterogeneous PatchTST seeds. A single-checkpoint policy will retain state-dependent variance (per-epoch std ≫ 0.003), and a recentred multi-ensemble (subtract per-batch mean) will too.

**Falsification.** Run two probes with seed3, 60 ep, blend=0.75, warm=10, cosafter=0:
1. `policy_checkpoint=<best single PatchTST seed by offline MSE>` (n=1, no averaging).
2. Same 5-checkpoint ensemble but apply `policy_pred ← policy_pred − running_mean(policy_pred)` (zero-mean residual).

If neither probe lifts best acc above **70.5**, the policy contributes essentially zero useful state-dependent signal at T=60; abandon the output-ensemble axis and pivot. If one of them reaches **≥ 70.7**, we have evidence that ensemble bias was the bottleneck and the next sweep should explore single-seed PatchTST + blend.

**Cost.** 2 × 40 min = **1.3 GPU-h**.

**Implementation pointer.** Policy ensembling lives in `reviewer_artifacts/patchtst_output_ensemble_blend075/code/model/lr_policy.py` (`OutputEnsemblePolicy.forward` returns the mean). Add a `--policy_zero_mean_residual` flag that subtracts a per-run EMA of `policy_pred` before the exp. For probe 1, just pass a single checkpoint to `--policy_checkpoint`.

### Action 2 — Replicate previous best 70.70 *(seed-noise bound, 1.3 GPU-h)*

**Hypothesis.** Prev_best 70.70 was a lucky single-seed sample of an underlying distribution with mean ≈ 69.9–70.1. The 0.41 pp gap to the current best (70.29) is below 2σ.

**Falsification.** Re-run the exact prev_best config (`top5, blend=0.75, warm=10, no cosafter, seed3, 60ep`) — note that this config is **missing from this sweep**; the closest was `blend0775` (69.69). Run it twice on industry (it's the same code path).

- If `mean({70.70, rep1, rep2}) ≤ 70.30`: 70.70 is an outlier; we have a tied-with-cosine ensemble policy; report a clean negative result for output-ensemble; verdict locks to FAIL/ABANDON.
- If `mean ≥ 70.50`: WEAK improvement is real but small; pivot to a multi-seed validation (Action 3 below).

**Cost.** 2 × 40 min = **1.3 GPU-h**.

**Implementation pointer.** Submit via the existing `run/run_remote_improve5h_*.sh` machinery with two new launch entries: `("outens_blend0750_warm10_seed3_60ep_rep1", "${TOP5_CKPTS}", 0.750, 10, 0)` and `_rep2`. Optionally vary `--seed` to `30`, `31` if cudnn is non-deterministic and you want true noise, or keep seed=3 to test cudnn-only variability.

### Action 3 — Multi-seed validation of top-3 variants vs cosine *(2.7 GPU-h, deferred unless Actions 1–2 are mixed)*

**Hypothesis.** Even if no variant beats prev_best on seed 3, one may have a better mean across seeds. Currently we have **0** variants benchmarked on > 1 seed; cosine is benchmarked on 3 seeds (RESULTS_full_pipeline.md:60-65).

**Falsification.** Pick the top 1 variant `blend0800_warm05` (70.29) and rerun it on seeds 1 and 2. Compute paired Wilcoxon vs cosine seeds 1, 2, 3 (means 70.00, 70.15, 70.45).

- If `mean_blend0800_warm05_acc(seeds 1,2,3) ≥ cosine_mean (70.20) − 0.05`: weakly positive; consider sample size n=6 for the final write-up.
- Otherwise: FAIL on multi-seed, ABANDON output-ensemble axis, pivot to architecture changes (different policy backbone, different feature set).

**Cost.** 2 × 40 min = **1.3 GPU-h** if Action 2 is run with seeds 1, 2 (then it doubles as Action 3 data).

**Implementation pointer.** Reuse the same launch script, change `--seed` to 1 and 2.

---

## What I checked but didn't find

- **Late degradation / overfitting.** No: in 10/12 runs best epoch = final epoch; remaining 2 are at ep58/59 (within 0.2 pp of final). Train-test gap is constant ~21 pp across all 12 runs (`plots/generalisation_gap.png`) — *not* a regularisation problem.
- **σ-collapse / NLL pathology.** N/A — this is a deterministic LR-policy task, not a probabilistic head.
- **Sampler / data-leak bugs.** Train/test split is the standard CIFAR-100 50k/10k; runs all show identical per-epoch wallclock (≈137 s train, ≈2.5 s eval), no early-exit or NaN.
- **Late-handoff (`cosine_after`) effects.** `cosine_after=45/50` neither helps nor hurts consistently (rank 2, 6, 9, 10). Once the policy is just constant attenuation, switching to pure cosine in the last 10 epochs only matters insofar as it cancels the attenuation in that tail, which is a tiny effect.
- **Warmup length.** Warm=5 wins at blend=0.80 (rank 1) but loses at blend=0.75 (rank 12). With constant `policy_pred`, the only effect of `warmup` is when the policy starts applying its (small) attenuation; the interaction with blend is unsurprising and not a real signal.

---

## Single-line summary for parent agent

**FAIL: 0/12 improve5h variants beat cosine seed3 (70.45) or prev_best (70.70); best is 70.29; root cause is `policy_pred` collapsing to a constant bias from output-ensemble averaging — next 1.3 GPU-h should run a single-checkpoint policy + a 2-seed replication of prev_best 70.70 before any further sweeping.**
