# Full Pipeline Results — cosine vs P3 residual_log

Last update: **2026-05-13** (after 60-epoch from-scratch run on stars).

This document summarises **all three deployment-horizon experiments** of the
P3 residual-on-log-cosine policy (`--policy_output residual_log`) against
the cosine baseline:

1. **35 epochs from scratch** (night10h + catchup, 6 seeds each) — design-horizon match
2. **50 epochs via warm-restart** (35 → 50 with abrupt LR jump at e36) — robustness test
3. **60 epochs from scratch** (3 seeds each) — out-of-distribution horizon (policy trained on T=35, deployed on T=60)

---

## 1. Headline numbers (mean ± std, student CIFAR-100 test acc)

| Horizon | Setup | cosine | P3 residual_log | Δ |
|---|---|---:|---:|---:|
| 35 ep | from scratch, 6 seeds | **69.81 ± 0.33** | **69.31 ± 0.43** (6 seeds) | -0.50 |
| 50 ep | warm-restart from e35 (LR 0.001 → 0.21 at e36) | **drop to ~52 %, recover to ~70**; ΔAcc ≈ -18 pp transient | **drop to ~63 %, recover to ~70**; ΔAcc ≈ -7 pp transient | residual_log **2.5× smaller shock** |
| 60 ep | from scratch, 3 seeds | **70.20 ± 0.23** (best) / **68.42 ± 0.35** (last-10 avg) | **69.89 ± 0.17** (best) / **68.47 ± 0.57** (last-10 avg) | best −0.31, smoothed +0.05 |

**Bottom-line takeaway**:
* On the **design horizon (T=35)** the learned policy matches cosine within ~0.5 pp.
* On a **scaled-up horizon (T=60)** the same offline-trained policy still tracks
  cosine within statistical noise (best Δ −0.31 pp, smoothed Δ +0.05 pp) — the
  `[-1, +1]` log-residual head provides a strong identity prior that makes the
  policy robust to OOD deployment horizons it never saw during pretraining.
* On a **warm-restart** scenario the residual_log policy is dramatically more
  robust: it loses **only 7 pp** of student accuracy after the LR shock vs cosine
  losing **18 pp**, and recovers within 5 epochs vs 10 epochs for cosine.

---

## 2. 60-epoch from-scratch run (this update)

* Server: **stars** (A100-80GB, single-GPU shared via 3 parallel processes).
* `tmux full60`, two sequential bundles (3-parallel each):
  - Bundle A (residual_log seeds 1, 2, 3): **8577 s ≈ 2 h 23 min**
  - Bundle B (cosine seeds 1, 2, 3): **8515 s ≈ 2 h 22 min**
  - Total wallclock: **4 h 45 min**
* Settings (both methods identical except scheduler / policy):
  - `cifar100`, `depth_list=110_20`, `batch_size=256`, `student_steps_ratio=4`,
    `alpha=0.5`, `epochs=60`, `lr=1.0`, seeds {1, 2, 3}.
  - residual_log: own offline policy pretraining each run (~30 s, GRU policy
    on `features_v3`, `policy_oracle_period=35`, `policy_window=10`,
    `policy_warmup_epochs=5`, `policy_max_lr=1.5`, `policy_min_lr=0.001`).

### Final-epoch metrics (epoch 60, mean ± std over 3 seeds)

| Method | Student Test Acc | Student Test Loss | Teacher Test Acc | Teacher Test Loss |
|---|---:|---:|---:|---:|
| cosine | **70.20 ± 0.23** | 1.121 ± 0.007 | 76.28 ± 0.23 | 0.833 ± 0.004 |
| residual_log | **69.81 ± 0.21** | 1.134 ± 0.010 | 76.30 ± 0.48 | 0.836 ± 0.009 |

### Per-seed best student test accuracy (any epoch)

| Method | Seed | Best Acc | @epoch |
|---|---:|---:|---:|
| cosine | 1 | 70.00 | 59 |
| cosine | 2 | 70.15 | 60 |
| cosine | 3 | 70.45 | 60 |
| residual_log | 1 | 69.70 | 59 |
| residual_log | 2 | 70.00 | 59 |
| residual_log | 3 | 69.98 | 60 |

### Smoothed final (last-10-epoch average)

| Method | Mean Acc (last 10 ep) | Std | n_seeds |
|---|---:|---:|---:|
| cosine | 68.42 | 0.35 | 3 |
| residual_log | **68.47** | 0.57 | 3 |

The smoothed metric is **statistically identical** between the two methods —
the policy successfully generalises from a 35-epoch training horizon to a
60-epoch deployment horizon.

---

## 3. Effective-LR trajectory (60-ep run)

The residual_log policy mostly tracks the cosine reference (its identity prior),
with bounded multiplicative deviations in `[exp(-1), exp(+1)] = [0.37, 2.72]`.
At early epochs (1–6) the policy issues lower LRs than cosine while warming
up; by epoch ~20 it converges onto the cosine trajectory and stays close to
it for the rest of training.

See `results/figures/cosine_vs_residual_log_60ep_lr.png`.

---

## 4. Figures

| File | Description |
|---|---|
| `results/figures/cosine_vs_residual_log.png` | 35-ep run, 6 seeds: test acc/loss |
| `results/figures/cosine_vs_residual_log_50ep.png` | 50-ep warm-restart: full sweep showing recovery |
| `results/figures/cosine_vs_residual_log_50ep_zoom.png` | Zoomed-in view around the e36 shock |
| `results/figures/cosine_vs_residual_log_60ep.png` | 60-ep from-scratch: mean ± std (3 seeds) |
| `results/figures/cosine_vs_residual_log_60ep_lr.png` | 60-ep effective student LR trajectory |

PDF versions of all plots are in the same folder.

---

## 5. Why does P3 residual_log generalise to OOD horizons?

The policy parametrises LR multiplicatively on top of a known schedule:

```
LR(t) = cosine(t, T) * exp(clip(policy_pred(features_t), -1, +1))
```

This has three properties that explain the robustness observed across all
three experiments:

1. **Identity prior** — output 0 means *exact cosine*, so an under-trained
   policy degrades gracefully to the strong cosine baseline rather than
   diverging.
2. **Bounded deviation** — the `[-1, +1]` clip on the log-multiplier caps
   per-step LR variation to ~e× in either direction, preventing destructive
   spikes even when features are OOD.
3. **Schedule-aware** — the policy operates in *residual space*, so its
   knowledge of "how to move LR" is independent of the absolute schedule
   length. Whether T=35 or T=60, the policy still asks the same question:
   "given current landscape + latent state, should LR be a bit higher or a
   bit lower than cosine prescribes?".

This explains why a policy trained on T=35 trajectories transfers cleanly
to T=60 deployment, and why it survives warm-restart shocks that destabilise
the cosine baseline.

---

## 6. SOTA-architecture sweep (advanced_industry, parallel branch)

We also evaluated **4 SOTA time-series architectures** as the residual_log
policy backbone, replacing the default GRU. All 5 architectures (GRU baseline +
4 SOTA) were offline-pretrained on `features_v3` for 80 epochs each (~88 s
total on industry A100), then deployed on the 35-epoch CIFAR-100 distillation
loop (seed 20 for all archs, plus seeds 21, 22 for the winner).

### Phase 1 — offline pretraining (best test MSE)

| Arch | params | best test_mse |
|---|---:|---:|
| **patchtst** | **6 984** | **0.258** |
| tcn | 115 713 | 0.288 |
| gru baseline | 349 332 | 0.307 |
| nbeats | 1 922 679 | 0.321 |
| dlinear | 253 | 0.362 |

PatchTST wins offline with **50× fewer parameters than GRU** and 280× fewer
than NBeats — patch-based attention captures the short-window dynamics of
landscape + latent features more efficiently than recurrent or deep-MLP
stacks. NBeats actually overfits (best epoch 0!) — its 1.9M parameters
memorise training windows but generalise worse.

### Phase 2/3 — deployment on 35-epoch CIFAR-100 (seed 20–22)

| Arch | n_seeds | mean Final Acc | std | Δ vs cosine 69.81 |
|---|---:|---:|---:|---:|
| **patchtst** | 3 | **69.68** | 0.16 | -0.13 |
| tcn | 1 | 69.37 | — | -0.44 |
| residual_log GRU (ref) | 6 | 69.31 | 0.43 | -0.50 |
| nbeats | 0 | crash | — | — |
| dlinear | 0 | crash | — | — |

PatchTST matches cosine within statistical noise (Δ −0.13 pp, vs intra-cosine
σ=0.33) and **outperforms the GRU residual_log baseline by +0.37 pp** with
**50× fewer parameters**. With its tighter seed variance (σ=0.16 vs 0.43 for
GRU), PatchTST is the strongest learnable backbone we have so far.

NBeats and DLinear crashed at deploy epoch 1 due to a fixed-window assumption
(`expected (B,10,230), got (B,2,230)`) — the deployment loop in
`my_research.py` feeds variable-length feature history. Both implementations
need to inherit the GRU's tail-padding logic before they can be evaluated
online (cheap fix, deferred).

### Wallclock and figures

* Phase 1: 88 s for all 5 archs on A100.
* Phase 2 (4 archs × seed20): ~91 min wallclock (only TCN & PatchTST actually
  ran 35 epochs; NBeats/DLinear failed in <2 min).
* Phase 3 (PatchTST seeds 21, 22): ~86 min.
* Total: ~3 h.
* Figures:
  * `results/figures/advanced_archs_offline_mse.png` — offline learning curves
  * `results/figures/advanced_archs_test_acc.png` — deployment acc/loss

---

## 7. Reproduction

```bash
# On stars (5 hours wallclock, A100-80GB):
ssh stars
cd lr-policy
tmux new -s full60 'bash run/run_remote_60ep_stars.sh 2>&1 | tee logs/full60_stars/orchestrator.log'

# Pull + analyse locally:
scp "stars:/home/jovyan/lr-policy/logs/full60_stars/per_run/*.log" results/full60_stars/per_run/
python scripts/analyze_full60.py
```

Artefacts produced by the analysis script:
* `results/full60_stars/per_run_metrics.csv` — every parsed epoch metric
* `results/full60_stars/summary.md` — auto-generated table summary
* `results/figures/cosine_vs_residual_log_60ep.{png,pdf}` — comparison plot
* `results/figures/cosine_vs_residual_log_60ep_lr.{png,pdf}` — LR trajectory

