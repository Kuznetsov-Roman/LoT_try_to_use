# Night 10h — Literature Review Hypothesis Sweep — RESULTS

Generated: 2026-05-13T07:48:15

## Setup

Implemented from `results/trainable_lr_literature_review.md`:

- **P6.1 Schedule-Free SGD/AdamW** (`schedulefree.SGDScheduleFree`/`AdamWScheduleFree`,
  winner of MLCommons AlgoPerf 2024 Self-Tuning, no LR schedule needed).
- **P6.2 Hypergradient SGD** (`HypergradScheduler` in `trainer/my_research.py`):
  finite-difference of the existing 30-dim probe at the closest LR_GRID point,
  multiplicative log-LR step. No offline training, no oracle, no policy.
- **P2 AdaLRS-style veto** (`--policy_veto_mode adalrs --policy_veto_clamp 0.7`):
  if test loss has been monotonically rising for 2 consecutive epochs *and*
  the policy chose a LR > 1.05 × cosine_lr, override LR to 0.7 × cosine_lr.
  Targets the late-training argmin saturation observed in `2h_curve/RESULTS.md`.
- **P3 residual-on-log-cosine head** (`--policy_output residual_log`):
  GRU output ∈ [-1, +1] decoded as `LR = cosine · exp(out)`. Naturally bounded
  identity-init at output=0; pretrained online via `lr_policy_training` from npy.
- **P5** (extra features), **P1** (val-acc forecast head), **P4** (bandit reformulation),
  **P7** (per-stage LR) — deferred. P5 needs new policy retrain; P1/P4/P7 need
  oracle regeneration or DynamicScheduler refactor.

Resource budget: 10 hours overnight on 2 A100 servers.

- **stars** (free A100): orchestrator hit a bash arithmetic bug in `bundle_end()`
  after the first bundle and exited silently. Only 3 runs completed: P6 baselines seed=1.
- **industry** (A100 shared with TS-forecasting sweep): all 7 sequential jobs
  completed cleanly between 22:31–03:37 MSK (43 min/job avg). No interaction with
  the parallel TS sweep.

**Cosine reference**: prior 4-seed mean = 69.81 ± 0.33 pp.
Two fresh cosine seeds were also added on industry to anchor the comparison.

## Headline final accuracy (last completed epoch = 35)

| Variant | n_seeds | seeds | mean ± std | Δ vs cosine 69.81 | Best peak (any epoch) |
|---|---:|---|---|---:|---:|
| `cosine` | 2 | 10,11 | **70.02** ± 0.01 | +0.21 | 70.03 |
| `curve_veto_clamp07` | 1 | 10 | **50.95** ± 0.00 | -18.86 | 64.09 |
| `hypergrad` | 2 | 1,10 | **41.99** ± 2.36 | -27.82 | 54.94 |
| `residual_log` | 2 | 10,11 | **69.12** ± 0.47 | -0.69 | 69.45 |
| `sf_adamw` | 1 | 1 | **48.64** ± 0.00 | -21.17 | 48.64 |
| `sf_sgd` | 2 | 1,10 | **51.94** ± 0.10 | -17.87 | 54.17 |

_Reference cosine: **69.81 ± 0.33 pp** (4 prior seeds), depth_list=110_20, student_steps_ratio=4, batch_size=256, 35 epochs._

## Per-run breakdown

| Variant | Seed | Server | Final acc | Final loss | Best acc | Best epoch |
|---|---:|---|---:|---:|---:|---:|
| `cosine` | 10 | industry | 70.01 | 1.0807 | 70.02 | 34 |
| `cosine` | 11 | industry | 70.03 | 1.0829 | 70.03 | 35 |
| `curve_veto_clamp07` | 10 | industry | 50.95 | 1.8159 | 64.09 | 25 |
| `hypergrad` | 1 | stars | 43.66 | 2.3241 | 53.50 | 31 |
| `hypergrad` | 10 | industry | 40.32 | 2.5606 | 54.94 | 18 |
| `residual_log` | 10 | industry | 68.79 | 1.1382 | 68.99 | 34 |
| `residual_log` | 11 | industry | 69.45 | 1.1235 | 69.45 | 35 |
| `sf_adamw` | 1 | stars | 48.64 | 1.8925 | 48.64 | 35 |
| `sf_sgd` | 1 | stars | 51.87 | 1.9152 | 53.29 | 22 |
| `sf_sgd` | 10 | industry | 52.01 | 1.8893 | 54.17 | 34 |

## Verdict

### Top-3 by mean final accuracy

1. `cosine` — 70.02 ± 0.01 pp (n=2, peak=70.03, Δ vs cosine=+0.21)
2. `residual_log` — 69.12 ± 0.47 pp (n=2, peak=69.45, Δ vs cosine=-0.69)
3. `sf_sgd` — 51.94 ± 0.10 pp (n=2, peak=54.17, Δ vs cosine=-17.87)

Best variant `cosine` reached **70.02 ± 0.01 pp** (+0.21 vs cosine 69.81). **MATCH**.

## Hypothesis-by-hypothesis interpretation

### P6.1 Schedule-Free SGD (lr=0.5, MLCommons AlgoPerf winner)

- 2 seeds: **51.94 ± 0.10** pp (-17.87 vs cosine).
- Verdict: **BELOW COSINE**.
- Note: SF tuned with lr=0.5 (matches our cosine peak); did not search for optimal lr.

### P6.1 Schedule-Free AdamW (lr=5e-4)

- 1 seed(s): **48.64** pp (-21.17 vs cosine).
- AdamW typically benefits from longer training; 35 epochs may be too short.

### P6.2 Hypergradient SGD (own impl, no offline data)

- 2 seeds: **41.99 ± 2.36** pp (-27.82 vs cosine).
- Cheapest possible adaptive baseline — uses only the 30-dim probe we already compute.
- Note: teacher reached 74.9% (cosine reference) — hypergrad on student got stuck at LR ~0.58
  (never decayed below ~0.5). Same root cause as `curve_full_opt`: probe gradient near
  the wide-flat-minimum region is too noisy to drive LR to the cosine endpoint of ~0.

### P2 AdaLRS-style veto on top of curve_argmin (clamp=0.7)

- 1 seed(s): **50.95** pp (-18.86 vs cosine).
- Compared to no-veto control from `1h_curve_opt` (≈50.16 pp): Δ = +0.79 pp.
- Verdict: did the veto rescue the policy? See peak vs final to see whether late-stage
  reverts to cosine helped.

### P3 Residual-on-log-cosine head (--policy_output residual_log)

- 2 seeds: **69.12 ± 0.47** pp (-0.69 vs cosine).
- New policy mode: GRU output ∈ [−1, +1], `LR = cosine · exp(out)`.
  Identity-initialised, naturally bounded. Pretrained online from `targets_v3`.
- Most novel of tonight's contributions; closest to cosine_multiplier in spirit but
  with a stricter, identity-init bound.

### Cosine baseline (replication, fresh seeds)

- 2 seeds (10, 11): **70.02 ± 0.01** pp.
- Combined with prior 4-seed reference 69.81 ± 0.33, gives total 6-seed cosine baseline.

## Caveats

1. **Stars budget lost**: bash arithmetic bug in `run_remote_night10h_stars.sh::bundle_end`
   killed the orchestrator after Bundle 1. We lost ~5 hours of stars compute = 5 bundles
   = 15 runs. Specifically lost:
   - Bundles 2–3: P6 baselines seeds 2, 3 (would give 3-seed std for sf_sgd/sf_adamw/hypergrad)
   - Bundles 4–5: P2 veto ablations (`clamp=0.5`, `no_veto` control, 2 more seeds for `clamp=0.7`)
   - Bundle 6: P3 residual_log seeds 1–3 on stars
   Fix: replace `T=$(bundle_start ...)` with explicit `bundle_start NAME && T=$(date +%s)`.
   Or just use `set +e; date +%s` in `bundle_start` (no echo capture).

2. **Hyperparameter not tuned for new optimisers**: SF/Prodigy/Hypergrad each have their
   own optimal LR. We used lr=0.5 for SF SGD (matches cosine peak) and lr=5e-4 for SF AdamW
   (PyTorch AdamW default), no search. A 1-seed lr-sweep would change these numbers.

3. **35 epochs is short** for evaluating SF/AdamW which benefit from longer schedules.
   Cosine on this exact recipe peaks at ~70% at epoch 35; SF/AdamW might catch up at 50+ epochs.

## Ranked next steps

1. **Fix the orchestrator bug and re-run lost stars bundles** (5h GPU): get the missing
   2 seeds for sf_sgd/sf_adamw/hypergrad + the P2 ablations + P3 residual_log on stars.
   Same recipe, just bash fix.
2. **LR sweep for the best non-cosine optimiser** (1 bundle = 80 min): sweep lr ∈ {0.1, 0.3, 1.0}
   for the winner of P6 (probably SF SGD). Likely closes 1-2 pp.
3. **Implement P5 (extra features)** + **P3 + P5** combo on existing modular policy.
   Adds 3 cheap signals (volatility ratio, sharpness proxy, gradnorm decay) to the
   policy input. Requires policy retrain on npy with new feature dim.
4. **Implement P1 (val-acc forecast head)**: requires regenerating the oracle to log
   per-LR validation accuracy, not just probe loss. ~6 GPU-hours but highest ceiling.
5. **AdaLRS-style velocity veto on residual_log** (combine P2 + P3): drop in safeguard
   without changing the policy. ~30 min code.

## Artifacts

- `results/night10h/snapshots/night10h_{stars,industry}/<run>/metrics.jsonl` — per-epoch metrics
- `results/night10h/logs/night10h_{stars,industry}/per_run/<run>.log` — full stdout
- `results/night10h/per_epoch.csv` — every epoch of every run, flattened