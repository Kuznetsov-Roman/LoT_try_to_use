# LR Policy Robustness Reproducibility Package

This package is a self-contained result artifact for the CIFAR-100 Learning-from-Teaching distillation experiments with a trainable learning-rate policy. It contains the code snapshot, oracle policy-training data, CIFAR-100 cache, policy checkpoints, final model checkpoints, raw logs, parsed tables, figures, and notebooks needed to audit and reproduce the reported results.

## Direct Notebook Links

- [notebooks/inference_and_figures.ipynb](notebooks/inference_and_figures.ipynb) - headline PatchTST seed-3 result, training curves, LR trajectory.
- [notebooks/robust12h_summary.ipynb](notebooks/robust12h_summary.ipynb) - robustness campaign summary, paired deltas, and winning seeds.
- [RESULTS.md](RESULTS.md) - standalone English results file.
- [RESULTS_RU.md](RESULTS_RU.md) - Russian version of the results file.

## Executive Summary

The trainable LR policy is not a universal replacement for cosine. The strongest and most reproducible conclusion is narrower:

- The best single run is `PatchTST output ensemble top5 blend075`, seed 3, 60 epochs: **70.70** vs cosine seed 3 **70.45**.
- Across more seeds, PatchTST mostly **matches** cosine rather than clearly beating it: top5 blend075 gives **70.08 +/- 0.59** vs cosine **70.20 +/- 0.23** at 60 epochs.
- The clearest robust win is a mild LR-down perturbation at epoch 20: `shock_lr=0.10`, PatchTST **70.17 +/- 0.38** vs cosine **69.45 +/- 0.52**, delta **+0.72 pp**.
- Under label noise, input noise, compound perturbations, and extreme LR shocks, cosine is still stronger or statistically tied.
- The value of the approach is best described as **trajectory repair** under selected schedule perturbations, not as a globally superior default scheduler.

Intermediate conclusion: the residual-log parameterization is the right safety prior. It keeps behavior close to cosine, prevents catastrophic raw-LR failure, and creates a few useful robustness cases.

## Research Timeline and All Results Map

All source markdown reports from the study are preserved under `reports/source_results/`. The table below maps each research stage to the result that should be cited from this package.

| Stage | Source report in package | Main result | Intermediate conclusion |
|---|---|---|---|
| Local smoke test / raw LR | `reports/source_results/hour_gpu_summary.md`, `reports/source_results/final_resume.md` | Raw dynamic LR: **60.23** final vs cosine **69.13**. Safe cosine multiplier: **69.17 +/- 0.14** over 6 seeds. | Direct raw LR prediction fails because it keeps LR too high late; constrained policy outputs are required. |
| Overnight policy baselines | `reports/source_results/overnight_summary.md` | Current GRU policy: **34.45-44.34**; modular policy: **68.29-69.05**; cosine seeds around **69.35-70.11**. | Architecture changes alone do not solve the LR target problem. |
| Six-hour multiplier follow-up | `reports/source_results/six_hour_policy_summary.md` | Best variants around **69.33-69.44**, still below cosine reference **69.81 +/- 0.33**. | More multiplier tuning gives baseline-level behavior but no clear win. |
| Curve / MPC policy POC | `reports/source_results/2h_curve/RESULTS.md` | Curve frozen **32.73**, curve online **31.73**, cosine ref **70.64** on seed 400. | Predicting the curve can improve MSE, but argmin selection saturates to destructive high LR. |
| Curve optimized follow-up | `reports/source_results/1h_curve_opt/RESULTS.md` | Best optimized curve variant peaks at **55.03**, final/last observed **50.16**, still far below cosine **69.81 +/- 0.33**. | Capping/blending improves the failure by about +23 pp vs the POC, but the curve axis remains null. |
| Literature-driven proposals | `reports/source_results/trainable_lr_literature_review.md` | Ranked proposals P1-P7: residual-log head, AdaLRS veto, schedule-free/hypergrad baselines, bandit reformulation, sharpness features. | The successful direction should be bounded residual control over cosine, not unconstrained LR prediction. |
| Night10h + catchup | `reports/source_results/night10h/RESULTS.md`, `reports/source_results/RESULTS_combined.md` | Residual-log GRU: **69.31 +/- 0.43** over 6 seeds vs cosine **69.81 +/- 0.33**. Hypergrad/SF baselines stay far below cosine. | P3 residual-log is the first tight learnable-policy regime, but still slightly below cosine on mean. |
| Full pipeline horizons | `reports/source_results/RESULTS_full_pipeline.md` | 35ep: -0.50 pp; 60ep final: -0.39 pp; 60ep last-10: **+0.05 pp**; warm restart has 2.5x smaller transient drop. | Residual-log generalizes across horizons and improves shock tolerance, but not average final accuracy. |
| SOTA architecture sweep | `reports/source_results/advanced_industry/summary.md` | PatchTST offline MSE **0.258** with 6,984 params; deployment **69.68 +/- 0.16** at 35ep. | PatchTST is the best compact LR-policy backbone. |
| PatchTST follow-ups | `reports/source_results/patchtst_followups_current/current_summary.md` | Fixed NBeats **69.64**, DLinear **69.56** at 35ep; PatchTST ext50 around **69.4-69.6** while still running. | Fixed-window architectures can run after padding fixes, but PatchTST remains the stronger practical path. |
| Blend-tuning screen | `reports/source_results/blendtune3h_combined/SUMMARY.md` | Blend080 single seed reached **70.55**, +0.10 vs cosine seed3 but below previous **70.70**. | Moderate cosine blends are the only PatchTST variants that occasionally beat cosine. |
| Beat-cos campaign | `reports/source_results/beatcos_combined/SUMMARY.md` | Top5 blend075: **70.083 +/- 0.592** vs cosine **70.197 +/- 0.234**; seed3 reaches **70.70**. | The headline win is real but seed-sensitive; the multi-seed mean is tied/slightly below cosine. |
| Improve5h postmortem | `reports/source_results/improve5h_combined/SUMMARY.md`, `reports/source_results/improve5h_combined/analysis/report.md` | Best improve5h run **70.29**, below cosine seed3 **70.45** and prior PatchTST **70.70**. | Output ensembling collapses `policy_pred` into near-constant LR attenuation, not adaptive control. |
| External adaptive LR comparison | `reports/source_results/sota_lr_comparison/SUMMARY.md`, `reports/source_results/sota_lr_comparison/analysis/report.md` | Best external comparator AdaLRS default: **67.41 +/- 0.82**, -2.79 pp vs cosine mean. | Generic adaptive LR controllers do not beat cosine in this LoT setup. |
| Robust12h campaign | `reports/source_results/robust12h/REPORT.md`, `reports/robust12h_PACKAGE_SUMMARY.md` | 167 completed runs. Best paired robust cell: PatchTST `shock_lr=0.10`, **+0.72 pp** vs cosine. | The learned LR policy is useful as targeted trajectory repair, especially after mild LR-down shocks. |

Intermediate conclusion: the complete history is a funnel. Raw LR, curve argmin, schedule-free, hypergrad, bandit-style controls, and broad blend tuning mostly fail; the surviving claim is bounded residual-log PatchTST as a conservative, occasionally useful repair controller over cosine.

## Method

The headline policy is an output ensemble of five PatchTST residual-log LR policies:

```text
policy_lr = cosine_lr * exp(clip(mean(policy_outputs), -1, 1))
next_lr = 0.25 * policy_lr + 0.75 * cosine_lr
```

The first 10 epochs use cosine before policy corrections are applied. The ensemble checkpoints are in `checkpoints/policies_beatcos/`.

Intermediate conclusion: the identity prior is essential. When the policy is uncertain, output near zero means "use cosine"; this is why OOD-horizon results stay close to baseline instead of collapsing.

## Methods and Architectures Applied

| Method / architecture | Role in the study | Brief architecture / rule | Result-level takeaway |
|---|---|---|---|
| `cosine` | Main hand-designed baseline | Deterministic cosine annealing from initial LR to the minimum LR over the training horizon. No learned state. | Strongest default baseline; most learned variants match or lose to it. |
| `raw_lr` dynamic policy | First learned-LR baseline | GRU policy consumes a window of landscape/student features and directly regresses scalar LR. | Failed: **60.23** vs cosine **69.13** in the local smoke test because late LR stayed too high. |
| `cosine_multiplier` | Safer early learned policy | Learned scalar multiplier applied on top of cosine, with caps/blending to keep LR near the baseline schedule. | Recovered baseline-level behavior, about **69.17 +/- 0.14**, but no clear win. |
| Modular multiplier policy | Architecture ablation | MLP/modular policy over landscape and latent student statistics; predicts bounded schedule-relative LR. | Stayed near **69.3-69.4**, below cosine reference. |
| `residual_log` GRU | Main safe GRU policy | GRU over feature windows; output is clipped to `[-1, 1]` and decoded as `cosine_lr * exp(output)`. | First stable learned policy: **69.31 +/- 0.43** at 35ep; tied on 60ep last-10 average. |
| `curve_argmin` / CurveLRPolicy | Multi-step landscape-forecast policy | Neural policy predicts a full 30-point loss curve for candidate LRs; deployment chooses argmin or softened argmin. | Catastrophic without strong constraints: curve MSE improves, but argmin drifts to destructive high LR. |
| Curve policy with cap/blend/online MPC | Stabilized curve follow-up | Same curve forecast head, but LR is capped, blended with cosine, and optionally updated online by curve MSE. | Improved from ~32% to peak ~55%, still far below cosine. |
| PatchTST residual-log | Best compact policy backbone | Patch-based time-series Transformer over short feature windows; residual-log output over cosine. | Best backbone: offline MSE **0.258**, 6,984 params, 35ep deployment **69.68 +/- 0.16**. |
| PatchTST output ensemble | Headline learned policy | Mean of five PatchTST policy outputs, then clipped residual-log transform and `0.25 policy + 0.75 cosine` blend. | Best single run **70.70** vs cosine seed3 **70.45**; multi-seed mean tied/slightly below cosine. |
| PatchTST checkpoint soup | Ensemble/averaging ablation | Averages or soups multiple PatchTST checkpoints before residual-log deployment. | Stable but did not beat cosine mean; best soup variants stayed around **69.9-70.0**. |
| PatchTST zero-mean EMA | Bias-removal ablation | Subtracts running mean from policy output before applying residual-log correction. | Did not help: **69.72 +/- 0.31** in robust12h open-item check. |
| TCN policy | Time-series architecture ablation | Temporal convolutional network over feature windows, residual-log output. | Ran successfully and reached **69.37** on one 35ep seed; below PatchTST. |
| NBeats policy | Time-series architecture ablation | Deep block-based forecast architecture adapted to LR-policy features. | Initially failed due fixed-window assumptions; after fixes reached about **69.64** on one 35ep seed. |
| DLinear policy | Lightweight linear time-series ablation | Decomposition/linear time-series head over policy windows. | Initially failed due fixed-window assumptions; after fixes reached about **69.56** on one 35ep seed. |
| Schedule-Free SGD / AdamW | External adaptive optimizer baseline | Schedule-free optimizer family with iterate averaging; no learned LR policy. | Strong negative control in this setup: far below cosine in both night10h and robust12h sweeps. |
| Hypergradient / Hypergrad-HB | Online LR adaptation baseline | Updates LR with hypergradient-style feedback from loss/probe dynamics, no offline policy training. | Failed to decay LR properly; remained far below cosine. |
| AdaLRS-style veto / AdaLRS comparators | Loss-guided adaptive LR baseline | Uses loss trend / velocity-style safeguards to accept, reject, or clamp LR changes. | Veto did not rescue curve policy; external AdaLRS was best SOTA comparator but still **-2.79 pp** vs cosine mean. |
| Bandit EXP3 / UCB | Discrete LR-grid baseline | Treats LR selection as a non-stationary bandit over candidate LR arms. | Very unstable and well below cosine in the SOTA comparison. |
| Robust perturbation protocols | Stress tests rather than optimizers | LR shocks, label noise, input noise, compound noise+shock, delayed noise onset, and 90ep OOD horizon. | Identified the useful regime: PatchTST repairs mild LR-down shock, but does not solve noisy data or extreme shocks. |

Intermediate conclusion: the method search narrowed from unconstrained LR prediction and curve argmin toward bounded residual control over cosine. PatchTST improves the compactness and stability of that controller, but the controller remains most useful as a targeted repair mechanism rather than a universal scheduler.

## Full Pipeline Results

These are the main deployment-horizon experiments for the residual-log family.

| Horizon / setup | Cosine | Learnable policy | Delta | Main conclusion |
|---|---:|---:|---:|---|
| 35 epochs from scratch, 6 seeds | 69.81 +/- 0.33 | GRU residual-log 69.31 +/- 0.43 | -0.50 | matches but below cosine |
| 50 epoch warm-restart, LR jump at e36 | drop about -18 pp, recover in about 10 epochs | drop about -7 pp, recover in about 5 epochs | 2.5x smaller transient | residual policy is more shock tolerant |
| 60 epochs from scratch, 3 seeds | 70.20 +/- 0.23 | GRU residual-log 69.81 +/- 0.21 | -0.39 | near-cosine final accuracy |
| 60 epochs, last-10 average | 68.42 +/- 0.35 | GRU residual-log 68.47 +/- 0.57 | +0.05 | smoothed metric is tied |

Intermediate conclusion: on the training horizon and OOD horizon, the policy does not improve the mean final accuracy, but it tracks cosine closely. The important positive signal is robustness to schedule shocks.

## PatchTST Architecture and Beat-Cosine Campaign

PatchTST replaced the original GRU LR-policy backbone. It is much smaller and performed better offline.

| Policy backbone | Parameters | Offline test MSE | Deployment result |
|---|---:|---:|---|
| PatchTST | 6,984 | 0.258 | strongest backbone |
| TCN | 115,713 | 0.288 | 69.37 on one seed |
| GRU baseline | 349,332 | 0.307 | 69.31 +/- 0.43 at 35ep |
| NBeats | 1,922,679 | 0.321 | overfit / deploy crash in early tests |
| DLinear | 253 | 0.362 | deploy crash in early tests |

60 epoch PatchTST campaign:

| Method | n | Final mean | Final std | Last-10 mean | Delta vs cosine final |
|---|---:|---:|---:|---:|---:|
| cosine | 3 | 70.197 | 0.234 | 68.425 | 0.000 |
| patchtst_output_ens_top5_blend075 | 3 | 70.083 | 0.592 | 68.590 | -0.113 |
| patchtst_output_ens_top5_blend090_ema03 | 3 | 70.003 | 0.214 | 67.977 | -0.193 |
| patchtst_soup_top5_blend090 | 3 | 69.943 | 0.146 | 68.323 | -0.253 |
| residual_log GRU reference | 3 | 69.807 | 0.205 | 68.473 | -0.390 |
| patchtst_plain | 3 | 69.627 | 0.192 | 67.499 | -0.570 |

Important seed-level result:

| Method | Seed | Final acc | Final loss | Last-10 acc |
|---|---:|---:|---:|---:|
| cosine | 3 | 70.45 | 1.113 | 68.83 |
| patchtst_output_ens_top5_blend075 | 3 | **70.70** | **1.093** | **69.12** |

Intermediate conclusion: PatchTST is the best policy backbone, but the 70.70 headline is a single-seed win. The multi-seed mean is tied/slightly below cosine.

## Robust12h Campaign

The robustness campaign ran on three A100 servers and scanned **167 completed runs**:

- `stars`: LR shocks and no-shock controls.
- `industry`: label noise, input noise, and compound label-noise plus LR-shock perturbations.
- `mlspace`: open-item checks, top5 replication, single-checkpoint PatchTST, zero-mean EMA, Schedule-Free SGD sweep, noise onset, and 90 epoch OOD horizon.

### Paired Robustness Summary

Positive delta means the learnable method beats matched cosine in that perturbation cell.

| Experiment | Perturbation | Method | Cosine | Method | Delta |
|---|---|---|---:|---:|---:|
| shock | shock_lr=0.10 | patchtst | 69.45 | **70.17** | **+0.72** |
| shock | shock_lr=0.10 | residgru | 69.45 | 69.31 | -0.14 |
| noshock | noshock | patchtst | 69.90 | 69.88 | -0.02 |
| O6_long90 | long90 | patchtst | 70.10 | 69.99 | -0.11 |
| O5_noise_onset_e15 | label=0.20@e15 | patchtst | 64.50 | 64.25 | -0.25 |
| innoise | input_noise=0.05 | patchtst | 65.98 | 65.90 | -0.08 |
| innoise | input_noise=0.15 | patchtst | 57.33 | 56.99 | -0.34 |
| lblnoise | label_noise=0.10 | patchtst | 66.60 | 66.34 | -0.26 |
| lblnoise | label_noise=0.20 | patchtst | 63.71 | 63.19 | -0.52 |
| lblnoise | label_noise=0.30 | patchtst | 60.81 | 60.29 | -0.52 |
| compound | label=0.20+shock_lr=1.0@e15 | patchtst | 64.08 | 63.11 | -0.98 |
| shock | shock_lr=0.50 | patchtst | 70.02 | 69.53 | -0.49 |
| shock | shock_lr=1.00 | patchtst | 69.90 | 69.69 | -0.21 |
| shock | shock_lr=2.00 | patchtst | 62.14 | 59.77 | -2.37 |
| shock | shock_lr=1.00 | residgru | 69.90 | 62.83 | -7.07 |
| shock | shock_lr=2.00 | residgru | 62.14 | 60.19 | -1.95 |

Intermediate conclusion: the positive region is specific. PatchTST helps when the LR is temporarily too small, but not when noise dominates the objective or when LR is forced too high.

## Winning Seeds Case

The strongest reproducible positive case is the mild LR-down shock. During epochs 20-21, the student LR is forced to `0.10`, then the scheduler resumes. PatchTST wins 3 of 4 paired seeds.

| Seed | Cosine final acc | PatchTST final acc | Delta | Status |
|---:|---:|---:|---:|---|
| 1 | 69.94 | 69.51 | -0.43 | non-win |
| 2 | 69.58 | 70.30 | +0.72 | WIN |
| 3 | 69.71 | 70.44 | +0.73 | WIN |
| 4 | 68.57 | 70.42 | +1.85 | WIN |

The package includes final student and teacher checkpoints for all paired shock01 runs under `checkpoints/winning_shock01/`.

Intermediate conclusion: this is the best case to show as a separate qualitative result. It is seed-paired, has a positive mean delta, and has interpretable mechanism: cosine is fixed by epoch, while the residual policy can correct after the down-shock.

## Noise and Compound Perturbations

| Perturbation | Cosine | PatchTST | Delta | Conclusion |
|---|---:|---:|---:|---|
| input_noise=0.05 | 65.98 +/- 0.43 | 65.90 +/- 0.44 | -0.08 | tie |
| input_noise=0.15 | 57.33 +/- 0.37 | 56.99 +/- 0.11 | -0.34 | slight cosine win |
| label_noise=0.10 | 66.60 +/- 0.49 | 66.34 +/- 0.23 | -0.26 | tie |
| label_noise=0.20 | 63.71 +/- 0.23 | 63.19 +/- 0.62 | -0.52 | cosine win |
| label_noise=0.30 | 60.81 +/- 0.28 | 60.29 +/- 0.34 | -0.52 | cosine win |
| label=0.20 + shock_lr=1.0@e15 | 64.08 +/- 0.35 | 63.11 +/- 0.44 | -0.98 | cosine win |
| label=0.20@e15 onset | 64.50 +/- 0.10 | 64.25 +/- 0.38 | -0.25 | tie |

Intermediate conclusion: policy corrections do not solve label or input corruption. When the data distribution is noisy, cosine remains a very strong regularized baseline.

## OOD Horizon and Replication Checks

| Open item | n | Result | Delta / interpretation |
|---|---:|---:|---|
| PatchTST top5 blend075 replication | 6 | 69.99 +/- 0.44 | -0.21 vs 60ep cosine ref |
| Single-checkpoint PatchTST | 5 | 69.82 +/- 0.29 | worse than top5 ensemble |
| Zero-mean EMA ensemble fix | 3 | 69.72 +/- 0.31 | did not help |
| 90ep OOD horizon, cosine | 5 | 70.10 +/- 0.40 | reference |
| 90ep OOD horizon, PatchTST | 5 | 69.99 +/- 0.10 | -0.11, much lower variance |

Intermediate conclusion: the 70.70 headline is likely a high seed rather than a new mean. However, OOD horizon stability remains a useful property: PatchTST has lower variance at 90 epochs.

## External Adaptive-LR Comparators

None of the external adaptive-LR controller families reached the cosine regime under the same 60 epoch LoT distillation setting.

| Method family | Variant | n | Final mean | Final std | Delta vs cosine mean |
|---|---|---:|---:|---:|---:|
| AdaLRS | default | 3 | 67.41 | 0.82 | -2.79 |
| AdaLRS | aggressive | 3 | 65.99 | 2.29 | -4.21 |
| AdaLRS | narrow_safe | 3 | 54.73 | 4.92 | -15.46 |
| Bandit EXP3 | safe | 3 | 53.61 | 17.94 | -16.58 |
| Bandit EXP3 | fast | 3 | 52.09 | 6.49 | -18.10 |
| Bandit UCB | ucb | 3 | 40.88 | 23.43 | -29.31 |
| Hypergrad-HB | fast | 3 | 42.67 | 8.21 | -27.53 |
| Hypergrad-HB | safe | 3 | 39.75 | 3.95 | -30.45 |
| Hypergrad-HB | smooth | 3 | 34.09 | 8.06 | -36.11 |

Schedule-Free SGD LR sweep from robust12h:

| SF-SGD LR | n | Final mean | Final std | Delta vs 35ep cosine ref |
|---:|---:|---:|---:|---:|
| 0.1 | 3 | 39.54 | 1.42 | -30.27 |
| 0.3 | 3 | 49.02 | 1.13 | -20.79 |
| 0.7 | 3 | 57.79 | 0.38 | -12.02 |
| 1.0 | 3 | 58.93 | 1.39 | -10.88 |

Intermediate conclusion: the negative controls are important. The result is not that any adaptive LR method helps; the useful behavior is specific to a bounded residual over cosine.

## Improve5h Postmortem

The follow-up sweep over blends, warmups, cosine handoff, and ensemble sizes did not find a better configuration:

- Best improve5h run: `outens_blend0800_warm05_seed3_60ep` at **70.29**, below cosine seed3 **70.45** and below the previous PatchTST headline **70.70**.
- Root cause: output ensemble predictions collapsed to an almost constant bias.
- Per-epoch `policy_pred` standard deviation over epochs 10-55 was only about `0.003`, while the mean bias was around `-0.264` for top5.
- The sweep mostly changed fixed cosine attenuation in the range about `0.94x` to `0.98x`, not state-dependent control.

Intermediate conclusion: further tuning of blend/EMA/warmup is unlikely to create a robust mean improvement. More useful directions would require new state-dependent targets or richer perturbation-aware training.

## Final Interpretation

The evidence supports this conservative claim:

1. A bounded residual-log LR policy can safely match cosine across normal and OOD horizons.
2. PatchTST is the best compact backbone for the policy.
3. The strongest positive result is trajectory repair after mild LR-down perturbation.
4. The approach does not consistently beat cosine under label noise, input noise, compound perturbations, or extreme LR shocks.
5. The single-seed 70.70 headline is real and reproducible from the bundled checkpoint/logs, but the multi-seed mean is tied with cosine rather than better.

## Package Contents

- `code/` - package-local code snapshot used by the reproduction scripts.
- `data/oracle/` - oracle features and targets used to train the LR policy.
- `data/robust12h/` - parsed robustness tables: `cells.csv`, `pairs.csv`, `per_run.csv`, `per_epoch.csv`.
- `data/cifar/cifar-100-python/` - bundled CIFAR-100 cache.
- `checkpoints/policies_beatcos/` - five PatchTST policy checkpoints used by the top-5 ensemble.
- `checkpoints/final_headline_seed3/` - final student/teacher checkpoints for the 70.70 seed-3 run.
- `checkpoints/winning_shock01/` - final student/teacher checkpoints for paired cosine and PatchTST shock01 seeds 1-4.
- `logs/` and `snapshots/` - raw logs and per-epoch metrics needed to audit the tables.
- `figures/` - headline and robust12h figures.
- `reports/` - curated source reports plus `reports/source_results/`, a preserved copy of every markdown result report from the project `results/` tree.
- `sota_comparison/` - archived negative-control adaptive LR comparator results.
- `RESULTS.md` and `RESULTS_RU.md` - standalone English and Russian result files.

## Reproduce

From the package root on a CUDA machine:

```bash
PYTHON_BIN=/path/to/python bash scripts/reproduce_headline_seed3.sh
```

To reproduce the paired LR-down shock case:

```bash
PYTHON_BIN=/path/to/python bash scripts/reproduce_shock01_winning_seeds.sh
```

To regenerate package-local tables/summary and figures from bundled CSV/logs:

```bash
python scripts/analyze_robust12h.py
python scripts/plot_figures.py
```

## Data Notes

CIFAR-100 is bundled under `data/cifar/cifar-100-python/`. The reproduction scripts still pass `--download` to torchvision; this is harmless when the cache is present and gives a fallback if the directory is removed. All experiment-specific data required by the LR policy (oracle arrays, policy checkpoints, logs, parsed tables) is bundled.

## Integrity

`MANIFEST.sha256` contains SHA256 hashes for all files in this package except the manifest itself. Verify with:

```bash
sha256sum -c MANIFEST.sha256
```
