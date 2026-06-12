# SOTA Adaptive LR Comparator Report

This folder contains the external adaptive / trainable LR comparison used to
contextualize the PatchTST output-ensemble result.

## Scope

All runs use the same CIFAR-100 ResNet110 -> ResNet20 distillation protocol as
the headline experiment: 60 epochs, batch size 256, alpha 0.5, and
student_steps_ratio 4.

Compared methods:

- `adalrs`: loss-guided LR search over the existing 30-point LR probe.
- `bandit_exp3` / `bandit_ucb`: non-stationary bandits over the LR grid.
- `hypergrad_hb`: batch-level hypergradient descent with heavy-ball smoothing.

## Verdict

No external adaptive LR comparator beats cosine or the PatchTST output ensemble.
The best SOTA-comparator run is `adalrs_default_seed1_60ep` at 68.22% student
test accuracy, while the PatchTST headline reaches 70.70% and cosine seed3
reaches 70.45%.

## Files

- `report.md` - concise reviewer-facing analysis and diagnosis.
- `SUMMARY.md` - generated full ranking table for all 27 comparator runs.
- `tables/summary.csv` - per-run metrics.
- `tables/method_summary.csv` - method-level aggregates.
- `figures/sota_method_final_accuracy.png` - method mean comparison.
- `figures/sota_top_runs.png` - best individual comparator runs.
- `logs/` - raw logs for all comparator runs.
- `scripts/` - exact launch and aggregation scripts.
- `scripts/plot_sota_comparison.py` - regenerate the SOTA comparison figures.
- `code_snapshot/my_research.py` - training entrypoint snapshot containing the
  comparator scheduler implementations.

## Checkpoints

Best comparator checkpoints are stored outside this folder under
`checkpoints/sota_best/`:

- `adalrs_default_seed1/` - best external comparator overall.
- `bandit_ucb_seed1/` - best bandit run.
- `hypergrad_hb_fast_seed3/` - best Hypergrad-HB run.

These are intentionally included as negative-control checkpoints. They document
that the comparator training completed and make post-hoc inspection possible,
but they are not recommended as deployable policies.
