# Robust12h Campaign Plan

Started: 2026-05-17 18:24 MSK (`mlspace` restarted at 18:26 after duplicate-name fix).

Goal: find perturbation regimes where the residual-log LR policy, especially the
PatchTST top-5 policy, beats cosine, and explain those wins via recovery
metrics, LR trajectories, and noise/OOD behavior.

## Servers

| Server | Session | Focus | Expected runs |
|---|---|---|---:|
| `stars` | `robust12h_stars` | LR shocks / warm-restart-like stress | 60 |
| `industry` | `robust12h_industry` | label noise, input noise, compound noise + LR shock | 63 |
| `mlspace` | `robust12h_mlspace` | open questions: single PatchTST, top-5 replication, zero-mean EMA, SF-SGD sweep, mid-training noise onset, 90ep OOD | 52 |

All servers run 3 parallel processes on a single A100-80GB.

## Experiment Families

- `stars`: `shock_lr ∈ {0.1, 0.5, 1.0, 2.0}` at epoch 20 for 2 epochs,
  methods `{cosine, residgru, patchtst}`, seeds `{1,2,3,4}`. Extra no-shock
  and medium-shock PatchTST/cosine controls use seeds `{5,6,7,8}`.
- `industry`: symmetric label noise from start (`0.10, 0.20, 0.30`), input
  Gaussian noise from start (`0.05, 0.15`), compound perturbation
  (`label_noise=0.20 + shock_lr=1.0@e15`), with extra residual-GRU compound
  cells.
- `mlspace`: single-checkpoint PatchTST (`O1`), top-5 PatchTST replication
  (`O2`), zero-mean EMA ensemble fix (`O3`), Schedule-Free SGD LR sweep (`O4`),
  mid-training label-noise onset (`O5`), and 90-epoch OOD horizon (`O6`).

## Analysis Outputs

Watcher: `scripts/watch_robust12h.py`.

When all three tmux sessions finish, it pulls:

- `logs/robust12h_{stars,industry,mlspace}`
- `snapshots/robust12h_{stars,industry,mlspace}`

Then it runs `scripts/analyze_robust12h.py`, producing:

- `results/robust12h/per_run.csv`
- `results/robust12h/per_epoch.csv`
- `results/robust12h/cells.csv`
- `results/robust12h/REPORT.md`
- `results/robust12h/figures/`

Primary metrics:

- final / best / last-10 student accuracy,
- shock dip and recovery epochs,
- post-noise drop after noise onset,
- per-cell mean/std and deltas vs matching cosine cells.
