# 3h Blend-Tuning Results

Run date: 2026-05-14.

Goal: tune `output ensemble top-5 PatchTST` around conservative cosine blends
`0.65-0.85`, prioritising many variants over stable replication.

Reference for this screening:

| Reference | Seed | Final acc | Last-10 avg |
|---|---:|---:|---:|
| cosine | 3 | 70.45 | 68.83 |
| previous output ensemble blend075 | 3 | 70.70 | 69.12 |

## Completed 60ep runs

| Variant | Server | Seed | Status | Final acc | Final loss | Last-10 avg | Result vs cosine seed3 |
|---|---|---:|---|---:|---:|---:|---:|
| output_ens_top5_blend080_ema00 | stars | 3 | OK | **70.55** | 1.131 | **68.87** | **+0.10** |
| output_ens_top5_blend065_ema00 | stars | 3 | OK | 70.10 | 1.128 | 68.33 | -0.35 |
| output_ens_top5_blend070_ema00 | stars | 3 | OK | 69.68 | 1.130 | 68.11 | -0.77 |

## Partial / invalid runs

`industry` did not complete the high-blend/EMA side of the grid. Logs stop at
epoch 15-16 and no `RUN_DONE` markers were written:

| Variant | Last epoch | Best acc so far | Last acc | Status |
|---|---:|---:|---:|---|
| output_ens_top5_blend085_ema00 | 15 | 49.50 | 43.83 | partial |
| output_ens_top5_blend075_ema015 | 15 | 45.41 | 30.53 | partial |
| output_ens_top5_blend080_ema015 | 15 | 44.71 | 21.92 | partial |

The partial `industry` jobs were much slower (~7-8 min per epoch) and are not
included in headline comparisons.

## Verdict

The sweep found one positive single-seed candidate:

`output_ens_top5_blend080_ema00_seed3_60ep` reached **70.55%**, beating
`cosine seed3` by **+0.10 pp**.

It does **not** beat the previous best single-seed output ensemble
(`blend075`, 70.70%), but it supports the same direction: output ensembles
with moderate cosine blend are the only PatchTST variants that can beat cosine
at all.

Next replication target:

`output ensemble top-5 PatchTST`, `blend=0.75-0.80`, `EMA=0`, seeds 1-5.
