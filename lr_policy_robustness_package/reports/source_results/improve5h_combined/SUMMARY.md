# improve5h combined results

Generated: 2026-05-14 18:21:48 UTC

Baselines: cosine seed3 final acc = 70.45; previous best PatchTST output ensemble = 70.70.

## Verdict
No improve5h variant beats cosine. Best final result is 70.29, -0.16 pp vs cosine.

## Ranking by final student test accuracy

| rank | server | variant | final acc | final loss | best acc | best epoch | delta vs cosine | delta vs prev best |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | stars | `outens_blend0800_warm05_seed3_60ep` | 70.29 | 1.123 | 70.29 | 60 | -0.16 | -0.41 |
| 2 | industry | `outens_top7_blend0800_cosafter45_seed3_60ep` | 70.11 | 1.134 | 70.11 | 60 | -0.34 | -0.59 |
| 3 | industry | `outens_top7_blend0750_warm10_seed3_60ep` | 70.04 | 1.101 | 70.04 | 60 | -0.41 | -0.66 |
| 4 | stars | `outens_blend0725_warm10_seed3_60ep` | 69.96 | 1.128 | 69.98 | 59 | -0.49 | -0.74 |
| 5 | industry | `outens_top5_blend0825_warm10_seed3_60ep` | 69.86 | 1.125 | 69.99 | 58 | -0.59 | -0.84 |
| 6 | industry | `outens_top5_blend0775_cosafter50_seed3_60ep` | 69.86 | 1.120 | 69.86 | 60 | -0.59 | -0.84 |
| 7 | stars | `outens_blend0750_warm10_cosafter45_seed3_60ep` | 69.78 | 1.127 | 69.86 | 59 | -0.67 | -0.92 |
| 8 | stars | `outens_blend0775_warm10_seed3_60ep` | 69.69 | 1.118 | 69.69 | 60 | -0.76 | -1.01 |
| 9 | stars | `outens_blend0800_warm10_cosafter45_seed3_60ep` | 69.62 | 1.116 | 69.62 | 60 | -0.83 | -1.08 |
| 10 | industry | `outens_top3_blend0800_cosafter50_seed3_60ep` | 69.60 | 1.144 | 69.60 | 60 | -0.85 | -1.10 |
| 11 | industry | `outens_top3_blend0750_warm10_seed3_60ep` | 69.57 | 1.125 | 69.57 | 60 | -0.88 | -1.13 |
| 12 | stars | `outens_blend0750_warm05_seed3_60ep` | 69.53 | 1.147 | 69.53 | 60 | -0.92 | -1.17 |

## Pattern notes
- Best variant: `outens_blend0800_warm05_seed3_60ep` on stars with top-5 ensemble, blend=0.800, warmup=5, cosine_after=0.
- Late cosine handoff variants did not help in this run; they cluster below the best no-handoff variants.
- Increasing ensemble size to top-7 hurt badly; top-3/top-5 are safer.
- The sweep found no new configuration that improves the reviewer artifact result; keep the existing artifact as the headline result.

CSV: `d:\lr-policy\results\improve5h_combined\summary.csv`