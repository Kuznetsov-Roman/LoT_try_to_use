# SOTA LR comparison final analysis

Status: all intended SOTA comparator sweeps completed and parsed: 27 runs total (AdaLRS, bandit EXP3/UCB, Hypergrad-HB), 3 seeds per variant.

## Verdict

FAIL. None of the tested external trainable/adaptive LR methods beat cosine seed3=70.45, PatchTST headline=70.70, or even cosine 3-seed mean=70.197. Best final run is `adalrs_default_seed1_60ep` at 68.22 (-2.23 pp vs cosine seed3). Best method mean is `adalrs/default` at 67.41 ± 0.82 (-2.79 pp vs cosine mean).

## Method ranking

| method | variant | n | final mean | final std | last10 mean | delta vs cosine mean |
|---|---|---:|---:|---:|---:|---:|
| adalrs | `default` | 3 | 67.41 | 0.82 | 66.36 | -2.79 |
| adalrs | `aggressive` | 3 | 65.99 | 2.29 | 65.52 | -4.21 |
| adalrs | `narrow_safe` | 3 | 54.73 | 4.92 | 50.28 | -15.46 |
| bandit_exp3 | `safe` | 3 | 53.61 | 17.94 | 48.37 | -16.58 |
| bandit_exp3 | `fast` | 3 | 52.09 | 6.49 | 44.70 | -18.10 |
| hypergrad_hb | `fast` | 3 | 42.67 | 8.21 | 41.35 | -27.53 |
| bandit_ucb | `ucb` | 3 | 40.88 | 23.43 | 40.60 | -29.31 |
| hypergrad_hb | `safe` | 3 | 39.75 | 3.95 | 42.59 | -30.45 |
| hypergrad_hb | `smooth` | 3 | 34.09 | 8.06 | 40.17 | -36.11 |

## Diagnosis

AdaLRS is the least bad, but it is still not competitive. Its one-step LR probe repeatedly prefers too-high LR candidates, then the safeguard clamps after loss damage is already visible. It sometimes reaches best epochs around 69.2-69.35, but final accuracy falls to 66-68.

Bandit methods are unstable because the reward is a noisy epoch-to-epoch loss delta and the action space spans the full LR grid. EXP3/UCB often jump to extreme LR arms, causing late collapse and very poor last-10 averages.

Hypergrad-HB is a clear failure in this implementation. It never gets close to the 60-epoch cosine regime: best variant mean is 42.67, with high variance. Batch-level gradient-dot-product hypergrad is too local/noisy for this LoT distillation setup and destabilizes the student LR.

## Recommendation

Do not spend more GPU on these controller families in their current form. For the paper, use them as negative external SOTA/adaptive-LR comparators: our PatchTST output-ensemble result remains the only trainable LR policy that reaches the cosine regime and occasionally beats it on seed3. If continuing, the only plausible controller direction is a trust-region version: LR changes bounded relative to cosine, no full-grid jumps, and decisions based on smoothed multi-epoch validation signals rather than one-step probes or one-epoch rewards.
