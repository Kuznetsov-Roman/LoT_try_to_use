# Package Robust12h Summary

## Paired deltas
| experiment | perturb_key | method | cosine_final_mean | method_final_mean | delta_pp | n_cosine | n_method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shock | shock_lr=0.10 | patchtst | 69.45 | 70.168 | 0.718 | 4 | 4 |
| noshock | noshock | patchtst | 69.898 | 69.882 | -0.015 | 4 | 4 |
| innoise | input_noise=0.05 | patchtst | 65.98 | 65.9 | -0.08 | 4 | 4 |
| O6_long90 | long90 | patchtst | 70.1 | 69.988 | -0.112 | 5 | 5 |
| shock | shock_lr=0.10 | residgru | 69.45 | 69.31 | -0.14 | 4 | 4 |
| shock | shock_lr=1.00 | patchtst | 69.898 | 69.688 | -0.21 | 4 | 4 |
| O5_noise_onset_e15 | label=0.20@e15 | patchtst | 64.502 | 64.255 | -0.248 | 4 | 4 |
| lblnoise | label_noise=0.10 | patchtst | 66.598 | 66.34 | -0.257 | 4 | 4 |
| innoise | input_noise=0.15 | patchtst | 57.33 | 56.988 | -0.343 | 4 | 4 |
| shock | shock_lr=0.50 | patchtst | 70.018 | 69.529 | -0.489 | 4 | 8 |
| lblnoise | label_noise=0.30 | patchtst | 60.808 | 60.29 | -0.517 | 4 | 4 |
| lblnoise | label_noise=0.20 | patchtst | 63.71 | 63.185 | -0.525 | 4 | 4 |
| shock | shock_lr=0.50 | residgru | 70.018 | 69.27 | -0.748 | 4 | 4 |
| compound | label=0.20+shock_lr=1.0@e15 | patchtst | 64.085 | 63.105 | -0.98 | 4 | 4 |
| lblnoise | label_noise=0.10 | residgru | 66.598 | 65.452 | -1.145 | 4 | 4 |
| lblnoise | label_noise=0.20 | residgru | 63.71 | 62.39 | -1.32 | 4 | 4 |
| compound | label=0.20+shock_lr=1.0@e15 | residgru | 64.085 | 62.303 | -1.782 | 4 | 3 |
| lblnoise | label_noise=0.30 | residgru | 60.808 | 58.978 | -1.83 | 4 | 4 |
| shock | shock_lr=2.00 | residgru | 62.14 | 60.195 | -1.945 | 4 | 4 |
| shock | shock_lr=2.00 | patchtst | 62.14 | 59.772 | -2.368 | 4 | 4 |
| shock | shock_lr=1.00 | residgru | 69.898 | 62.828 | -7.07 | 4 | 4 |

## Winning shock01 seeds
| seed | cosine | patchtst | patchtst_minus_cosine |
| --- | --- | --- | --- |
| 1.0 | 69.94 | 69.51 | -0.43 |
| 2.0 | 69.58 | 70.3 | 0.72 |
| 3.0 | 69.71 | 70.44 | 0.73 |
| 4.0 | 68.57 | 70.42 | 1.85 |