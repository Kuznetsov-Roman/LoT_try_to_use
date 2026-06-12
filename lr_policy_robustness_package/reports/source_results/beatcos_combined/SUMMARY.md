# Beat-Cosine PatchTST Campaign Results

All queued runs finished with exit=0. Metrics are CIFAR-100 student test accuracy/loss.

## 60-epoch method summary

| method                                  |   n |   final_mean |   final_std |   best_mean |   best_std |   last10_mean |   last10_std |   last_epoch_min |   last_epoch_max |
|:----------------------------------------|----:|-------------:|------------:|------------:|-----------:|--------------:|-------------:|-----------------:|-----------------:|
| cosine                                  |   3 |       70.197 |       0.234 |      70.200 |      0.229 |        68.425 |        0.355 |               60 |               60 |
| patchtst_output_ens_top5_blend075       |   3 |       70.083 |       0.592 |      70.083 |      0.592 |        68.590 |        0.481 |               60 |               60 |
| patchtst_output_ens_top5_blend090_ema03 |   3 |       70.003 |       0.214 |      70.003 |      0.214 |        67.977 |        0.311 |               60 |               60 |
| patchtst_soup_top5_blend090             |   3 |       69.943 |       0.146 |      69.987 |      0.110 |        68.323 |        0.111 |               60 |               60 |
| patchtst_soup_top5_blend075_ema05       |   3 |       69.907 |       0.323 |      69.913 |      0.327 |        67.753 |        0.382 |               60 |               60 |
| mixed_arch_ens_blend075_ema03           |   3 |       69.893 |       0.775 |      69.903 |      0.760 |        68.158 |        0.456 |               60 |               60 |
| residual_log                            |   3 |       69.807 |       0.205 |      69.893 |      0.168 |        68.473 |        0.565 |               60 |               60 |
| mixed_arch_ens_blend090_ema05           |   3 |       69.703 |       0.297 |      69.853 |      0.180 |        67.680 |        0.315 |               60 |               60 |
| nbeats_fixed35                          |   1 |       69.640 |     nan     |      69.640 |    nan     |        67.341 |      nan     |               35 |               35 |
| patchtst_plain                          |   3 |       69.627 |       0.192 |      69.640 |      0.211 |        67.499 |        0.320 |               60 |               60 |
| patchtst_soup_top5_pure                 |   3 |       69.597 |       0.387 |      69.597 |      0.387 |        68.018 |        0.397 |               60 |               60 |
| dlinear_fixed35                         |   1 |       69.560 |     nan     |      69.560 |    nan     |        67.310 |      nan     |               35 |               35 |
| patchtst_ext50                          |   3 |       69.467 |       0.093 |      69.707 |      0.120 |        66.523 |        0.380 |               50 |               50 |

## Delta vs 60ep cosine reference

| method                                  |   n |   final_mean |   final_std |   last10_mean |   last10_std |   delta_vs_cosine_final |   delta_vs_cosine_last10 |
|:----------------------------------------|----:|-------------:|------------:|--------------:|-------------:|------------------------:|-------------------------:|
| patchtst_output_ens_top5_blend075       |   3 |       70.083 |       0.592 |        68.590 |        0.481 |                  -0.113 |                    0.166 |
| patchtst_output_ens_top5_blend090_ema03 |   3 |       70.003 |       0.214 |        67.977 |        0.311 |                  -0.193 |                   -0.448 |
| patchtst_soup_top5_blend090             |   3 |       69.943 |       0.146 |        68.323 |        0.111 |                  -0.253 |                   -0.102 |
| patchtst_soup_top5_blend075_ema05       |   3 |       69.907 |       0.323 |        67.753 |        0.382 |                  -0.290 |                   -0.672 |
| mixed_arch_ens_blend075_ema03           |   3 |       69.893 |       0.775 |        68.158 |        0.456 |                  -0.303 |                   -0.267 |
| residual_log                            |   3 |       69.807 |       0.205 |        68.473 |        0.565 |                  -0.390 |                    0.048 |
| mixed_arch_ens_blend090_ema05           |   3 |       69.703 |       0.297 |        67.680 |        0.315 |                  -0.493 |                   -0.745 |
| patchtst_plain                          |   3 |       69.627 |       0.192 |        67.499 |        0.320 |                  -0.570 |                   -0.925 |
| patchtst_soup_top5_pure                 |   3 |       69.597 |       0.387 |        68.018 |        0.397 |                  -0.600 |                   -0.406 |

## Per-run summary

| method                                  |   seed | source                   |   last_epoch |   final_acc |   final_loss |   best_acc |   best_epoch |   last10_acc |   last_student_lr |
|:----------------------------------------|-------:|:-------------------------|-------------:|------------:|-------------:|-----------:|-------------:|-------------:|------------------:|
| cosine                                  |      1 | full60_ref               |           60 |      69.990 |        1.125 |     70.000 |           59 |       68.171 |             0.001 |
| cosine                                  |      2 | full60_ref               |           60 |      70.150 |        1.124 |     70.150 |           60 |       68.273 |             0.001 |
| cosine                                  |      3 | full60_ref               |           60 |      70.450 |        1.113 |     70.450 |           60 |       68.830 |             0.001 |
| dlinear_fixed35                         |     20 | industry_followups       |           35 |      69.560 |        1.124 |     69.560 |           35 |       67.310 |             0.001 |
| mixed_arch_ens_blend075_ema03           |      1 | stars_beatcos            |           60 |      70.670 |        1.098 |     70.670 |           60 |       68.611 |             0.002 |
| mixed_arch_ens_blend075_ema03           |      2 | stars_beatcos            |           60 |      69.890 |        1.134 |     69.890 |           60 |       68.163 |             0.002 |
| mixed_arch_ens_blend075_ema03           |      3 | stars_beatcos            |           60 |      69.120 |        1.159 |     69.150 |           59 |       67.699 |             0.002 |
| mixed_arch_ens_blend090_ema05           |      1 | industry_beatcos         |           60 |      69.480 |        1.110 |     69.840 |           59 |       68.024 |             0.005 |
| mixed_arch_ens_blend090_ema05           |      2 | industry_beatcos         |           60 |      69.590 |        1.134 |     69.680 |           59 |       67.611 |             0.005 |
| mixed_arch_ens_blend090_ema05           |      3 | industry_beatcos         |           60 |      70.040 |        1.121 |     70.040 |           60 |       67.405 |             0.005 |
| nbeats_fixed35                          |     20 | industry_followups       |           35 |      69.640 |        1.129 |     69.640 |           35 |       67.341 |             0.001 |
| patchtst_ext50                          |     20 | industry_followups       |           50 |      69.530 |        1.122 |     69.830 |           35 |       66.102 |             0.003 |
| patchtst_ext50                          |     21 | industry_followups       |           50 |      69.510 |        1.108 |     69.700 |           35 |       66.629 |             0.003 |
| patchtst_ext50                          |     22 | industry_followups       |           50 |      69.360 |        1.134 |     69.590 |           36 |       66.839 |             0.003 |
| patchtst_output_ens_top5_blend075       |      1 | stars_beatcos            |           60 |      70.030 |        1.121 |     70.030 |           60 |       68.467 |             0.002 |
| patchtst_output_ens_top5_blend075       |      2 | stars_beatcos            |           60 |      69.520 |        1.137 |     69.520 |           60 |       68.183 |             0.002 |
| patchtst_output_ens_top5_blend075       |      3 | stars_beatcos            |           60 |      70.700 |        1.093 |     70.700 |           60 |       69.121 |             0.002 |
| patchtst_output_ens_top5_blend090_ema03 |      1 | industry_beatcos         |           60 |      69.760 |        1.128 |     69.760 |           60 |       67.883 |             0.003 |
| patchtst_output_ens_top5_blend090_ema03 |      2 | industry_beatcos         |           60 |      70.160 |        1.124 |     70.160 |           60 |       68.324 |             0.003 |
| patchtst_output_ens_top5_blend090_ema03 |      3 | industry_beatcos         |           60 |      70.090 |        1.136 |     70.090 |           60 |       67.723 |             0.003 |
| patchtst_plain                          |      1 | stars_patchtst60         |           60 |      69.660 |        1.127 |     69.660 |           60 |       67.559 |             0.002 |
| patchtst_plain                          |      2 | stars_patchtst60         |           60 |      69.800 |        1.111 |     69.840 |           59 |       67.785 |             0.002 |
| patchtst_plain                          |      3 | stars_patchtst60         |           60 |      69.420 |        1.130 |     69.420 |           60 |       67.154 |             0.002 |
| patchtst_soup_top5_blend075_ema05       |      1 | stars_beatcos            |           60 |      69.540 |        1.140 |     69.540 |           60 |       67.648 |             0.005 |
| patchtst_soup_top5_blend075_ema05       |      2 | stars_beatcos            |           60 |      70.030 |        1.112 |     70.050 |           59 |       67.434 |             0.005 |
| patchtst_soup_top5_blend075_ema05       |      3 | stars_beatcos            |           60 |      70.150 |        1.112 |     70.150 |           60 |       68.177 |             0.005 |
| patchtst_soup_top5_blend090             |      1 | industry_beatcos         |           60 |      69.810 |        1.116 |     69.880 |           59 |       68.366 |             0.002 |
| patchtst_soup_top5_blend090             |      2 | industry_beatcos         |           60 |      69.920 |        1.126 |     69.980 |           59 |       68.406 |             0.002 |
| patchtst_soup_top5_blend090             |      3 | industry_beatcos         |           60 |      70.100 |        1.112 |     70.100 |           60 |       68.197 |             0.002 |
| patchtst_soup_top5_pure                 |      1 | industry_patchtst_soup60 |           60 |      70.020 |        1.141 |     70.020 |           60 |       68.477 |             0.002 |
| patchtst_soup_top5_pure                 |      2 | industry_patchtst_soup60 |           60 |      69.260 |        1.119 |     69.260 |           60 |       67.779 |             0.002 |
| patchtst_soup_top5_pure                 |      3 | industry_patchtst_soup60 |           60 |      69.510 |        1.139 |     69.510 |           60 |       67.799 |             0.002 |
| residual_log                            |      1 | full60_ref               |           60 |      69.580 |        1.145 |     69.700 |           59 |       68.536 |             0.001 |
| residual_log                            |      2 | full60_ref               |           60 |      69.860 |        1.125 |     70.000 |           59 |       69.004 |             0.001 |
| residual_log                            |      3 | full60_ref               |           60 |      69.980 |        1.131 |     69.980 |           60 |       67.879 |             0.003 |

## Offline MSE ranking (policy pretraining)

| log                                                                 |   best_test_mse |
|:--------------------------------------------------------------------|----------------:|
| results\beatcos_stars\per_run\patchtst_offline_seed0.log            |        0.257802 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed0.log |        0.257802 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed2.log |        0.293661 |
| results\beatcos_stars\per_run\patchtst_offline_seed2.log            |        0.293661 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed5.log |        0.299965 |
| results\beatcos_stars\per_run\patchtst_offline_seed5.log            |        0.299965 |
| results\beatcos_stars\per_run\tcn_offline_seed0.log                 |        0.313180 |
| results\beatcos_stars\per_run\patchtst_offline_seed7.log            |        0.313291 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed7.log |        0.313291 |
| results\beatcos_stars\per_run\patchtst_offline_seed4.log            |        0.313638 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed4.log |        0.313638 |
| results\beatcos_stars\per_run\patchtst_offline_seed3.log            |        0.319581 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed3.log |        0.319581 |
| results\beatcos_stars\per_run\nbeats_offline_seed0.log              |        0.321177 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed6.log |        0.329619 |
| results\beatcos_stars\per_run\patchtst_offline_seed6.log            |        0.329619 |
| results\patchtst_soup60_industry\per_run\patchtst_offline_seed1.log |        0.331504 |
| results\beatcos_stars\per_run\patchtst_offline_seed1.log            |        0.331504 |
| results\beatcos_stars\per_run\dlinear_offline_seed0.log             |        0.361824 |

## Figures

- `results/figures/beatcos_top_60ep.png`
- `results/figures/patchtst_ext50_final.png`
