# Six Hour LR Policy Follow-up

Generated: 2026-05-11T13:01:39

Goal: test whether the modular policy can beat or match cosine after controlling the LR schedule, and whether GRU improves when forced onto the smoothed multiplier target.

Reference from overnight cosine baseline: student_acc mean 69.813, std 0.332.

## Variant Summary

- `gru_smooth_cap1p0`: n=3 final_acc=69.437±0.159 loss=1.0846 acc20=40.970 acc30=55.557 lr_min=0.0050 lr_max=1.0000 lr_last_mean=0.0050
- `modular_blend05_cap1p2`: n=3 final_acc=69.440±0.185 loss=1.0894 acc20=43.817 acc30=60.770 lr_min=0.0030 lr_max=0.9980 lr_last_mean=0.0030
- `modular_cap0p8`: n=3 final_acc=69.360±0.594 loss=1.1031 acc20=45.110 acc30=56.220 lr_min=0.0049 lr_max=0.9980 lr_last_mean=0.0050
- `modular_cap1p0`: n=3 final_acc=69.330±0.406 loss=1.0952 acc20=40.080 acc30=57.727 lr_min=0.0049 lr_max=1.0000 lr_last_mean=0.0050

## Runs

- `gru_smooth_cap1p0_seed60`: final_acc=69.300 final_loss=1.0914 acc20=37.640 acc30=57.410 lr_min=0.0050 lr_max=1.0000 lr_last=0.0050
- `gru_smooth_cap1p0_seed61`: final_acc=69.350 final_loss=1.0881 acc20=42.700 acc30=52.910 lr_min=0.0050 lr_max=1.0000 lr_last=0.0050
- `gru_smooth_cap1p0_seed62`: final_acc=69.660 final_loss=1.0742 acc20=42.570 acc30=56.350 lr_min=0.0050 lr_max=1.0000 lr_last=0.0050
- `modular_blend05_cap1p2_seed57`: final_acc=69.330 final_loss=1.0953 acc20=47.790 acc30=60.640 lr_min=0.0030 lr_max=0.9980 lr_last=0.0030
- `modular_blend05_cap1p2_seed58`: final_acc=69.700 final_loss=1.0704 acc20=40.210 acc30=60.080 lr_min=0.0030 lr_max=0.9980 lr_last=0.0030
- `modular_blend05_cap1p2_seed59`: final_acc=69.290 final_loss=1.1026 acc20=43.450 acc30=61.590 lr_min=0.0030 lr_max=0.9980 lr_last=0.0030
- `modular_cap0p8_seed54`: final_acc=69.930 final_loss=1.0994 acc20=44.760 acc30=56.020 lr_min=0.0049 lr_max=0.9980 lr_last=0.0049
- `modular_cap0p8_seed55`: final_acc=69.610 final_loss=1.0977 acc20=43.880 acc30=57.460 lr_min=0.0050 lr_max=0.9980 lr_last=0.0050
- `modular_cap0p8_seed56`: final_acc=68.540 final_loss=1.1121 acc20=46.690 acc30=55.180 lr_min=0.0049 lr_max=0.9980 lr_last=0.0049
- `modular_cap1p0_seed51`: final_acc=69.770 final_loss=1.0776 acc20=42.330 acc30=55.520 lr_min=0.0050 lr_max=1.0000 lr_last=0.0050
- `modular_cap1p0_seed52`: final_acc=69.430 final_loss=1.0838 acc20=38.490 acc30=59.430 lr_min=0.0050 lr_max=1.0000 lr_last=0.0050
- `modular_cap1p0_seed53`: final_acc=68.790 final_loss=1.1243 acc20=39.420 acc30=58.230 lr_min=0.0049 lr_max=1.0000 lr_last=0.0049

## Preliminary Recommendation

Best completed variant: `modular_blend05_cap1p2` with mean student_acc=69.440. If it is still below cosine, the next target should be multi-step oracle rather than more architecture tuning.
