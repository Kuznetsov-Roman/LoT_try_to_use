# Wafer-ready variant

This folder is based on `code(1).7z` and adds a multilabel wafer path while keeping CIFAR modes intact.

## Expected wafer data

Pass an `.npz` file with:

- `arr_0`: wafer map images, e.g. `[N,H,W]`, `[N,1,H,W]`, `[N,H,W,1]`, or RGB variants;
- `arr_1`: multilabel defect targets, normally 8 columns:
  `Center, Donut, Edge_Loc, Edge_Ring, Loc, Near_Full, Scratch, Random`.

## First smoke run

Use a non-policy scheduler first, because `--scheduler dynamic` needs wafer LR-policy feature/target files or a compatible checkpoint.

```bash
python -m trainer.my_research \
  --dataset wafer \
  --wafer_npz /path/to/Wafer_Map_Datasets.npz \
  --scheduler cosine \
  --epochs 1 \
  --max_train_batches 10 \
  --batch_size 64 \
  --num_workers 0 \
  --depth_list 110_20 \
  --policy_probe_depth 20 \
  --exp_name wafer_smoke
```

## Dynamic policy notes

Correct wafer feature length is:

```text
30 LR-probe values + 8 mean probabilities + 8 std probabilities = 46
```

So a newly trained wafer policy should use 46-dimensional features.

The older exploratory wafer script accidentally used `student_concat[:, 1:]` after `hstack(labels, probs)`, which leaked 7 label columns into the feature vector and produced:

```text
30 + 15 + 15 = 60 dims
```

For compatibility with old `features_wafer_v3.npy` / old checkpoints, run with:

```bash
--wafer_legacy_feature_slice --policy_input_dim 60
```

Otherwise keep the default corrected 46-dimensional feature generation.
