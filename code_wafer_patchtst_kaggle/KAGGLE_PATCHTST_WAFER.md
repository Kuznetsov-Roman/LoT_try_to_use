# Kaggle: wafer + LoT + PatchTST LR policy

This project uses PatchTST as the LR-policy model (`--policy_model_type patchtst` / `--arch patchtst`), not as the wafer classifier backbone. The wafer classifier remains the LoT teacher/student PreResNet pipeline.

## Expected Kaggle inputs

- `code_wafer_patchtst_kaggle.zip` uploaded as a Kaggle Dataset or Notebook file.
- `Wafer_Map_Datasets.npz` uploaded as another Kaggle Dataset. The loader expects `arr_0` images and `arr_1` multilabel targets.
- Optional, for PatchTST dynamic scheduler: offline policy arrays
  - `features_wafer_v3_train.npy`
  - `targets_wafer_v3_train.npy`
  - `features_wafer_v3_test.npy`
  - `targets_wafer_v3_test.npy`

## 1. Unpack

```bash
unzip -q /kaggle/input/lot-wafer-code/code_wafer_patchtst_kaggle.zip -d /kaggle/working/
cd /kaggle/working/code_wafer_patchtst_kaggle
```

## 2. Smoke-test wafer pipeline without policy checkpoint

```bash
python -m trainer.my_research \
  --dataset wafer \
  --wafer_npz /kaggle/input/wafer-map-datasets/Wafer_Map_Datasets.npz \
  --scheduler cosine \
  --epochs 1 \
  --max_train_batches 5 \
  --batch_size 64 \
  --num_workers 2 \
  --wafer_resize 56 \
  --depth_list 110_20 \
  --policy_probe_depth 20 \
  --student_steps_ratio 1 \
  --exp_name wafer_smoke
```

## 3. Train PatchTST LR-policy from offline features

Use this only if you have feature/target `.npy` files. If the feature dimension is 46, use the normal corrected wafer mode. If it is 60, this is the old legacy wafer feature format and deploy must include `--wafer_legacy_feature_slice`.

```bash
python -m trainer.train_policy_advanced \
  --arch patchtst \
  --output residual_log \
  --features_train /kaggle/input/wafer-policy/features_wafer_v3_train.npy \
  --targets_train  /kaggle/input/wafer-policy/targets_wafer_v3_train.npy \
  --features_test  /kaggle/input/wafer-policy/features_wafer_v3_test.npy \
  --targets_test   /kaggle/input/wafer-policy/targets_wafer_v3_test.npy \
  --window 15 \
  --policy_oracle_period 180 \
  --lr 1.0 \
  --policy_min_lr 0.001 \
  --epochs 80 \
  --batch_size 32 \
  --save /kaggle/working/policies/wafer_patchtst.pt
```

## 4. Run wafer pipeline with PatchTST dynamic LR-policy

```bash
python -m trainer.my_research \
  --dataset wafer \
  --wafer_npz /kaggle/input/wafer-map-datasets/Wafer_Map_Datasets.npz \
  --scheduler dynamic \
  --policy_checkpoint /kaggle/working/policies/wafer_patchtst.pt \
  --policy_model_type patchtst \
  --policy_output residual_log \
  --policy_window 15 \
  --policy_warmup_epochs 15 \
  --policy_min_lr 0.001 \
  --policy_max_lr 2.5 \
  --epochs 180 \
  --batch_size 64 \
  --num_workers 2 \
  --wafer_resize 56 \
  --depth_list 110_20 \
  --policy_probe_depth 20 \
  --student_steps_ratio 1 \
  --exp_name wafer_patchtst_dynamic \
  --snapshot_dir /kaggle/working/snapshots \
  --save /kaggle/working/ckpt/wafer_patchtst
```

If your PatchTST policy was trained on old 60-dimensional wafer features, add this to the deploy command:

```bash
  --wafer_legacy_feature_slice
```

## Notes

- `--scheduler dynamic --policy_model_type patchtst` is not enough by itself. Use `--policy_checkpoint`; otherwise the legacy in-script trainer builds a GRU policy.
- For a quick Kaggle run, reduce `--epochs`, `--max_train_batches`, and set `--student_steps_ratio 1`.
- The per-epoch LR-landscape probe is expensive: each evaluation makes 30 one-step model copies.
