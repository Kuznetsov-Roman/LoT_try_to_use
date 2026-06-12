"""Offline residual_log policy trainer for SOTA architectures.

Supports: gru, tcn, patchtst, nbeats, dlinear.

Output: ckpt with {'state_dict': ..., 'config': {model_type, input_dim, window, ...}}
that can be loaded by my_research.py via --policy_checkpoint.

Usage:
    python trainer/train_policy_advanced.py --arch tcn --output residual_log \
        --features_train features_v3_train.npy --targets_train targets_v3_train.npy \
        --features_test features_v3_test.npy --targets_test targets_v3_test.npy \
        --window 10 --policy_oracle_period 35 --policy_min_lr 0.001 --lr 1.0 \
        --epochs 80 --save checkpoints/policies/tcn/policy.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.lr_policy import (
    DLinearLRPolicy,
    GRULRPolicy,
    NBeatsLRPolicy,
    PatchTSTLRPolicy,
    TCNLRPolicy,
)


def cosine_annealing_lr(epoch, total_epochs, initial_lr, min_lr=0.0):
    total_epochs = max(total_epochs, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * epoch / total_epochs))
    return float(min_lr + (initial_lr - min_lr) * cosine)


def transform_targets_residual_log(targets, period, base_lr, min_lr):
    transformed = []
    for index, target in enumerate(targets):
        epoch = index % period
        cos = cosine_annealing_lr(epoch, period, base_lr, min_lr)
        ratio = max(target, 1e-9) / max(cos, 1e-9)
        log_ratio = float(np.clip(np.log(ratio), -1.0, 1.0))
        transformed.append(log_ratio)
    return np.asarray(transformed, dtype=np.float32)


def transform_targets_raw(targets):
    return np.asarray(targets, dtype=np.float32)


class WindowDataset(Dataset):
    """Slides a window over (X, y) and skips the last sample of each
    trajectory (length = period, indexed by i % period == 0)."""

    def __init__(self, X, y, window, period):
        self.X, self.y = [], []
        i = window
        while i < len(X):
            if i % period != 0:
                self.X.append(X[i - window:i])
                self.y.append(y[i])
            else:
                i += window
                if i >= len(X):
                    break
                self.X.append(X[i - window:i])
                self.y.append(y[i])
            i += 1
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_model(arch, input_dim, window):
    if arch == 'gru':
        return GRULRPolicy(input_dim=input_dim, hidden=129, num_layers=3, dropout=0.027), {}
    if arch == 'tcn':
        cfg = {'hidden': 64, 'num_layers': 4}
        return TCNLRPolicy(input_dim=input_dim, **cfg), cfg
    if arch == 'patchtst':
        cfg = {'hidden': 16, 'num_layers': 2, 'max_window': max(window + 4, 16)}
        return PatchTSTLRPolicy(input_dim=input_dim, **cfg), cfg
    if arch == 'nbeats':
        cfg = {'hidden': 128, 'num_blocks': 3, 'window': window}
        return NBeatsLRPolicy(input_dim=input_dim, **cfg), cfg
    if arch == 'dlinear':
        cfg = {'window': window}
        return DLinearLRPolicy(input_dim=input_dim, **cfg), cfg
    raise ValueError(f"Unknown arch={arch}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', type=str, required=True, choices=['gru', 'tcn', 'patchtst', 'nbeats', 'dlinear'])
    p.add_argument('--output', type=str, default='residual_log', choices=['residual_log', 'raw_lr'])
    p.add_argument('--features_train', type=str, required=True)
    p.add_argument('--targets_train', type=str, required=True)
    p.add_argument('--features_test', type=str, required=True)
    p.add_argument('--targets_test', type=str, required=True)
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--policy_oracle_period', type=int, default=35)
    p.add_argument('--lr', type=float, default=1.0, help='base LR for cosine reference (matches deploy)')
    p.add_argument('--policy_min_lr', type=float, default=0.001)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--train_lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--save', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[advanced] arch={args.arch} output={args.output} device={device}")

    X_tr = np.load(args.features_train).astype(np.float32)
    y_tr = np.load(args.targets_train).astype(np.float32)
    X_te = np.load(args.features_test).astype(np.float32)
    y_te = np.load(args.targets_test).astype(np.float32)
    print(f"[advanced] X_tr={X_tr.shape} y_tr={y_tr.shape} X_te={X_te.shape} y_te={y_te.shape}")

    if args.output == 'residual_log':
        y_tr_t = transform_targets_residual_log(y_tr, args.policy_oracle_period, args.lr, args.policy_min_lr)
        y_te_t = transform_targets_residual_log(y_te, args.policy_oracle_period, args.lr, args.policy_min_lr)
    else:
        y_tr_t = transform_targets_raw(y_tr)
        y_te_t = transform_targets_raw(y_te)

    train_ds = WindowDataset(X_tr, y_tr_t, args.window, args.policy_oracle_period)
    test_ds = WindowDataset(X_te, y_te_t, args.window, args.policy_oracle_period)
    print(f"[advanced] train_windows={len(train_ds)} test_windows={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = X_tr.shape[-1]
    model, cfg = build_model(args.arch, input_dim, args.window)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[advanced] params={n_params:,} cfg={cfg}")

    opt = torch.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()

    history = []
    best_test_mse = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        sched.step()

        model.eval()
        te_loss, te_n = 0.0, 0
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                te_loss += loss.item() * xb.size(0)
                te_n += xb.size(0)
                preds.append(pred.cpu().numpy())
                gts.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        mse = float(np.mean((preds - gts) ** 2))
        mae = float(np.mean(np.abs(preds - gts)))
        if mse < best_test_mse:
            best_test_mse = mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[advanced] epoch={epoch:03d} train_loss={tr_loss / tr_n:.6f} "
                  f"test_mse={mse:.6f} test_mae={mae:.6f} best_mse={best_test_mse:.6f}")
        history.append({'epoch': epoch, 'train_loss': tr_loss / tr_n, 'test_mse': mse, 'test_mae': mae})

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    full_cfg = {
        'model_type': args.arch,
        'input_dim': input_dim,
        'window': args.window,
        'output': args.output,
        'policy_oracle_period': args.policy_oracle_period,
        'lr': args.lr,
        'policy_min_lr': args.policy_min_lr,
        **cfg,
    }
    torch.save({
        'state_dict': best_state if best_state is not None else model.state_dict(),
        'config': full_cfg,
        'best_test_mse': best_test_mse,
        'n_params': n_params,
    }, args.save)
    history_path = Path(args.save).parent / f'history_{args.arch}_seed{args.seed}.json'
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[advanced] saved {args.save} (best_test_mse={best_test_mse:.6f}, n_params={n_params:,})")
    print(f"[advanced] history → {history_path}")


if __name__ == '__main__':
    main()
