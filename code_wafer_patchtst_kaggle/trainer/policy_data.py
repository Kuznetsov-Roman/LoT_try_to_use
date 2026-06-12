import math

import numpy as np
import torch
from torch.utils.data import Dataset


LR_GRID = np.asarray([
    0.0005, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
], dtype=np.float32)


def cosine_annealing_lr(epoch, total_epochs, initial_lr, min_lr=0.001):
    total_epochs = max(total_epochs, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    return float(min_lr + (initial_lr - min_lr) * cosine)


def parabolic_argmin(values):
    """Sub-grid argmin via 3-point parabolic interpolation.
    Returns a fractional index in ``[0, len(values) - 1]``.

    For monotonic boundary cases we just clamp to the endpoint integer index.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return float(int(np.argmin(values)))
    i = int(np.argmin(values))
    if i == 0 or i == values.size - 1:
        return float(i)
    y_left = values[i - 1]
    y_mid = values[i]
    y_right = values[i + 1]
    denom = (y_left - 2.0 * y_mid + y_right)
    if abs(denom) < 1e-12:
        return float(i)
    delta = 0.5 * (y_left - y_right) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(i + delta)


def fractional_index_to_lr(fractional_index, lr_grid=LR_GRID):
    """Linear interpolation of LR_GRID at a fractional index."""
    lr_grid = np.asarray(lr_grid, dtype=np.float64)
    frac = float(np.clip(fractional_index, 0.0, lr_grid.size - 1))
    lo = int(np.floor(frac))
    hi = min(lo + 1, lr_grid.size - 1)
    t = frac - lo
    return float((1.0 - t) * lr_grid[lo] + t * lr_grid[hi])


def softmax_weighted_lr(curve, temperature=0.1, lr_grid=LR_GRID):
    """Soft argmin: LR weighted by softmax(-curve / temperature) over LR_GRID.

    For sharp curves this approaches the hard argmin LR; for flat curves it averages
    toward the centre of mass of the low-loss region. More robust than parabolic
    interpolation when the curve is noisy or has a wide minimum.
    """
    curve = np.asarray(curve, dtype=np.float64)
    lr_grid = np.asarray(lr_grid, dtype=np.float64)
    if curve.size != lr_grid.size:
        raise ValueError(f"curve size {curve.size} != lr_grid size {lr_grid.size}")
    temperature = max(float(temperature), 1e-6)
    logits = -curve / temperature
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / max(weights.sum(), 1e-12)
    return float((weights * lr_grid).sum())


def smooth_by_trajectory(values, trajectory_ids, window=5):
    values = np.asarray(values, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids)
    result = values.copy()
    radius = max(window // 2, 0)
    for trajectory in np.unique(trajectory_ids):
        idx = np.where(trajectory_ids == trajectory)[0]
        seq = values[idx]
        smoothed = []
        for i in range(len(seq)):
            left = max(0, i - radius)
            right = min(len(seq), i + radius + 1)
            smoothed.append(float(np.median(seq[left:right])))
        result[idx] = np.asarray(smoothed, dtype=np.float32)
    return result


def append_time_features(features, epochs, total_epochs, initial_lr, min_lr=0.001, previous_lrs=None):
    features = np.asarray(features, dtype=np.float32)
    epochs = np.asarray(epochs, dtype=np.float32)
    norm_epoch = epochs / max(float(total_epochs), 1.0)
    cosine_lr = np.asarray(
        [cosine_annealing_lr(int(epoch), total_epochs, initial_lr, min_lr) for epoch in epochs],
        dtype=np.float32,
    )
    if previous_lrs is None:
        previous_lrs = cosine_lr
    previous_lrs = np.asarray(previous_lrs, dtype=np.float32)
    time_features = np.stack([norm_epoch, cosine_lr, previous_lrs], axis=1)
    return np.concatenate([features, time_features], axis=1)


class WindowedTrajectoryDataset(Dataset):
    def __init__(self, features, targets, trajectory_ids, window=15):
        self.x = []
        self.y = []
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        trajectory_ids = np.asarray(trajectory_ids)
        for trajectory in np.unique(trajectory_ids):
            idx = np.where(trajectory_ids == trajectory)[0]
            idx = idx[np.argsort(idx)]
            if len(idx) <= window:
                continue
            for pos in range(window, len(idx)):
                self.x.append(features[idx[pos - window : pos]])
                self.y.append(targets[idx[pos]])
        self.x = torch.tensor(np.asarray(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class WindowedCurveDataset(Dataset):
    """Each sample is a window of past features and a stack of future landscape curves.

    For a window ending at trajectory position ``pos``, the target is
    ``features[pos: pos + lookahead_n, :landscape_dim]`` — i.e. the landscape curves
    at the current epoch and the next ``lookahead_n - 1`` epochs.

    Windows without enough lookahead room are skipped.
    """

    def __init__(self, features, trajectory_ids, window=10, lookahead_n=2, landscape_dim=30):
        self.x = []
        self.y = []
        features = np.asarray(features, dtype=np.float32)
        trajectory_ids = np.asarray(trajectory_ids)
        if lookahead_n < 1:
            raise ValueError("lookahead_n must be >= 1")
        for trajectory in np.unique(trajectory_ids):
            idx = np.where(trajectory_ids == trajectory)[0]
            idx = idx[np.argsort(idx)]
            if len(idx) <= window + lookahead_n - 1:
                continue
            for pos in range(window, len(idx) - lookahead_n + 1):
                window_idx = idx[pos - window : pos]
                target_idx = idx[pos : pos + lookahead_n]
                self.x.append(features[window_idx])
                self.y.append(features[target_idx, :landscape_dim])
        if not self.x:
            raise ValueError(
                f"WindowedCurveDataset is empty (window={window}, lookahead_n={lookahead_n})"
            )
        self.x = torch.tensor(np.asarray(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
