import logging
import traceback
import torch
import sys
import copy
import os
import time
from torch import nn
import torch.nn.functional as F
#import wandb
import configparser
import argparse
import json
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import f1_score
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import get_torch_dataset
from model.preresnet import PreResNet
from model.lr_policy import (
    AttentionModularLRPolicy,
    CurveLRPolicy,
    DLinearLRPolicy,
    GRULRPolicy,
    ModularLRPolicy,
    NBeatsLRPolicy,
    PatchTSTLRPolicy,
    TCNLRPolicy,
)
from trainer.policy_data import (
    LR_GRID,
    append_time_features,
    fractional_index_to_lr,
    parabolic_argmin,
    softmax_weighted_lr,
)


class MyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, lrs, last_epoch=-1):
        self.lrs = lrs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        
        if epoch >= len(self.lrs):
            return [self.lrs[-1] for _ in self.optimizer.param_groups]
        
        return [self.lrs[epoch] for _ in self.optimizer.param_groups]
    

class DynamicScheduler(_LRScheduler):
    def __init__(self, optimizer, init_lr=0.01, last_epoch=-1):
        self.current_lr = init_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.current_lr for _ in self.optimizer.param_groups]

    def set_lr(self, lr):
        self.current_lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr
        self._last_lr = [self.current_lr for _ in self.optimizer.param_groups]


class HypergradScheduler(_LRScheduler):
    """Per-epoch finite-difference hypergradient on top of the 30-dim probe.

    After each epoch we already evaluate the loss at all 30 LR points (LR_GRID).
    This scheduler approximates the Baydin et al. 2017 hypergradient by a
    central-difference of that probe at the closest grid point to the current
    LR, then takes a multiplicative gradient-descent step in log-LR space:

        log lr_{t+1} = log lr_t - beta * sign(dL/d log lr) * |lr_t * dL/dlr|.

    No offline pretraining, no policy network, no oracle. Uses only what we
    already compute (the probe).
    """

    def __init__(self, optimizer, init_lr=0.01, beta=0.05, lr_min=1e-3,
                 lr_max=1.5, last_epoch=-1):
        self.current_lr = float(init_lr)
        self.beta = float(beta)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.current_lr for _ in self.optimizer.param_groups]

    def set_lr(self, lr):
        self.current_lr = float(np.clip(lr, self.lr_min, self.lr_max))
        for group in self.optimizer.param_groups:
            group['lr'] = self.current_lr
        self._last_lr = [self.current_lr for _ in self.optimizer.param_groups]

    def hypergrad_step(self, probe_curve, lr_grid):
        """Apply one hypergradient step using the probe at LR_GRID."""
        probe_curve = np.asarray(probe_curve, dtype=np.float64)
        lr_grid = np.asarray(lr_grid, dtype=np.float64)
        idx = int(np.argmin(np.abs(lr_grid - self.current_lr)))
        idx = max(1, min(len(lr_grid) - 2, idx))
        d_loss = (probe_curve[idx + 1] - probe_curve[idx - 1]) / (lr_grid[idx + 1] - lr_grid[idx - 1] + 1e-12)
        # Multiplicative update in log-LR for scale invariance.
        log_lr = float(np.log(max(self.current_lr, 1e-9)))
        new_log_lr = log_lr - self.beta * np.sign(d_loss) * float(np.abs(self.current_lr * d_loss))
        new_lr = float(np.exp(new_log_lr))
        self.set_lr(new_lr)
        return float(d_loss)


def _nearest_grid_index(lr_grid, lr):
    lr_grid = np.asarray(lr_grid, dtype=np.float64)
    return int(np.argmin(np.abs(lr_grid - float(lr))))


class AdaLRSScheduler(DynamicScheduler):
    """Loss-guided LR search over the existing probe grid.

    The scheduler compares the best local probe loss with the cosine-reference
    probe loss. It accepts the local winner only when the probe improvement is
    larger than a small noise margin; otherwise it falls back to a conservative
    fraction of cosine. This is intentionally cheap: it reuses the 30 LR probes
    already computed by evaluate().
    """

    def __init__(
        self,
        optimizer,
        init_lr=0.01,
        lr_min=1e-3,
        lr_max=1.5,
        alpha=0.5,
        beta=1.5,
        margin=0.002,
        clamp=0.7,
        last_epoch=-1,
    ):
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.margin = float(margin)
        self.clamp = float(clamp)
        self.last_decision = {}
        super().__init__(optimizer, init_lr=init_lr, last_epoch=last_epoch)

    def set_lr(self, lr):
        super().set_lr(float(np.clip(lr, self.lr_min, self.lr_max)))

    def adalrs_step(self, epoch, probe_curve, lr_grid, cosine_lr, losses, warmup_epochs, bad_epochs):
        probe_curve = np.asarray(probe_curve, dtype=np.float64)
        lr_grid = np.asarray(lr_grid, dtype=np.float64)
        cosine_lr = float(np.clip(cosine_lr, self.lr_min, self.lr_max))

        if epoch <= warmup_epochs:
            self.set_lr(cosine_lr)
            decision = "warmup_cosine"
            best_lr = cosine_lr
            best_loss = float("nan")
            cosine_loss = float("nan")
            improvement = 0.0
        else:
            lo = max(self.lr_min, cosine_lr * self.alpha)
            hi = min(self.lr_max, cosine_lr * self.beta)
            mask = (lr_grid >= lo) & (lr_grid <= hi)
            if not np.any(mask):
                mask = np.ones_like(lr_grid, dtype=bool)
            candidate_indices = np.where(mask)[0]
            best_idx = int(candidate_indices[np.argmin(probe_curve[candidate_indices])])
            cosine_idx = _nearest_grid_index(lr_grid, cosine_lr)
            best_lr = float(lr_grid[best_idx])
            best_loss = float(probe_curve[best_idx])
            cosine_loss = float(probe_curve[cosine_idx])
            improvement = (cosine_loss - best_loss) / max(abs(cosine_loss), 1e-12)

            need = max(2, int(bad_epochs) + 1)
            rising = False
            if len(losses) >= need:
                tail = losses[-need:]
                rising = all(tail[i + 1] > tail[i] for i in range(len(tail) - 1))

            if rising:
                next_lr = self.clamp * cosine_lr
                decision = "rising_loss_clamp"
            elif improvement > self.margin:
                next_lr = best_lr
                decision = "accept_probe"
            else:
                next_lr = self.clamp * cosine_lr
                decision = "fallback_cosine_clamp"
            self.set_lr(next_lr)

        self.last_decision = {
            "decision": decision,
            "best_lr": float(best_lr),
            "best_loss": float(best_loss),
            "cosine_loss": float(cosine_loss),
            "improvement": float(improvement),
            "next_lr": float(self.current_lr),
        }
        return self.last_decision


class BanditLRScheduler(DynamicScheduler):
    """Non-stationary bandit over LR_GRID arms."""

    def __init__(
        self,
        optimizer,
        lr_grid,
        mode="exp3",
        init_lr=0.01,
        eta=0.07,
        gamma=0.10,
        lr_min=1e-3,
        lr_max=1.5,
        last_epoch=-1,
    ):
        self.lr_grid = np.asarray(lr_grid, dtype=np.float64)
        valid = (self.lr_grid >= float(lr_min)) & (self.lr_grid <= float(lr_max))
        self.arm_indices = np.where(valid)[0]
        if len(self.arm_indices) == 0:
            self.arm_indices = np.arange(len(self.lr_grid))
        self.mode = mode
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.weights = np.ones(len(self.arm_indices), dtype=np.float64)
        self.counts = np.zeros(len(self.arm_indices), dtype=np.float64)
        self.values = np.zeros(len(self.arm_indices), dtype=np.float64)
        self.t = 0
        self.pending_arm = None
        self.prev_loss = None
        self.last_decision = {}
        super().__init__(optimizer, init_lr=init_lr, last_epoch=last_epoch)

    def set_lr(self, lr):
        super().set_lr(float(np.clip(lr, self.lr_min, self.lr_max)))

    def _arm_position_for_lr(self, lr):
        absolute_idx = _nearest_grid_index(self.lr_grid[self.arm_indices], lr)
        return int(absolute_idx)

    def _probabilities(self):
        weights = np.maximum(self.weights, 1e-12)
        probs = weights / weights.sum()
        if self.mode == "exp3":
            probs = (1.0 - self.gamma) * probs + self.gamma / len(probs)
        return probs

    def bandit_step(self, epoch, current_loss, cosine_lr, warmup_epochs):
        current_loss = float(current_loss)
        reward = 0.0
        if self.prev_loss is not None and self.pending_arm is not None:
            # Positive reward means the last selected LR reduced validation loss.
            reward = float(np.clip(self.prev_loss - current_loss, -1.0, 1.0))
            arm = int(self.pending_arm)
            self.counts[arm] += 1.0
            n = self.counts[arm]
            self.values[arm] += (reward - self.values[arm]) / n
            if self.mode == "exp3":
                probs = self._probabilities()
                scaled_reward = reward / max(probs[arm], 1e-12)
                self.weights[arm] *= float(np.exp(self.eta * scaled_reward / len(self.arm_indices)))

        self.prev_loss = current_loss
        self.t += 1

        if epoch <= warmup_epochs:
            next_lr = float(np.clip(cosine_lr, self.lr_min, self.lr_max))
            next_arm = self._arm_position_for_lr(next_lr)
            decision = "warmup_cosine"
        elif self.mode == "ucb":
            bonus = self.gamma * np.sqrt(np.log(max(self.t, 2.0)) / np.maximum(self.counts, 1.0))
            next_arm = int(np.argmax(self.values + bonus))
            next_lr = float(self.lr_grid[self.arm_indices[next_arm]])
            decision = "ucb"
        else:
            probs = self._probabilities()
            next_arm = int(np.random.choice(len(self.arm_indices), p=probs))
            next_lr = float(self.lr_grid[self.arm_indices[next_arm]])
            decision = "exp3"

        self.pending_arm = next_arm
        self.set_lr(next_lr)
        self.last_decision = {
            "decision": decision,
            "reward": float(reward),
            "arm": int(self.arm_indices[next_arm]),
            "next_lr": float(self.current_lr),
            "value": float(self.values[next_arm]),
            "count": int(self.counts[next_arm]),
        }
        return self.last_decision


class HypergradHBScheduler(DynamicScheduler):
    """Batch-level hypergradient descent with heavy-ball smoothing."""

    def __init__(
        self,
        optimizer,
        init_lr=0.5,
        beta=0.03,
        momentum=0.9,
        lr_min=1e-3,
        lr_max=1.5,
        last_epoch=-1,
    ):
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.beta = float(beta)
        self.momentum = float(momentum)
        self.prev_grads = None
        self.velocity = 0.0
        self.last_signal = 0.0
        self.updates = 0
        super().__init__(optimizer, init_lr=init_lr, last_epoch=last_epoch)

    def set_lr(self, lr):
        super().set_lr(float(np.clip(lr, self.lr_min, self.lr_max)))

    def hypergrad_step_from_grads(self, parameters):
        current = []
        for param in parameters:
            if param.grad is not None:
                current.append(param.grad.detach().float().flatten().cpu())
        if not current:
            return 0.0
        current_vec = torch.cat(current)
        if self.prev_grads is None or self.prev_grads.numel() != current_vec.numel():
            self.prev_grads = current_vec.clone()
            return 0.0

        dot = float(torch.dot(current_vec, self.prev_grads).item())
        norm = float(current_vec.norm().item() * self.prev_grads.norm().item() + 1e-12)
        signal = float(np.clip(dot / norm, -1.0, 1.0))
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * signal
        new_lr = float(np.exp(np.log(max(self.current_lr, 1e-12)) + self.beta * self.velocity))
        self.set_lr(new_lr)
        self.prev_grads = current_vec.clone()
        self.last_signal = signal
        self.updates += 1
        return signal
    



class WindowDataset(Dataset):
    def __init__(self, X, y, window=15):
        self.X, self.y = [], []
        i = window
        while i < len(X):
            if i%180 != 0:
                self.X.append(X[i-window:i])
                self.y.append(y[i])
            else:
                i += window 
                self.X.append(X[i-window:i])
                self.y.append(y[i])
            i += 1
        
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def cosine_annealing_lr(epoch, total_epochs, initial_lr, min_lr=0.0):
    total_epochs = max(total_epochs, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * epoch / total_epochs))
    return float(min_lr + (initial_lr - min_lr) * cosine)


def transform_policy_targets(targets, args):
    if args.policy_output == "raw_lr":
        return targets
    if args.policy_output == "residual_log":
        # Predict bounded log-multiplier of cosine: target = log(LR / cosine_lr)
        # clipped to [-1, +1]. Decoded as LR = cosine * exp(clip(out, -1, +1)).
        transformed = []
        for index, target in enumerate(targets):
            epoch = index % args.policy_oracle_period
            base_lr = cosine_annealing_lr(epoch, args.policy_oracle_period, args.lr, args.policy_min_lr)
            ratio = max(target, 1e-9) / max(base_lr, 1e-9)
            log_ratio = float(np.clip(np.log(ratio), -1.0, 1.0))
            transformed.append(log_ratio)
        return np.asarray(transformed, dtype=np.float32)

    transformed = []
    for index, target in enumerate(targets):
        epoch = index % args.policy_oracle_period
        base_lr = cosine_annealing_lr(epoch, args.policy_oracle_period, args.lr, args.policy_min_lr)
        multiplier = target / max(base_lr, args.policy_min_lr)
        transformed.append(np.clip(multiplier, args.policy_multiplier_min, args.policy_multiplier_max))
    return np.asarray(transformed, dtype=np.float32)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, batch_first):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :] 
        return self.head(h).squeeze(-1)


def lr_policy_training(args):
    X_train = torch.tensor(np.load(args.features_train))
    y_train = torch.tensor(transform_policy_targets(np.load(args.targets_train), args))

    X_test = torch.tensor(np.load(args.features_test))
    y_test = torch.tensor(transform_policy_targets(np.load(args.targets_test), args))
        
    train_ds = WindowDataset(X_train, y_train, window=args.policy_window)
    test_ds = WindowDataset(X_test, y_test, window=args.policy_window)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_gru = GRUModel(input_dim=X_train.shape[-1], hidden=129, num_layers=3, dropout=0.027, batch_first=True).to(device)


    opt = torch.optim.Adam(model_gru.parameters(), lr=0.000131, weight_decay=1e-3)
    criterion = nn.MSELoss()


    for _ in range(args.policy_epochs):
        model_gru.train()
        tr_loss, tr_n = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_gru(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
    

    model_gru.eval()
    preds_scaled, targets_scaled = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model_gru(xb).cpu().numpy()
            preds_scaled.append(p)
            targets_scaled.append(yb.numpy())
    preds_scaled = np.concatenate(preds_scaled).reshape(-1, 1).flatten()
    targets_scaled = np.concatenate(targets_scaled).reshape(-1, 1).flatten()

    policy_mse = float(np.mean((preds_scaled - targets_scaled) ** 2))
    policy_mae = float(np.mean(np.abs(preds_scaled - targets_scaled)))
    print(
        f"[LR policy] output={args.policy_output} train_windows={len(train_ds)} "
        f"test_windows={len(test_ds)} test_mse={policy_mse:.6f} test_mae={policy_mae:.6f}"
    )

    
    return model_gru


class PolicyOutputEnsemble(nn.Module):
    """Average raw policy outputs across independently trained checkpoints."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


def _build_policy_from_checkpoint(checkpoint, args, device):
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', args.policy_model_type)
    input_dim = int(config.get('input_dim', args.policy_input_dim))
    if model_type == 'modular':
        model = ModularLRPolicy(input_dim=input_dim).to(device)
    elif model_type == 'attention_modular':
        window = int(config.get('window', args.policy_window))
        model = AttentionModularLRPolicy(input_dim=input_dim, max_window=max(window, 16)).to(device)
    elif model_type == 'gru':
        model = GRULRPolicy(input_dim=input_dim).to(device)
    elif model_type == 'tcn':
        hidden = int(config.get('hidden', 64))
        num_layers = int(config.get('num_layers', 4))
        model = TCNLRPolicy(input_dim=input_dim, hidden=hidden, num_layers=num_layers).to(device)
    elif model_type == 'patchtst':
        hidden = int(config.get('hidden', 16))
        num_layers = int(config.get('num_layers', 2))
        max_window = int(config.get('max_window', 20))
        model = PatchTSTLRPolicy(
            input_dim=input_dim, hidden=hidden, num_layers=num_layers, max_window=max_window
        ).to(device)
    elif model_type == 'nbeats':
        hidden = int(config.get('hidden', 128))
        num_blocks = int(config.get('num_blocks', 3))
        window = int(config.get('window', args.policy_window))
        model = NBeatsLRPolicy(
            input_dim=input_dim, window=window, hidden=hidden, num_blocks=num_blocks
        ).to(device)
    elif model_type == 'dlinear':
        window = int(config.get('window', args.policy_window))
        model = DLinearLRPolicy(input_dim=input_dim, window=window).to(device)
    elif model_type == 'curve':
        lookahead_n = int(config.get('lookahead_n', args.policy_lookahead_n))
        landscape_dim = int(config.get('landscape_dim', 30))
        model = CurveLRPolicy(
            input_dim=input_dim,
            landscape_dim=landscape_dim,
            lookahead_n=lookahead_n,
        ).to(device)
        args.policy_lookahead_n = lookahead_n
    else:
        raise ValueError(f"Unknown policy model type: {model_type}")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, model_type, input_dim


def load_policy_checkpoint(args, device):
    checkpoint_paths = [path.strip() for path in args.policy_checkpoint.split(',') if path.strip()]
    if not checkpoint_paths:
        raise ValueError("--policy_checkpoint is empty")

    models = []
    model_types = []
    input_dims = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model, model_type, input_dim = _build_policy_from_checkpoint(checkpoint, args, device)
        models.append(model)
        model_types.append(model_type)
        input_dims.append(input_dim)

    model = models[0] if len(models) == 1 else PolicyOutputEnsemble(models).to(device).eval()
    model_type = model_types[0] if len(set(model_types)) == 1 else 'ensemble'
    input_dim = input_dims[0]
    args.policy_model_type = model_type
    args.policy_input_dim = input_dim
    print(
        f"[LR policy] loaded checkpoint={args.policy_checkpoint} "
        f"model_type={model_type} input_dim={input_dim} n_models={len(models)}"
    )
    return model


def deploy_curve_policy_step(
    epoch,
    args,
    device,
    model_gru,
    features_list,
    curve_buffer,
    online_optimizer,
    student_scheduler,
    snapshot_dir,
):
    """One online MPC step for the curve policy.

    Steps:
      1. Read the just-probed real landscape ``R_t`` (first 30 dims of latest feature).
      2. If ``curve_buffer[t]`` has past windows that produced lookahead forecasts of
         epoch ``t``, re-run the policy on each stored window with gradients enabled,
         compute MSE against ``R_t`` for the matching lookahead step ``k``, and take a
         single online SGD step (skipped when ``online_optimizer`` is None).
      3. Run the policy on the trailing window (no grad) to (a) record forecasts for
         post-hoc analysis, (b) decide ``next_lr`` from the argmin of the current-epoch
         forecast, and (c) push the trailing window into the buffer keyed by future
         epochs ``t + k``.
      4. Apply ``next_lr`` to the student scheduler.

    Buffer entries are ``(made_at_epoch, lookahead_step k, window_tensor)``. We store
    the window itself (not the prediction) so the online loss is differentiable wrt
    the *current* policy parameters.
    """
    landscape_dim = 30
    real_curve_np = features_list[-1].detach().cpu().numpy()[:landscape_dim].astype(np.float32)
    real_curve = torch.from_numpy(real_curve_np).to(device)

    online_loss_value = float('nan')
    pending = curve_buffer.pop(epoch, [])
    pending_count = len(pending)

    if pending:
        # Only stack windows of identical length. We use the most common length found
        # in the pending list (in steady state all windows are ``policy_window`` long;
        # short windows from the warm-up period are simply skipped).
        from collections import Counter
        lengths = Counter(entry[2].shape[0] for entry in pending)
        target_len = max(lengths, key=lambda key: lengths[key])
        usable = [entry for entry in pending if entry[2].shape[0] == target_len]
        if usable:
            if online_optimizer is not None:
                model_gru.train()
                windows = torch.stack([entry[2] for entry in usable], dim=0).to(device)
                predicted_batch = model_gru(windows)
                loss_terms = [
                    F.mse_loss(predicted_batch[batch_i, k], real_curve)
                    for batch_i, (_, k, _) in enumerate(usable)
                ]
                online_loss = torch.stack(loss_terms).sum()
                online_optimizer.zero_grad(set_to_none=True)
                online_loss.backward()
                online_optimizer.step()
                online_loss_value = float(online_loss.detach().cpu().item())
            else:
                model_gru.eval()
                with torch.no_grad():
                    windows = torch.stack([entry[2] for entry in usable], dim=0).to(device)
                    predicted_batch = model_gru(windows)
                    online_loss_value = float(
                        torch.stack(
                            [
                                F.mse_loss(predicted_batch[batch_i, k], real_curve)
                                for batch_i, (_, k, _) in enumerate(usable)
                            ]
                        )
                        .sum()
                        .cpu()
                        .item()
                    )

    model_gru.eval()
    with torch.no_grad():
        window_tensor = torch.stack(features_list[-args.policy_window:]).float()
        x_batch = window_tensor.to(device).unsqueeze(0)
        predicted = model_gru(x_batch)
        predicted_curves = predicted[0]

    curves_np = predicted_curves.detach().cpu().numpy().astype(np.float32)

    next_epoch_curve_np = curves_np[0]
    argmin_mode = getattr(args, 'policy_argmin_mode', 'parabolic')
    if argmin_mode == 'parabolic' and not args.policy_argmin_refine:
        argmin_mode = 'hard'
    if argmin_mode == 'softmax':
        argmin_lr = softmax_weighted_lr(
            next_epoch_curve_np,
            temperature=getattr(args, 'policy_argmin_temperature', 0.1),
        )
        frac_idx = float(int(np.argmin(next_epoch_curve_np)))
    elif argmin_mode == 'parabolic':
        frac_idx = parabolic_argmin(next_epoch_curve_np)
        argmin_lr = fractional_index_to_lr(frac_idx)
    else:  # hard
        frac_idx = float(int(np.argmin(next_epoch_curve_np)))
        argmin_lr = float(LR_GRID[int(frac_idx)])

    cosine_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
    if epoch <= args.policy_warmup_epochs:
        next_lr = cosine_lr
        veto_triggered = False
    else:
        next_lr = float(np.clip(argmin_lr, args.policy_min_lr, args.policy_max_lr))
        if args.policy_cosine_blend > 0:
            blend = float(np.clip(args.policy_cosine_blend, 0.0, 1.0))
            next_lr = (1.0 - blend) * next_lr + blend * cosine_lr

        # AdaLRS-style veto: if the test loss has been monotonically rising
        # for ``policy_veto_consecutive`` consecutive epochs while we used a
        # policy LR above cosine, revert to a fraction of cosine. Cheap
        # safeguard targeting the late-training argmin saturation observed
        # in 2h_curve_results.
        veto_triggered = False
        if args.policy_veto_mode == 'adalrs':
            losses = getattr(args, '_student_test_losses', [])
            need = args.policy_veto_consecutive + 1
            if len(losses) >= need:
                tail = losses[-need:]
                strictly_rising = all(tail[i + 1] > tail[i] for i in range(len(tail) - 1))
                last_lr_higher = (
                    getattr(args, '_last_policy_lr', None) is not None
                    and args._last_policy_lr > cosine_lr * 1.05
                )
                if strictly_rising and last_lr_higher:
                    clamp = float(np.clip(args.policy_veto_clamp, 0.1, 1.5))
                    next_lr = clamp * cosine_lr
                    veto_triggered = True

    student_scheduler.set_lr(next_lr)
    args._last_policy_lr = next_lr

    # Push the just-used window into the buffer for future online updates.
    # The offline curve dataset aligns ``curves[k]`` with absolute epoch
    # ``epoch + k + 1`` (window ends at ``epoch``, targets start at the next epoch).
    # We mirror this online alignment so that forecast supervision is consistent.
    cpu_window = window_tensor.detach().cpu()
    for k in range(predicted_curves.shape[0]):
        future_epoch = epoch + k + 1
        if future_epoch <= args.epochs:
            curve_buffer.setdefault(future_epoch, []).append((epoch, k, cpu_window))

    real_argmin_idx = int(np.argmin(real_curve_np))
    pred_argmin_idx = int(np.argmin(next_epoch_curve_np))
    immediate_curve_mse = float(np.mean((next_epoch_curve_np - real_curve_np) ** 2))

    print(
        f"[LR policy] epoch={epoch} output=curve_argmin mode={argmin_mode} "
        f"online_loss={online_loss_value:.6f} immediate_mse={immediate_curve_mse:.6f} "
        f"argmin_idx_pred={pred_argmin_idx} argmin_idx_real={real_argmin_idx} "
        f"argmin_lr={argmin_lr:.6f} next_student_lr={next_lr:.6f} "
        f"online_lr={args.policy_online_lr:.2e} lookahead_n={args.policy_lookahead_n} "
        f"pending={pending_count} veto={veto_triggered}"
    )

    if args.policy_curve_save_artifacts:
        np.save(
            os.path.join(snapshot_dir, f'predicted_curves_epoch_{epoch}.npy'),
            curves_np,
        )
        np.save(
            os.path.join(snapshot_dir, f'real_curve_epoch_{epoch}.npy'),
            real_curve_np,
        )
        with open(os.path.join(snapshot_dir, 'curve_online.jsonl'), 'a', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {
                        'epoch': epoch,
                        'online_loss': online_loss_value,
                        'immediate_mse': immediate_curve_mse,
                        'argmin_idx_pred': pred_argmin_idx,
                        'argmin_idx_real': real_argmin_idx,
                        'argmin_lr': argmin_lr,
                        'next_lr': next_lr,
                        'predicted_curves_shape': list(curves_np.shape),
                        'pending_predictions': pending_count,
                    }
                )
                + '\n'
            )


def prepare_policy_feature(feature, epoch, args, student_scheduler):
    feature_np = feature.detach().cpu().numpy().astype(np.float32)
    if (
        args.policy_model_type in {'modular', 'attention_modular', 'curve'}
        or args.policy_append_time_features
    ):
        previous_lr = float(student_scheduler.get_last_lr()[0])
        feature_np = append_time_features(
            feature_np.reshape(1, -1),
            np.asarray([epoch], dtype=np.float32),
            args.epochs,
            args.lr,
            args.policy_min_lr,
            previous_lrs=np.asarray([previous_lr], dtype=np.float32),
        )[0]
    return torch.tensor(feature_np, dtype=torch.float32)


def one_step(model, x_batch, y_batch, loss, loss_func, lr):
    grads = torch.autograd.grad(loss, model.parameters())
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p -= lr * g
    new_loss = loss_func(model(x_batch), y_batch)
    return new_loss.item()

def research(model, x_batch_list, y_batch_list, loss_func, snapshot, device):
    dict_loss  = {0.0005: [], 0.001: [], 0.01: [], 0.025: [], 0.05: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [],  
                      0.5: [],  0.6: [], 0.7: [], 0.8:[], 0.9:[], 1.0: [], 1.1:[], 1.2: [], 1.3: [], 1.4: [],  
                      1.5: [],  1.6: [], 1.7: [], 1.8:[], 1.9:[], 2.0:[], 2.1: [], 2.2: [], 2.3: [], 2.4: [], 2.5: []}

    base_state = snapshot
        
    for j in dict_loss.keys():
        model_copy = copy.deepcopy(model).to(device)
        model_copy.load_state_dict(base_state)
        model_copy.train()
            
        loss = loss_func(model_copy(x_batch_list[0]), y_batch_list[0])
        dict_loss[j].append(one_step(model_copy, x_batch_list[1], y_batch_list[1], loss,
                                            loss_func, lr=j))
    return dict_loss

def get_device():
    gpu = int(getattr(args, "gpu", 0)) if "args" in globals() else 0
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def try_cuda(*wargs):
    """Backward-compatible helper: move tensors/modules to the configured device.

    The original code always called .cuda(), which made CPU smoke tests fail.
    """
    device = get_device()
    moved = []
    for arg in wargs:
        if hasattr(arg, 'to'):
            moved.append(arg.to(device))
        else:
            moved.append(arg)
    return tuple(moved)


def is_multilabel_task():
    return bool(getattr(args, "is_multilabel", False) or getattr(args, "dataset", "") in ("wafer", "mydataset"))


def supervised_loss(logits, targets):
    if is_multilabel_task():
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    return F.cross_entropy(logits, targets.long())


def kl_div_logits(p, q, T):
    if is_multilabel_task():
        p_prob = torch.sigmoid(p / T).clamp(1e-6, 1 - 1e-6)
        q_prob = torch.sigmoid(q / T).clamp(1e-6, 1 - 1e-6)
        kl = (
            p_prob * (p_prob.log() - q_prob.log()) +
            (1 - p_prob) * ((1 - p_prob).log() - (1 - q_prob).log())
        )
        return kl.mean() * (T * T)
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


def get_batch(data_loader, batch_index):
    start_index = batch_index * data_loader.batch_size
    end_index = start_index + data_loader.batch_size
    batch_data = []
    batch_targets = []
    
    for i in range(start_index, end_index):
        if i >= len(data_loader.dataset):
            break
        data, target = data_loader.dataset[i]
        batch_data.append(data)
        batch_targets.append(target)
    if not batch_data:
        raise IndexError(f"empty batch_index={batch_index} for dataset length {len(data_loader.dataset)}")
    if isinstance(batch_targets[0], torch.Tensor):
        targets = torch.stack(batch_targets)
    else:
        targets = torch.tensor(batch_targets)
    return torch.stack(batch_data), targets


def evaluate(teacher, student, loader, epoch):

    teacher_vector = []
    student_vector = []
    snapshot_dir = os.path.join(args.snapshot_dir, args.exp_name)

    os.makedirs(snapshot_dir, exist_ok=True)

    teacher.eval()
    student.eval()
    sf_optimizers = getattr(args, '_sf_optimizers', None)
    if sf_optimizers is not None:
        for opt in sf_optimizers:
            if hasattr(opt, 'eval'):
                opt.eval()
    teacher_loss, student_loss = 0.0, 0.0
    start = time.time()

    first_batch_inputs = None
    first_batch_targets = None
    second_batch_inputs = None
    second_batch_targets = None

    if is_multilabel_task():
        teacher_correct_bits, student_correct_bits = 0, 0
        teacher_correct_exact, student_correct_exact = 0, 0
        total_bits, total_samples = 0, 0
        all_targets, all_teacher_preds, all_student_preds = [], [], []

        for batch_idx, batch in enumerate(loader):
            with torch.no_grad():
                inputs, targets = try_cuda(*batch[:2])
                targets = targets.float()

                if batch_idx == 0:
                    first_batch_inputs = inputs.cpu()
                    first_batch_targets = targets.cpu()
                if batch_idx == 1:
                    second_batch_inputs = inputs.cpu()
                    second_batch_targets = targets.cpu()

                teacher_logits = teacher(inputs)
                student_logits = student(inputs)
                teacher_loss += supervised_loss(teacher_logits, targets).item()
                student_loss += supervised_loss(student_logits, targets).item()

                teacher_probs = torch.sigmoid(teacher_logits)
                student_probs = torch.sigmoid(student_logits)
                threshold = float(getattr(args, 'wafer_threshold', 0.5))
                teacher_preds = (teacher_probs > threshold).float()
                student_preds = (student_probs > threshold).float()

                teacher_correct_bits += (teacher_preds == targets).sum().item()
                student_correct_bits += (student_preds == targets).sum().item()
                teacher_correct_exact += (teacher_preds == targets).all(dim=1).sum().item()
                student_correct_exact += (student_preds == targets).all(dim=1).sum().item()
                total_bits += targets.numel()
                total_samples += targets.size(0)

                all_targets.append(targets.detach().cpu().numpy())
                all_teacher_preds.append(teacher_preds.detach().cpu().numpy())
                all_student_preds.append(student_preds.detach().cpu().numpy())
                teacher_vector.append(np.hstack([targets.detach().cpu().numpy(), teacher_probs.detach().cpu().numpy()]))
                student_vector.append(np.hstack([targets.detach().cpu().numpy(), student_probs.detach().cpu().numpy()]))

        end = time.time()
        avg_teacher_loss = teacher_loss / max(len(loader), 1)
        avg_student_loss = student_loss / max(len(loader), 1)
        y_true = np.concatenate(all_targets, axis=0)
        y_teacher = np.concatenate(all_teacher_preds, axis=0)
        y_student = np.concatenate(all_student_preds, axis=0)
        teacher_f1 = f1_score(y_true, y_teacher, average="macro", zero_division=0)
        student_f1 = f1_score(y_true, y_student, average="macro", zero_division=0)
        teacher_micro_f1 = f1_score(y_true, y_teacher, average="micro", zero_division=0)
        student_micro_f1 = f1_score(y_true, y_student, average="micro", zero_division=0)

        teacher_bit_acc = 100.0 * teacher_correct_bits / max(total_bits, 1)
        student_bit_acc = 100.0 * student_correct_bits / max(total_bits, 1)
        teacher_exact_acc = 100.0 * teacher_correct_exact / max(total_samples, 1)
        student_exact_acc = 100.0 * student_correct_exact / max(total_samples, 1)

        print('[Eval] Epoch: %d | Teacher Loss: %.3f | Teacher Bit Acc: %.3f | Teacher Exact Acc: %.3f | '
              'Student Loss: %.3f | Student Bit Acc: %.3f | Student Exact Acc: %.3f | '
              'Teacher F1 macro/micro: %.3f/%.3f | Student F1 macro/micro: %.3f/%.3f | Time: %.3f | '
              % (epoch, avg_teacher_loss, teacher_bit_acc, teacher_exact_acc,
                 avg_student_loss, student_bit_acc, student_exact_acc,
                 teacher_f1, teacher_micro_f1, student_f1, student_micro_f1, end - start))

        metrics = {
            'epoch': epoch,
            'teacher_loss': avg_teacher_loss,
            'teacher_acc': teacher_bit_acc,
            'teacher_exact_acc': teacher_exact_acc,
            'teacher_f1': teacher_f1,
            'teacher_micro_f1': teacher_micro_f1,
            'student_loss': avg_student_loss,
            'student_acc': student_bit_acc,
            'student_exact_acc': student_exact_acc,
            'student_f1': student_f1,
            'student_micro_f1': student_micro_f1,
            'elapsed_sec': end - start,
            'task': 'multilabel',
        }
        student_concat = np.concatenate(student_vector, axis=0)
        if bool(getattr(args, 'wafer_legacy_feature_slice', False)):
            # Compatibility mode for old features_wafer_v3.npy generation.
            # The historical wafer script used [:, 1:], which leaked label columns
            # into the LR-policy feature vector and produced 60 dims for 8 classes.
            student_scores_for_features = student_concat[:, 1:]
        else:
            student_scores_for_features = student_concat[:, args.num_classes:]

    else:
        teacher_correct, student_correct = 0, 0
        total = 0
        for batch_idx, batch in enumerate(loader):
            with torch.no_grad():
                inputs, targets = try_cuda(*batch[:2])
                targets_float = targets.reshape(-1, 1).float()

                if batch_idx == 0:
                    first_batch_inputs = inputs.cpu()
                    first_batch_targets = targets.cpu()
                if batch_idx == 1:
                    second_batch_inputs = inputs.cpu()
                    second_batch_targets = targets.cpu()

                teacher_logits = teacher(inputs)
                student_logits = student(inputs)
                teacher_scores = F.log_softmax(teacher_logits, dim=-1)
                student_scores = F.log_softmax(student_logits, dim=-1)
                teacher_loss += F.cross_entropy(teacher_logits, targets.long()).item()
                student_loss += F.cross_entropy(student_logits, targets.long()).item()
                total += targets.size(0)
                teacher_correct += teacher_scores.max(1)[1].eq(targets).sum().item()
                student_correct += student_scores.max(1)[1].eq(targets).sum().item()

                teacher_vector.append(np.hstack([targets_float.detach().cpu().numpy(), teacher_scores.detach().cpu().numpy()]))
                student_vector.append(np.hstack([targets_float.detach().cpu().numpy(), student_scores.detach().cpu().numpy()]))

        end = time.time()
        avg_teacher_loss = teacher_loss / max(len(loader), 1)
        avg_student_loss = student_loss / max(len(loader), 1)
        teacher_concat = np.concatenate(teacher_vector, axis=0)
        student_concat = np.concatenate(student_vector, axis=0)
        all_targets = teacher_concat[:, 0].astype(int)
        teacher_scores_np = teacher_concat[:, 1:]
        student_scores_np = student_concat[:, 1:]
        teacher_labels = teacher_scores_np.argmax(axis=1)
        student_labels = student_scores_np.argmax(axis=1)
        teacher_f1 = f1_score(all_targets, teacher_labels, average="macro", zero_division=0)
        student_f1 = f1_score(all_targets, student_labels, average="macro", zero_division=0)
        labels = np.arange(args.num_classes)
        teacher_top5 = top_k_accuracy_score(all_targets, teacher_scores_np, k=min(5, args.num_classes), labels=labels)
        student_top5 = top_k_accuracy_score(all_targets, student_scores_np, k=min(5, args.num_classes), labels=labels)

        print('[Eval] Epoch: %d | Teacher Test Loss: %.3f | Teacher Test Acc: %.3f | Student Test Loss: %.3f | Student Test Acc: %.3f '
              '| Teacher Test F1: %.3f | Student Test F1: %.3f | Teacher Top 5 Accuracy: %.3f | Student Top 5 Accuracy: %.3f | Time: %.3f | '
              % (epoch, avg_teacher_loss, 100. * teacher_correct / max(total, 1), avg_student_loss, 100. * student_correct / max(total, 1),
                 teacher_f1, student_f1, teacher_top5, student_top5, end-start))

        metrics = {
            'epoch': epoch,
            'teacher_loss': avg_teacher_loss,
            'teacher_acc': 100. * teacher_correct / max(total, 1),
            'teacher_f1': teacher_f1,
            'teacher_top5': teacher_top5,
            'student_loss': avg_student_loss,
            'student_acc': 100. * student_correct / max(total, 1),
            'student_f1': student_f1,
            'student_top5': student_top5,
            'elapsed_sec': end - start,
            'task': 'singlelabel',
        }
        student_scores_for_features = student_concat[:, 1:]

    torch.save(metrics, os.path.join(snapshot_dir, f'data_epoch_{epoch}.pt'))
    with open(os.path.join(snapshot_dir, 'metrics.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics) + '\n')

    mean_vector = student_scores_for_features.mean(axis=0)
    std_vector = student_scores_for_features.std(axis=0)

    device = get_device()
    template_model = PreResNet(num_classes=args.num_classes, depth=args.policy_probe_depth, input_size=args.input_size).to(device)

    print("Running student landscape probe...")
    if second_batch_inputs is None or second_batch_targets is None:
        second_batch_inputs = first_batch_inputs
        second_batch_targets = first_batch_targets
    result_student = research(
        model=template_model,
        x_batch_list=[first_batch_inputs.to(device), second_batch_inputs.to(device)],
        y_batch_list=[first_batch_targets.to(device), second_batch_targets.to(device)],
        loss_func=supervised_loss,
        snapshot=student.state_dict(),
        device=device,
    )

    steps = np.column_stack(list(result_student.values()))[0]
    features = torch.tensor(np.concatenate([steps, mean_vector, std_vector]), dtype=torch.float32)
    return [avg_teacher_loss, avg_student_loss, features]


def train(teacher, student, loader, epoch, args, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler):
    teacher.train()
    student.train()
    if hasattr(teacher_optimizer, 'train'):
        teacher_optimizer.train()
    if hasattr(student_optimizer, 'train'):
        student_optimizer.train()
    loss = 0.0
    student_correct, teacher_correct = 0, 0
    total = 0
    start = time.time()
    for idx, (inputs, targets) in enumerate(loader):
        if args.max_train_batches and idx >= args.max_train_batches:
            break
        inputs, targets = try_cuda(inputs, targets)
        teacher_optimizer.zero_grad()
        student_optimizer.zero_grad()
        teacher_logits = teacher(inputs)
        student_logits = student(inputs)

        if args.loss == 'kl_ce':
            teacher_loss = supervised_loss(teacher_logits, targets) + args.alpha * kl_div_logits(teacher_logits, student_logits.detach(), args.T)
            student_loss = supervised_loss(student_logits, targets) + args.alpha * kl_div_logits(student_logits, teacher_logits.detach(), args.T)
        elif args.loss == 'kl':
            teacher_loss = supervised_loss(teacher_logits, targets) + args.alpha * kl_div_logits(teacher_logits, student_logits.detach(), args.T)
            student_loss = kl_div_logits(student_logits, teacher_logits.detach(), args.T)
        elif args.loss == 'symmetric_kl':
            teacher_loss = supervised_loss(teacher_logits, targets) + args.alpha * (
                kl_div_logits(teacher_logits, student_logits.detach(), args.T) +
                kl_div_logits(student_logits.detach(), teacher_logits, args.T)
            )
            student_loss = kl_div_logits(student_logits, teacher_logits.detach(), args.T) + kl_div_logits(teacher_logits.detach(), student_logits, args.T)
        elif args.loss == 'symmetric_kl_ce':
            teacher_loss = supervised_loss(teacher_logits, targets) + args.alpha * (
                kl_div_logits(teacher_logits, student_logits.detach(), args.T) +
                kl_div_logits(student_logits.detach(), teacher_logits, args.T)
            )
            student_loss = supervised_loss(student_logits, targets) + args.alpha * (
                kl_div_logits(student_logits, teacher_logits.detach(), args.T) +
                kl_div_logits(teacher_logits.detach(), student_logits, args.T)
            )
        else:
            raise ValueError(f"unknown loss: {args.loss}")

        teacher_loss.backward()
        student_loss.backward()
        if args.scheduler == 'hypergrad_hb':
            student_scheduler.hypergrad_step_from_grads(student.parameters())
        teacher_optimizer.step()
        student_optimizer.step()

        with torch.no_grad():
            if is_multilabel_task():
                threshold = float(getattr(args, 'wafer_threshold', 0.5))
                teacher_labels = (torch.sigmoid(teacher_logits) > threshold).float()
                student_labels = (torch.sigmoid(student_logits) > threshold).float()
                targets_float = targets.float()
                teacher_correct += (teacher_labels == targets_float).all(dim=1).sum().item()
                student_correct += (student_labels == targets_float).all(dim=1).sum().item()
                batch_n = targets.size(0)
            else:
                teacher_labels = teacher_logits.max(1)[1]
                student_labels = student_logits.max(1)[1]
                teacher_correct += teacher_labels.eq(targets).sum().item()
                student_correct += student_labels.eq(targets).sum().item()
                batch_n = targets.size(0)
        loss += teacher_loss.item() * batch_n
        total += batch_n

        for _ in range(args.student_steps_ratio - 1):
            s_inputs, s_targets = get_batch(loader, args.student_index)
            s_inputs, s_targets = try_cuda(s_inputs, s_targets)
            args.student_index = (args.student_index + 1) % len(loader)
            with torch.no_grad():
                teacher_extra_logits = teacher(s_inputs)
            student_extra_logits = student(s_inputs)
            if args.loss == 'kl_ce':
                student_loss = supervised_loss(student_extra_logits, s_targets) + args.alpha * kl_div_logits(student_extra_logits, teacher_extra_logits.detach(), args.T)
            elif args.loss == 'kl':
                student_loss = kl_div_logits(student_extra_logits, teacher_extra_logits.detach(), args.T)
            elif args.loss == 'symmetric_kl':
                student_loss = kl_div_logits(student_extra_logits, teacher_extra_logits.detach(), args.T) + kl_div_logits(teacher_extra_logits.detach(), student_extra_logits, args.T)
            elif args.loss == 'symmetric_kl_ce':
                student_loss = supervised_loss(student_extra_logits, s_targets) + args.alpha * (
                    kl_div_logits(student_extra_logits, teacher_extra_logits.detach(), args.T) +
                    kl_div_logits(teacher_extra_logits.detach(), student_extra_logits, args.T)
                )
            student_optimizer.zero_grad()
            student_loss.backward()
            if args.scheduler == 'hypergrad_hb':
                student_scheduler.hypergrad_step_from_grads(student.parameters())
            student_optimizer.step()
    end = time.time()
    step = epoch
    denom = max(total, 1)
    acc_name = "Exact Acc" if is_multilabel_task() else "Train Acc"
    print('[Train] Epoch: %d | Teacher lr=%.4f | Teacher Loss: %.3f | Teacher %s: %.3f | Student lr=%.4f | Student %s: %.3f | Time: %.3f |'
          % (step, teacher_scheduler.get_last_lr()[0], loss / denom, acc_name, 100. * teacher_correct / denom,
             student_scheduler.get_last_lr()[0], acc_name, 100. * student_correct / denom, end-start))
    teacher_scheduler.step()
    student_scheduler.step()


parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('--exp_name', type=str, default='LoT_ResNet')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--student_index', type=int, default=0, help='an independent index for student updating')
parser.add_argument('--student_steps_ratio', type=int, default=4)
parser.add_argument('--loss', type=str, default='kl_ce', choices=['kl', 'kl_ce', 'symmetric_kl', 'symmetric_kl_ce'])
# original
parser.add_argument('--dataset', type=str, default='cifar100', choices = ['cifar10', 'cifar100', 'wafer', 'mydataset'])
parser.add_argument('--datadir', type=str, default='data', help='data directory')
parser.add_argument('--download', action='store_true', help='download torchvision datasets if missing')
parser.add_argument('--wafer_npz', type=str, default='', help='path to MixedWM38/Wafer_Map_Datasets.npz with arr_0 images and arr_1 multilabel targets')
parser.add_argument('--wafer_num_classes', type=int, default=8, help='number of wafer defect labels')
parser.add_argument('--wafer_resize', type=int, default=56, help='resize wafer maps to this square size before PreResNet; 0 disables resize')
parser.add_argument('--wafer_test_size', type=float, default=0.2, help='test split fraction for wafer dataset')
parser.add_argument('--wafer_threshold', type=float, default=0.5, help='sigmoid threshold for wafer multilabel metrics')
parser.add_argument('--wafer_augment', action=argparse.BooleanOptionalAction, default=True, help='enable/disable wafer train augmentations')
parser.add_argument('--wafer_deduplicate', action=argparse.BooleanOptionalAction, default=True, help='deduplicate equal wafer maps before splitting')
parser.add_argument('--wafer_stratify', action=argparse.BooleanOptionalAction, default=True, help='stratify wafer split by multilabel combination when possible')
parser.add_argument('--wafer_legacy_feature_slice', action='store_true', help='reproduce old wafer feature bug [:,1:] for compatibility with 60-dim features_wafer_v3.npy')
parser.add_argument('--input_size', type=int, default=32, help='image input size')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--depth_list', type=str, default='110_20', help='resnet model depth list')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--scheduler', type=str, default='dynamic', choices=['cosine', 'custom', 'dynamic', 'sf_sgd', 'sf_adamw', 'hypergrad', 'adalrs', 'bandit_exp3', 'bandit_ucb', 'hypergrad_hb'])
parser.add_argument('--hypergrad_beta', type=float, default=0.05, help='hypergradient step size in log-LR space (--scheduler hypergrad)')
parser.add_argument('--hypergrad_init_lr', type=float, default=0.5, help='initial LR for --scheduler hypergrad')
parser.add_argument('--adalrs_alpha', type=float, default=0.5, help='lower local-search multiplier around cosine LR (--scheduler adalrs)')
parser.add_argument('--adalrs_beta', type=float, default=1.5, help='upper local-search multiplier around cosine LR (--scheduler adalrs)')
parser.add_argument('--adalrs_margin', type=float, default=0.002, help='relative probe-loss improvement required to accept AdaLRS LR')
parser.add_argument('--adalrs_clamp', type=float, default=0.7, help='cosine multiplier used when AdaLRS rejects the probe winner')
parser.add_argument('--adalrs_init_lr', type=float, default=0.01, help='initial student LR for AdaLRS before first probe decision')
parser.add_argument('--bandit_eta', type=float, default=0.07, help='EXP3 learning rate for bandit LR schedulers')
parser.add_argument('--bandit_gamma', type=float, default=0.10, help='EXP3 exploration or UCB bonus scale for bandit LR schedulers')
parser.add_argument('--bandit_init_lr', type=float, default=0.01, help='initial student LR for bandit LR schedulers')
parser.add_argument('--hypergrad_hb_beta', type=float, default=0.03, help='log-LR step size for batch-level hypergradient-HB')
parser.add_argument('--hypergrad_hb_momentum', type=float, default=0.9, help='heavy-ball smoothing for batch-level hypergradient signal')
parser.add_argument('--sf_warmup_steps', type=int, default=200, help='warmup steps for Schedule-Free optimizers')
parser.add_argument('--epochs', type=int, default=180)
parser.add_argument('--max_train_batches', type=int, default=0, help='limit train batches per epoch for smoke tests')
parser.add_argument('--snapshot_dir', type=str, default='snapshots', help='directory for per-epoch metrics and LR traces')
parser.add_argument('--features_train', type=str, default='features_v3_train.npy')
parser.add_argument('--targets_train', type=str, default='targets_v3_train.npy')
parser.add_argument('--features_test', type=str, default='features_v3_test.npy')
parser.add_argument('--targets_test', type=str, default='targets_v3_test.npy')
parser.add_argument('--policy_epochs', type=int, default=40)
parser.add_argument('--policy_window', type=int, default=15)
parser.add_argument('--policy_warmup_epochs', type=int, default=15)
parser.add_argument('--policy_min_lr', type=float, default=0.001)
parser.add_argument('--policy_max_lr', type=float, default=2.5)
parser.add_argument('--policy_output', type=str, default='raw_lr', choices=['raw_lr', 'cosine_multiplier', 'curve_argmin', 'residual_log'])
parser.add_argument('--policy_oracle_period', type=int, default=180)
parser.add_argument('--policy_multiplier_min', type=float, default=0.0)
parser.add_argument('--policy_multiplier_max', type=float, default=5.0)
parser.add_argument('--policy_probe_depth', type=int, default=20)
parser.add_argument('--policy_checkpoint', type=str, default='', help='load a trained LR policy checkpoint instead of fitting from npy files')
parser.add_argument('--policy_model_type', type=str, default='gru', choices=['gru', 'modular', 'attention_modular', 'curve', 'tcn', 'patchtst', 'nbeats', 'dlinear'])
parser.add_argument('--policy_input_dim', type=int, default=230)
parser.add_argument('--policy_append_time_features', action='store_true', help='append epoch/cosine/current LR features for checkpoint policies')
parser.add_argument('--policy_cosine_blend', type=float, default=0.0, help='blend dynamic LR with cosine LR after warmup; 0=pure policy, 1=pure cosine')
parser.add_argument('--policy_lr_ema', type=float, default=0.0, help='EMA smoothing for dynamic LR after warmup; 0 disables smoothing')
parser.add_argument('--policy_cosine_after_epoch', type=int, default=0, help='if >0, hand control back to cosine after this epoch')
parser.add_argument('--policy_lookahead_n', type=int, default=2, help='number of future epochs the curve policy predicts (curve_argmin only)')
parser.add_argument('--policy_online_lr', type=float, default=0.0, help='online SGD LR for the curve policy during deploy; 0 disables updates (frozen ablation)')
parser.add_argument('--policy_online_optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--policy_argmin_refine', action='store_true', help='use parabolic interpolation around argmin for sub-grid LR selection (curve_argmin only, ignored if --policy_argmin_mode is not parabolic)')
parser.add_argument('--policy_argmin_mode', type=str, default='parabolic', choices=['parabolic', 'softmax', 'hard'], help='LR selection rule from the predicted curve (curve_argmin only)')
parser.add_argument('--policy_argmin_temperature', type=float, default=0.1, help='temperature for softmax-weighted LR selection (curve_argmin only, --policy_argmin_mode softmax)')
parser.add_argument('--policy_veto_mode', type=str, default='none', choices=['none', 'adalrs'], help='AdaLRS-style boundary safeguard: revert to clamp*cosine if loss goes up vs cosine reference (curve_argmin only)')
parser.add_argument('--policy_veto_clamp', type=float, default=0.7, help='multiplier on cosine LR when veto triggers (default 0.7)')
parser.add_argument('--policy_veto_consecutive', type=int, default=2, help='number of consecutive bad epochs needed to trigger the veto')
parser.add_argument('--policy_curve_save_artifacts', action='store_true', help='persist predicted vs real curves and online losses for post-hoc analysis')
parser.add_argument('--custom_lrs', type=str, default='', help='comma-separated LR schedule for --scheduler custom')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/LoT_ResNet'+randomhash+'CIFAR.pt', help='path to save the final model')
parser.add_argument('--resume_student', type=str, default='', help='path to a saved student state_dict (.pt) to resume training from')
parser.add_argument('--resume_teacher', type=str, default='', help='path to a saved teacher state_dict (.pt) to resume training from')
parser.add_argument('--start_epoch', type=int, default=1, help='epoch to start training from (use with --resume_student/teacher to extend a finished run)')
# Perturbation / robustness experiments.
parser.add_argument('--label_noise_rate', type=float, default=0.0, help='symmetric label noise rate applied to training labels (0 disables)')
parser.add_argument('--label_noise_start_epoch', type=int, default=1, help='first epoch at which label noise is active (1 = from start of training)')
parser.add_argument('--input_noise_std', type=float, default=0.0, help='Gaussian std added to normalized training inputs (0 disables); test set stays clean')
parser.add_argument('--input_noise_start_epoch', type=int, default=1, help='first epoch at which input noise is active (1 = from start of training)')
parser.add_argument('--shock_epoch', type=int, default=0, help='first epoch of a one-time LR shock (0 disables); overrides scheduler/policy LR during [shock_epoch, shock_epoch+shock_duration)')
parser.add_argument('--shock_lr', type=float, default=0.0, help='LR value forced on the student during the shock window')
parser.add_argument('--shock_duration', type=int, default=1, help='number of epochs to hold shock_lr')
parser.add_argument('--policy_zero_mean_ema', type=float, default=0.0, help='EMA decay (0..1) for the running mean subtracted from policy_pred before exp (residual_log only); 0 disables; 0.9 = strong centering')
args = parser.parse_args()
if args.dataset in ('wafer', 'mydataset'):
    args.num_classes = int(args.wafer_num_classes)
    args.is_multilabel = True
    if args.wafer_resize and args.wafer_resize > 0:
        args.input_size = int(args.wafer_resize)
    # Correct wafer feature length is 30 probe values + 8 means + 8 stds = 46.
    # The old exploratory script accidentally used [:, 1:] after hstack(labels, probs),
    # which gives 30 + 15 + 15 = 60 dims; keep it available behind a flag.
    if args.policy_input_dim == 230:
        args.policy_input_dim = 60 if args.wafer_legacy_feature_slice else 46
else:
    args.is_multilabel = False
#print(json.dumps(vars(args), indent=4))


def main():
    try:
        model_gru = None
        features_list = []

        config=configparser.ConfigParser()
        config.read('key.config')
        #wandb_username=config.get('WANDB', 'USER_NAME')
        #wandb_key=config.get('WANDB', 'API_KEY')        
        #wandb.login(key=wandb_key)
        #wandb.init(project='LoT_ResNet_CIFAR_'+args.dataset, entity=wandb_username, name=args.exp_name)
        depth_list = [int(number) for number in args.depth_list.split('_')]
        print(f"depth_list: {depth_list}")
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
        if args.scheduler in ('cosine', 'sf_sgd', 'sf_adamw', 'hypergrad', 'adalrs', 'bandit_exp3', 'bandit_ucb', 'hypergrad_hb'):
            # Pure baselines: no LR policy network needed.
            model_gru = None
        elif args.policy_checkpoint:
            model_gru = load_policy_checkpoint(args, device)
        else:
            model_gru = lr_policy_training(args)
        train_loader, test_loader = get_torch_dataset(args)

        # init teacher
        torch.manual_seed(args.seed)
        print('teacher depth:', depth_list[0])
        teacher=PreResNet(num_classes=args.num_classes, depth=depth_list[0], input_size=args.input_size)
        teacher, =try_cuda(teacher)

        # init student
        torch.manual_seed(args.seed+1)
        print('student depth:', depth_list[1])
        student=PreResNet(num_classes=args.num_classes, depth=depth_list[1], input_size=args.input_size)
        student, =try_cuda(student)
        args.student_index=0

        total_params = sum(p.numel() for p in teacher.parameters())
        print(f"Total number of teacher parameters: {total_params:,}")
        total_params = sum(p.numel() for p in student.parameters())
        print(f"Total number of student parameters: {total_params:,}")

        # Resume support: load weights before optimizers/schedulers are created so
        # the parameter groups inside optimizer reference the freshly-loaded params.
        if args.resume_student:
            student.load_state_dict(torch.load(args.resume_student, map_location=device))
            print(f"[resume] loaded student from {args.resume_student}")
        if args.resume_teacher:
            teacher.load_state_dict(torch.load(args.resume_teacher, map_location=device))
            print(f"[resume] loaded teacher from {args.resume_teacher}")

        epoch = 0

        snapshot_dir = os.path.join(args.snapshot_dir, args.exp_name)
        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"Snapshots will be saved to: {snapshot_dir}")

        print(f"==== train and evaluate unequal restart ====")
        # Schedule-Free / Hypergradient bypass policy entirely; they are pure baselines.
        sf_optimizer_active = args.scheduler in ('sf_sgd', 'sf_adamw')
        if sf_optimizer_active:
            try:
                import schedulefree
            except ImportError as exc:
                raise RuntimeError(
                    "Schedule-Free baseline requested via --scheduler "
                    f"{args.scheduler!r} but `schedulefree` is not installed. "
                    "Install it with `pip install schedulefree`."
                ) from exc
            if args.scheduler == 'sf_sgd':
                teacher_optimizer = schedulefree.SGDScheduleFree(
                    teacher.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=args.weight_decay, warmup_steps=args.sf_warmup_steps,
                )
                student_optimizer = schedulefree.SGDScheduleFree(
                    student.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=args.weight_decay, warmup_steps=args.sf_warmup_steps,
                )
            else:  # sf_adamw
                teacher_optimizer = schedulefree.AdamWScheduleFree(
                    teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                    warmup_steps=args.sf_warmup_steps,
                )
                student_optimizer = schedulefree.AdamWScheduleFree(
                    student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                    warmup_steps=args.sf_warmup_steps,
                )
            args._sf_optimizers = (teacher_optimizer, student_optimizer)
            print(f"[scheduler] Schedule-Free {args.scheduler} active "
                  f"(lr={args.lr}, wd={args.weight_decay}, warmup={args.sf_warmup_steps})")
        elif args.optimizer == 'sgd':
            teacher_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, params=teacher.parameters())
            student_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, params=student.parameters())
        else:
            raise NotImplementedError(f"{args.optimizer} optimizer is not supported")
        if args.scheduler=='cosine':
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=student_optimizer)
        if args.scheduler == 'custom':
            if not args.custom_lrs:
                raise ValueError("--custom_lrs must be provided for --scheduler custom")
            lrs = [float(lr) for lr in args.custom_lrs.split(',')]
            student_scheduler = MyLRScheduler(student_optimizer, lrs)
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
        if args.scheduler == 'dynamic':
            student_scheduler = DynamicScheduler(student_optimizer, init_lr=0.01)
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
        if args.scheduler == 'hypergrad':
            student_scheduler = HypergradScheduler(
                student_optimizer, init_lr=args.hypergrad_init_lr,
                beta=args.hypergrad_beta, lr_min=args.policy_min_lr,
                lr_max=args.policy_max_lr,
            )
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            print(f"[scheduler] Hypergradient active "
                  f"(init_lr={args.hypergrad_init_lr}, beta={args.hypergrad_beta})")
        if args.scheduler == 'adalrs':
            student_scheduler = AdaLRSScheduler(
                student_optimizer, init_lr=args.adalrs_init_lr,
                lr_min=args.policy_min_lr, lr_max=args.policy_max_lr,
                alpha=args.adalrs_alpha, beta=args.adalrs_beta,
                margin=args.adalrs_margin, clamp=args.adalrs_clamp,
            )
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            print(
                f"[scheduler] AdaLRS active init_lr={args.adalrs_init_lr} "
                f"alpha={args.adalrs_alpha} beta={args.adalrs_beta} "
                f"margin={args.adalrs_margin} clamp={args.adalrs_clamp}"
            )
        if args.scheduler in ('bandit_exp3', 'bandit_ucb'):
            mode = 'ucb' if args.scheduler == 'bandit_ucb' else 'exp3'
            student_scheduler = BanditLRScheduler(
                student_optimizer, LR_GRID, mode=mode,
                init_lr=args.bandit_init_lr, eta=args.bandit_eta,
                gamma=args.bandit_gamma, lr_min=args.policy_min_lr,
                lr_max=args.policy_max_lr,
            )
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            print(
                f"[scheduler] Bandit LR active mode={mode} init_lr={args.bandit_init_lr} "
                f"eta={args.bandit_eta} gamma={args.bandit_gamma}"
            )
        if args.scheduler == 'hypergrad_hb':
            student_scheduler = HypergradHBScheduler(
                student_optimizer, init_lr=args.hypergrad_init_lr,
                beta=args.hypergrad_hb_beta, momentum=args.hypergrad_hb_momentum,
                lr_min=args.policy_min_lr, lr_max=args.policy_max_lr,
            )
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            print(
                f"[scheduler] Hypergradient-HB active init_lr={args.hypergrad_init_lr} "
                f"beta={args.hypergrad_hb_beta} momentum={args.hypergrad_hb_momentum}"
            )
        if sf_optimizer_active:
            # No real LR-scheduler for SF; create wrapper exposing get_last_lr() = [args.lr].
            class _SFLRStub:
                def __init__(self, optimizer):
                    self.optimizer = optimizer
                def step(self):
                    pass
                def get_last_lr(self):
                    return [self.optimizer.param_groups[0]['lr']]
            teacher_scheduler = _SFLRStub(teacher_optimizer)
            student_scheduler = _SFLRStub(student_optimizer)
            
        # Advance teacher scheduler so its internal step count matches start_epoch.
        # Student scheduler is either DynamicScheduler (set per-epoch by policy) or
        # the cosine/SF/hypergrad ones — same advancement logic applies via .step().
        if args.start_epoch > 1:
            for _ in range(args.start_epoch - 1):
                teacher_scheduler.step()
                if args.scheduler == 'cosine':
                    student_scheduler.step()
            print(f"[resume] advanced teacher scheduler by {args.start_epoch - 1} steps; "
                  f"now teacher_lr={teacher_scheduler.get_last_lr()[0]:.6f}")
            if args.scheduler == 'cosine':
                print(f"[resume] student scheduler advanced as well; "
                      f"now student_lr={student_scheduler.get_last_lr()[0]:.6f}")

        # When resuming, seed the first feature with the actual epoch index so
        # time features (cosine LR / position) reflect where we are in the schedule.
        _seed_epoch = max(args.start_epoch - 1, 0)
        features_list.append(prepare_policy_feature(evaluate(teacher, student, test_loader, _seed_epoch)[2], _seed_epoch, args, student_scheduler))
        previous_dynamic_lr = None

        # Curve-mode online state.
        # ``curve_buffer[future_epoch] = list of (made_at_epoch, predicted_curve_tensor)``.
        curve_buffer = {}
        online_optimizer = None
        if args.policy_output == 'curve_argmin' and args.policy_online_lr > 0:
            if args.policy_online_optimizer == 'adam':
                online_optimizer = torch.optim.Adam(model_gru.parameters(), lr=args.policy_online_lr)
            else:
                online_optimizer = torch.optim.SGD(model_gru.parameters(), lr=args.policy_online_lr)
            print(
                f"[LR policy] online updates enabled optimizer={args.policy_online_optimizer} "
                f"lr={args.policy_online_lr:.2e} lookahead_n={args.policy_lookahead_n}"
            )

        # Side-channel for AdaLRS veto + Schedule-Free callbacks.
        args._student_test_losses = []
        args._last_policy_lr = None
        args._policy_pred_ema_mean = 0.0
        args._policy_pred_ema_initialized = False

        shock_active_log = []

        for epoch in range(args.start_epoch, args.epochs+1):
            # ---- Perturbation toggles for this epoch ------------------------
            # Activate / deactivate dataset noise based on start_epoch flags.
            ds = getattr(train_loader, "dataset", None)
            if ds is not None and hasattr(ds, "label_noise_active"):
                ds.label_noise_active = (epoch >= max(args.label_noise_start_epoch, 1))
                ds.input_noise_active = (epoch >= max(args.input_noise_start_epoch, 1))
            shock_active = (
                args.shock_epoch > 0
                and args.shock_epoch <= epoch < args.shock_epoch + max(args.shock_duration, 1)
            )
            if shock_active:
                # Hard-pin student LR for the duration of the shock, regardless
                # of scheduler / policy. Bypasses cosine.step() inside train()
                # by re-setting param_group lr before each forward pass: the
                # scheduler.step() at the end of train() will then advance
                # normally, but the next epoch's start will re-override if
                # still in the shock window.
                for pg in student_optimizer.param_groups:
                    pg['lr'] = float(args.shock_lr)
                if hasattr(student_scheduler, "_last_lr"):
                    student_scheduler._last_lr = [float(args.shock_lr)] * len(student_optimizer.param_groups)
                if hasattr(student_scheduler, "current_lr"):
                    student_scheduler.current_lr = float(args.shock_lr)
                print(
                    f"[shock] epoch={epoch} forcing student_lr={args.shock_lr:.6f} "
                    f"(window={args.shock_epoch}..{args.shock_epoch + args.shock_duration - 1})"
                )
                shock_active_log.append(epoch)
            # ----------------------------------------------------------------
            # Snapshot the LR that will actually be applied during this epoch's
            # training (captures shock_lr if shock active, or scheduler/policy
            # decision from previous epoch otherwise).
            try:
                applied_lr = float(student_optimizer.param_groups[0]['lr'])
                cosine_ref = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                noise_state = {
                    'label_noise_active': bool(getattr(ds, 'label_noise_active', False)) if ds is not None else False,
                    'input_noise_active': bool(getattr(ds, 'input_noise_active', False)) if ds is not None else False,
                    'label_noise_rate': float(args.label_noise_rate),
                    'input_noise_std': float(args.input_noise_std),
                }
                with open(os.path.join(snapshot_dir, 'epoch_state.jsonl'), 'a', encoding='utf-8') as _fs:
                    _fs.write(json.dumps({
                        'epoch': epoch,
                        'applied_student_lr': applied_lr,
                        'cosine_ref_lr': cosine_ref,
                        'shock_active': bool(shock_active),
                        'shock_lr': float(args.shock_lr) if shock_active else None,
                        **noise_state,
                    }) + '\n')
            except Exception:
                pass
            train(teacher, student, train_loader, epoch, args, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler)
            eval_out = evaluate(teacher, student, test_loader, epoch)
            args._student_test_losses.append(float(eval_out[1]))
            features_list.append(prepare_policy_feature(eval_out[2], epoch, args, student_scheduler))

            if args.scheduler == 'hypergrad':
                # Last feature is the 30-dim probe + (latent, time). Just use first 30.
                probe_curve = features_list[-1].detach().cpu().numpy()[:30]
                d_loss = student_scheduler.hypergrad_step(probe_curve, LR_GRID)
                print(f"[hypergrad] epoch={epoch} d_loss/d_lr={d_loss:.6f} "
                      f"new_lr={student_scheduler.get_last_lr()[0]:.6f}")
            elif args.scheduler == 'adalrs':
                probe_curve = features_list[-1].detach().cpu().numpy()[:30]
                cosine_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                decision = student_scheduler.adalrs_step(
                    epoch=epoch,
                    probe_curve=probe_curve,
                    lr_grid=LR_GRID,
                    cosine_lr=cosine_lr,
                    losses=args._student_test_losses,
                    warmup_epochs=args.policy_warmup_epochs,
                    bad_epochs=args.policy_veto_consecutive,
                )
                print(
                    f"[adalrs] epoch={epoch} decision={decision['decision']} "
                    f"best_lr={decision['best_lr']:.6f} cosine_lr={cosine_lr:.6f} "
                    f"improvement={decision['improvement']:.6f} "
                    f"next_student_lr={decision['next_lr']:.6f}"
                )
            elif args.scheduler in ('bandit_exp3', 'bandit_ucb'):
                cosine_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                decision = student_scheduler.bandit_step(
                    epoch=epoch,
                    current_loss=eval_out[1],
                    cosine_lr=cosine_lr,
                    warmup_epochs=args.policy_warmup_epochs,
                )
                print(
                    f"[bandit] epoch={epoch} mode={student_scheduler.mode} "
                    f"decision={decision['decision']} reward={decision['reward']:.6f} "
                    f"arm={decision['arm']} value={decision['value']:.6f} "
                    f"count={decision['count']} next_student_lr={decision['next_lr']:.6f}"
                )
            elif args.scheduler == 'hypergrad_hb':
                print(
                    f"[hypergrad_hb] epoch={epoch} signal={student_scheduler.last_signal:.6f} "
                    f"velocity={student_scheduler.velocity:.6f} "
                    f"updates={student_scheduler.updates} "
                    f"new_lr={student_scheduler.get_last_lr()[0]:.6f}"
                )
            elif args.scheduler == 'dynamic' and len(features_list) >= 1:
                if args.policy_output == 'curve_argmin':
                    deploy_curve_policy_step(
                        epoch=epoch,
                        args=args,
                        device=device,
                        model_gru=model_gru,
                        features_list=features_list,
                        curve_buffer=curve_buffer,
                        online_optimizer=online_optimizer,
                        student_scheduler=student_scheduler,
                        snapshot_dir=snapshot_dir,
                    )
                else:
                    model_gru.eval()

                    with torch.no_grad():
                        x_batch_gru = torch.stack(features_list[-args.policy_window:]).float().to(device)

                        x_batch_gru = x_batch_gru.unsqueeze(0)
                        policy_pred = float(model_gru(x_batch_gru).cpu().item())
                        if args.policy_output == 'cosine_multiplier':
                            policy_pred = float(np.clip(policy_pred, args.policy_multiplier_min, args.policy_multiplier_max))
                            base_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                            predicted_lr = float(np.clip(base_lr * policy_pred, args.policy_min_lr, args.policy_max_lr))
                        elif args.policy_output == 'residual_log':
                            # Decode as LR = cosine * exp(clip(out - running_mean, -1, +1)).
                            # The optional zero-mean EMA centers policy_pred to fix the
                            # ensemble-bias collapse described in improve5h_combined/analysis/report.md.
                            policy_pred_centered = policy_pred
                            if args.policy_zero_mean_ema > 0.0:
                                ema = float(np.clip(args.policy_zero_mean_ema, 0.0, 0.999))
                                if not args._policy_pred_ema_initialized:
                                    args._policy_pred_ema_mean = float(policy_pred)
                                    args._policy_pred_ema_initialized = True
                                else:
                                    args._policy_pred_ema_mean = ema * args._policy_pred_ema_mean + (1.0 - ema) * policy_pred
                                policy_pred_centered = policy_pred - args._policy_pred_ema_mean
                            log_ratio = float(np.clip(policy_pred_centered, -1.0, 1.0))
                            base_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                            predicted_lr = float(np.clip(base_lr * np.exp(log_ratio), args.policy_min_lr, args.policy_max_lr))
                        else:
                            predicted_lr = float(np.clip(policy_pred, args.policy_min_lr, args.policy_max_lr))
                    if epoch <= args.policy_warmup_epochs:
                        next_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                    else:
                        base_lr = cosine_annealing_lr(epoch, args.epochs, args.lr, args.policy_min_lr)
                        next_lr = predicted_lr
                        if args.policy_cosine_blend > 0:
                            blend = float(np.clip(args.policy_cosine_blend, 0.0, 1.0))
                            next_lr = (1.0 - blend) * next_lr + blend * base_lr
                        if args.policy_cosine_after_epoch > 0 and epoch >= args.policy_cosine_after_epoch:
                            next_lr = base_lr
                        if args.policy_lr_ema > 0 and previous_dynamic_lr is not None:
                            ema = float(np.clip(args.policy_lr_ema, 0.0, 0.99))
                            next_lr = ema * previous_dynamic_lr + (1.0 - ema) * next_lr
                        previous_dynamic_lr = next_lr
                    student_scheduler.set_lr(next_lr)
                    print(
                        f"[LR policy] epoch={epoch} output={args.policy_output} "
                        f"policy_pred={policy_pred:.6f} predicted_lr={predicted_lr:.6f} "
                        f"cosine_blend={args.policy_cosine_blend:.3f} lr_ema={args.policy_lr_ema:.3f} "
                        f"cosine_after={args.policy_cosine_after_epoch} "
                        f"next_student_lr={next_lr:.6f}"
                    )

            torch.save({'lr' : student_scheduler.get_last_lr()}, os.path.join(snapshot_dir, f'lr_data_epoch_{epoch}.pt'))

        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save(teacher.state_dict(), args.save+'_teacher.pt')
        torch.save(student.state_dict(), args.save+'_student.pt')
        print('ckpt location:', args.save)
        #wandb.finish()

    except Exception:
        logging.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
