import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms


class NoisyClassificationDataset(Dataset):
    """Wraps a classification dataset and injects two kinds of perturbations:

    - Symmetric label flipping: with probability ``label_noise_rate`` each
      training sample is assigned a uniformly random *different* class.
      Flips are pre-computed once per (dataset, seed) so they are deterministic
      across epochs.
    - Per-sample Gaussian input noise: at ``__getitem__`` time the (already
      normalized) tensor is perturbed by ``N(0, input_noise_std)`` noise.
      Test set should not be wrapped.

    Both perturbations can be turned on/off mid-training by setting
    ``self.label_noise_active`` / ``self.input_noise_active`` (default True).
    This lets the trainer simulate "noise appearance at epoch N" experiments
    without rebuilding the dataset.
    """

    def __init__(self, base, num_classes, label_noise_rate=0.0,
                 input_noise_std=0.0, seed=0):
        self.base = base
        self.num_classes = int(num_classes)
        self.label_noise_rate = float(label_noise_rate)
        self.input_noise_std = float(input_noise_std)
        self.label_noise_active = True
        self.input_noise_active = True

        clean = np.asarray(getattr(base, "targets", None), dtype=np.int64)
        if clean.size == 0:
            raise ValueError("NoisyClassificationDataset: base dataset has no .targets attribute")

        noisy = clean.copy()
        n_flips = 0
        if self.label_noise_rate > 0.0:
            rng = np.random.RandomState(int(seed) + 12345)
            mask = rng.random(clean.shape[0]) < self.label_noise_rate
            for i in np.where(mask)[0]:
                t = int(clean[i])
                # Sample a class != t uniformly.
                alt = int(rng.randint(0, self.num_classes - 1))
                if alt >= t:
                    alt += 1
                noisy[i] = alt
                n_flips += 1
        self._noisy_targets = noisy
        self._clean_targets = clean
        print(
            f"[NoisyClassificationDataset] num_classes={self.num_classes} "
            f"label_noise_rate={self.label_noise_rate:.3f} "
            f"input_noise_std={self.input_noise_std:.4f} "
            f"flipped={n_flips}/{clean.shape[0]}"
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        if self.label_noise_active and self.label_noise_rate > 0.0:
            target = int(self._noisy_targets[idx])
        else:
            target = int(self._clean_targets[idx])
        if self.input_noise_active and self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return x, target


def image_transform(args):
    if args.dataset=='cifar100':
        mean_statistics = (0.5071, 0.4867, 0.4408)
        std_statistics = (0.2675, 0.2565, 0.2761)
        max_values = (1.0, 1.0, 1.0)
        min_values = (0.0, 0.0, 0.0)
        args.num_classes=100
    elif args.dataset=='cifar10':
        mean_statistics = (0.4914, 0.4822, 0.4465)
        std_statistics = (0.2470, 0.2435, 0.2616)
        max_values = (1.0, 1.0, 1.0)
        min_values = (0.0, 0.0, 0.0)
        args.num_classes=10
    else:
        raise NotImplementedError(f"{args.dataset} not supported")
    offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_values, max_values)]
    scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_values, min_values)]
    normalize = transforms.Normalize(mean=offset, std=scale)
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=args.input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, test_transform


def get_torch_dataset(args):
    data_path = getattr(args, "datadir", "data")
    download = bool(getattr(args, "download", False))
    train_transform, test_transform = image_transform(args)
    if args.dataset == "cifar10":
        dataset_cls = CIFAR10
    elif args.dataset == "cifar100":
        dataset_cls = CIFAR100
    else:
        raise NotImplementedError(f"{args.dataset} not supported")

    train_set = dataset_cls(data_path, train=True, transform=train_transform, download=download)
    test_set = dataset_cls(data_path, train=False, transform=test_transform, download=download)

    label_noise_rate = float(getattr(args, "label_noise_rate", 0.0) or 0.0)
    input_noise_std = float(getattr(args, "input_noise_std", 0.0) or 0.0)
    if label_noise_rate > 0.0 or input_noise_std > 0.0:
        train_set = NoisyClassificationDataset(
            train_set,
            num_classes=args.num_classes,
            label_noise_rate=label_noise_rate,
            input_noise_std=input_noise_std,
            seed=int(getattr(args, "seed", 0)),
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f'Dataset information: {args.dataset}\t {len(train_set)} images for training \t {len(test_set)} images for testing\t')
    return train_loader, test_loader

