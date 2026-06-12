import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from torchvision.datasets import CIFAR10, CIFAR100
    from torchvision import transforms
    _TORCHVISION_IMPORT_ERROR = None
except Exception as exc:  # Allows wafer-only runs in environments with broken torchvision builds.
    CIFAR10 = CIFAR100 = None
    transforms = None
    _TORCHVISION_IMPORT_ERROR = exc
from sklearn.model_selection import train_test_split


WAFER_LABEL_KEYS = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring",
    "Loc", "Near_Full", "Scratch", "Random",
]


class ComposeTensor:
    def __init__(self, ops):
        self.ops = [op for op in ops if op is not None]

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x


class ResizeTensor:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, x):
        return torch.nn.functional.interpolate(
            x.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
        ).squeeze(0)


class RandomHorizontalFlipTensor:
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=(-1,))
        return x


class RandomVerticalFlipTensor:
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, x):
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=(-2,))
        return x


class RandomRot90Tensor:
    """Fallback augmentation used only when torchvision.transforms is unavailable."""

    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, x):
        if torch.rand(()) < self.p:
            k = int(torch.randint(0, 4, ()).item())
            return torch.rot90(x, k=k, dims=(-2, -1))
        return x


def _require_torchvision_for_cifar():
    if transforms is None or CIFAR10 is None or CIFAR100 is None:
        raise RuntimeError(
            "CIFAR mode requires a working torchvision installation; "
            f"import failed with: {_TORCHVISION_IMPORT_ERROR}"
        )


def _compose(ops):
    if transforms is not None:
        return transforms.Compose(ops) if ops else None
    return ComposeTensor(ops)


def _wafer_resize(size):
    if transforms is not None:
        return transforms.Resize((size, size))
    return ResizeTensor((size, size))


def _wafer_hflip():
    if transforms is not None:
        return transforms.RandomHorizontalFlip()
    return RandomHorizontalFlipTensor()


def _wafer_vflip():
    if transforms is not None:
        return transforms.RandomVerticalFlip()
    return RandomVerticalFlipTensor()


def _wafer_rotation(degrees=15):
    if transforms is not None:
        return transforms.RandomRotation(degrees)
    return RandomRot90Tensor(p=0.25)


class NoisyClassificationDataset(Dataset):
    """Wraps a single-label classification dataset and injects label/input noise."""

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


class NoisyMultilabelDataset(Dataset):
    """Wraps a multilabel dataset and flips each label bit with probability label_noise_rate."""

    def __init__(self, base, label_noise_rate=0.0, input_noise_std=0.0, seed=0):
        self.base = base
        self.label_noise_rate = float(label_noise_rate)
        self.input_noise_std = float(input_noise_std)
        self.label_noise_active = True
        self.input_noise_active = True
        self.rng = np.random.RandomState(int(seed) + 54321)

        labels = np.asarray(getattr(base, "labels", None), dtype=np.float32)
        if labels.size == 0:
            raise ValueError("NoisyMultilabelDataset: base dataset has no .labels attribute")
        self._clean_targets = labels
        self._noisy_targets = labels.copy()
        n_flips = 0
        if self.label_noise_rate > 0.0:
            mask = self.rng.random(labels.shape) < self.label_noise_rate
            self._noisy_targets[mask] = 1.0 - self._noisy_targets[mask]
            n_flips = int(mask.sum())
        print(
            f"[NoisyMultilabelDataset] label_noise_rate={self.label_noise_rate:.3f} "
            f"input_noise_std={self.input_noise_std:.4f} bit_flips={n_flips}/{labels.size}"
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        if self.input_noise_active and self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std
        target = self._noisy_targets[idx] if self.label_noise_active else self._clean_targets[idx]
        return x, torch.tensor(target, dtype=torch.float32)


class WaferDataset(Dataset):
    """Dataset for MixedWM38-style wafer maps stored in an .npz file.

    Expected arrays are ``arr_0`` = images and ``arr_1`` = multilabel targets.
    Images may be stored as [N,H,W], [N,1,H,W], [N,H,W,1], or already RGB.
    Labels are returned as float32 vectors for BCEWithLogitsLoss.
    """

    def __init__(self, images, labels, transform=None):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels, dtype=np.float32)
        if self.labels.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _to_chw_tensor(image):
        x = np.asarray(image)
        if x.ndim == 2:
            x = x[None, :, :]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3):
                pass
            elif x.shape[-1] in (1, 3):
                x = np.transpose(x, (2, 0, 1))
            else:
                # Treat [H,W,C?] unknown as a single-channel map by taking first channel.
                x = x[..., 0][None, :, :]
        else:
            raise ValueError(f"Unsupported wafer image shape: {x.shape}")
        x = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
        if x.numel() and float(x.max()) > 1.0:
            x = x / 255.0
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x

    def __getitem__(self, idx):
        x = self._to_chw_tensor(self.images[idx])
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def _wafer_stratify_codes(labels):
    """Return robust stratification codes for multilabel rows, or None if unsafe."""
    arr = np.asarray(labels)
    if arr.ndim == 1:
        codes = arr.astype(str)
    else:
        codes = np.array(["".join(map(str, row.astype(int).tolist())) for row in arr])
    _, counts = np.unique(codes, return_counts=True)
    if len(counts) <= 1 or counts.min() < 2:
        print("[wafer] stratified split disabled: at least one label-combination has <2 samples")
        return None
    return codes


def _deduplicate_images(images, labels):
    flat = images.reshape(images.shape[0], -1)
    _, unique_indices = np.unique(flat, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    if len(unique_indices) != len(images):
        print(f"[wafer] removed duplicates: {len(images) - len(unique_indices)}")
    return images[unique_indices], labels[unique_indices]


def image_transform(args):
    if args.dataset == 'cifar100':
        _require_torchvision_for_cifar()
        args.num_classes = 100
        args.is_multilabel = False
        offset = (0.5, 0.5, 0.5)
        scale = (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean=offset, std=scale)
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=args.input_size, padding=4),
            _wafer_hflip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    elif args.dataset == 'cifar10':
        _require_torchvision_for_cifar()
        args.num_classes = 10
        args.is_multilabel = False
        offset = (0.5, 0.5, 0.5)
        scale = (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean=offset, std=scale)
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=args.input_size, padding=4),
            _wafer_hflip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    elif args.dataset in ('wafer', 'mydataset'):
        args.num_classes = int(getattr(args, 'wafer_num_classes', 8))
        args.is_multilabel = True
        resize = int(getattr(args, 'wafer_resize', 56) or 0)
        if resize > 0:
            args.input_size = resize
        train_ops = []
        test_ops = []
        if resize > 0:
            train_ops.append(_wafer_resize(resize))
            test_ops.append(_wafer_resize(resize))
        if bool(getattr(args, 'wafer_augment', True)):
            train_ops.extend([
                _wafer_hflip(),
                _wafer_vflip(),
                _wafer_rotation(15),
            ])
        train_transform = _compose(train_ops) if train_ops else None
        test_transform = _compose(test_ops) if test_ops else None
    else:
        raise NotImplementedError(f"{args.dataset} not supported")
    return train_transform, test_transform


def get_torch_dataset(args):
    data_path = getattr(args, "datadir", "data")
    download = bool(getattr(args, "download", False))
    train_transform, test_transform = image_transform(args)

    if args.dataset in ("cifar10", "cifar100"):
        dataset_cls = CIFAR10 if args.dataset == "cifar10" else CIFAR100
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

    elif args.dataset in ("wafer", "mydataset"):
        wafer_npz = getattr(args, "wafer_npz", "") or os.path.join(data_path, "Wafer_Map_Datasets.npz")
        if not os.path.exists(wafer_npz):
            raise FileNotFoundError(
                f"Wafer dataset not found: {wafer_npz}. "
                "Pass --wafer_npz /path/to/Wafer_Map_Datasets.npz"
            )
        data = np.load(wafer_npz)
        images = data["arr_0"]
        labels = data["arr_1"].astype(np.float32)
        if bool(getattr(args, "wafer_deduplicate", True)):
            images, labels = _deduplicate_images(images, labels)
        labels = (labels > 0).astype(np.float32)
        stratify = _wafer_stratify_codes(labels) if bool(getattr(args, "wafer_stratify", True)) else None
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                images,
                labels,
                test_size=float(getattr(args, "wafer_test_size", 0.2)),
                random_state=int(getattr(args, "seed", 42)),
                stratify=stratify,
            )
        except ValueError as exc:
            print(f"[wafer] stratified split failed ({exc}); falling back to random split")
            x_train, x_test, y_train, y_test = train_test_split(
                images,
                labels,
                test_size=float(getattr(args, "wafer_test_size", 0.2)),
                random_state=int(getattr(args, "seed", 42)),
                stratify=None,
            )
        train_set = WaferDataset(x_train, y_train, transform=train_transform)
        test_set = WaferDataset(x_test, y_test, transform=test_transform)

        label_noise_rate = float(getattr(args, "label_noise_rate", 0.0) or 0.0)
        input_noise_std = float(getattr(args, "input_noise_std", 0.0) or 0.0)
        if label_noise_rate > 0.0 or input_noise_std > 0.0:
            train_set = NoisyMultilabelDataset(
                train_set,
                label_noise_rate=label_noise_rate,
                input_noise_std=input_noise_std,
                seed=int(getattr(args, "seed", 0)),
            )
    else:
        raise NotImplementedError(f"{args.dataset} not supported")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print(
        f'Dataset information: {args.dataset}\t {len(train_set)} images for training\t '
        f'{len(test_set)} images for testing\t num_classes={args.num_classes}\t '
        f'multilabel={getattr(args, "is_multilabel", False)}'
    )
    return train_loader, test_loader
