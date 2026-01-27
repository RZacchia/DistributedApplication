# ------------------------------
# CIFAR-10 utilities (Dirichlet Non-IID partition as in experiments)
# ------------------------------

import random

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset




def dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 0,
) -> List[List[int]]:
    """
    Partition indices among clients with class-wise Dirichlet allocation (widely used in FL),
    matching the paper's "Dir(α)" Non-IID simulation (Section V-C, referencing FedAvg [25]).

    NOTE (PyTorch API compatibility):
      torch.distributions.Dirichlet.sample() does NOT accept a `generator=` argument on many
      PyTorch versions. To keep deterministic behavior, we set the global torch RNG seed
      locally via fork_rng and then call .sample() without passing a generator.
    """
    targets_t = torch.tensor(list(targets), dtype=torch.long)
    num_classes = int(targets_t.max().item() + 1)
    idx_by_class = [torch.where(targets_t == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    # Keep determinism without relying on generator support in Dirichlet.sample()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        for c in range(num_classes):
            idx_c = idx_by_class[c]
            if len(idx_c) == 0:
                continue

            # Shuffle indices within class (randperm is deterministic under manual_seed)
            idx_c = idx_c[torch.randperm(len(idx_c))]

            # Sample Dirichlet proportions (deterministic under manual_seed)
            proportions = torch.distributions.Dirichlet(
                torch.full((num_clients,), float(alpha))
            ).sample()

            # Convert to counts that sum to len(idx_c)
            counts = torch.round(proportions * len(idx_c)).long()

            # Fix rounding to exact total
            diff = len(idx_c) - int(counts.sum().item())
            if diff != 0:
                order = torch.argsort(proportions, descending=True)
                for j in range(abs(diff)):
                    k = int(order[j % num_clients].item())
                    counts[k] += 1 if diff > 0 else -1
            counts = counts.clamp_min(0)

            # Assign slices
            start = 0
            for i in range(num_clients):
                end = start + int(counts[i].item())
                if end > start:
                    client_indices[i].extend(idx_c[start:end].tolist())
                start = end

            # If any leftovers (due to clamping), distribute round-robin
            if start < len(idx_c):
                leftovers = idx_c[start:].tolist()
                for j, idx in enumerate(leftovers):
                    client_indices[j % num_clients].append(idx)

    # Final shuffle per client (python RNG; tie to seed for reproducibility)
    rnd = random.Random(seed)
    for i in range(num_clients):
        rnd.shuffle(client_indices[i])

    return client_indices



from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class KronoDroidStats:
    mean: np.ndarray  # (F,)
    std: np.ndarray   # (F,)


class KronoDroidNPZ(Dataset):
    """
    Loads kronodroid_{train,test}.npz created by your preprocessing script.

    NPZ format:
      - X: (N, F) float32 features
      - y: (N,) int64 labels in [0..12] (13 classes total)

    IMPORTANT:
      Your current LeNet-5 expects 32x32 inputs (so conv/pool -> 16*5*5).
      Therefore we pad features to 32*32=1024 and reshape to (C,32,32).

      Output per sample:
        x: (3, 32, 32) float32
        y: scalar long
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        stats: Optional[KronoDroidStats] = None,
        compute_stats: bool = False,
        in_channels: int = 3,
        image_side: int = 18,
    ):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"KronoDroid NPZ not found: {npz_path}")

        obj = np.load(npz_path, allow_pickle=False)
        if "X" not in obj or "y" not in obj:
            raise ValueError(f"{npz_path} must contain keys 'X' and 'y'.")

        X = obj["X"].astype(np.float32)
        y = obj["y"].astype(np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,F). Got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be 1D with same N as X. Got X={X.shape}, y={y.shape}")

        # compute stats on train split
        if compute_stats:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std = np.where(std < 1e-12, 1.0, std).astype(np.float32)
            stats = KronoDroidStats(mean=mean.astype(np.float32), std=std)

        if stats is None:
            raise ValueError("Provide stats=... or set compute_stats=True for training split.")

        # normalize
        X = (X - stats.mean) / stats.std

        # pad to image_side * image_side
        target = image_side * image_side
        F = X.shape[1]
        if F > target:
            raise ValueError(f"Feature dim F={F} exceeds {image_side}*{image_side}={target}.")
        if F < target:
            X = np.pad(X, ((0, 0), (0, target - F)), mode="constant", constant_values=0.0)

        # reshape to 1x32x32 then repeat to 3 channels for LeNet
        X = X.reshape(-1, 1, image_side, image_side)  # (N,1,18,18)
        if in_channels == 3:
            X = np.repeat(X, 3, axis=1)               # (N,3,18,18)
        elif in_channels != 1:
            raise ValueError("in_channels must be 1 or 3")

        self.X = X
        self.y = y
        self.stats = stats

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def load_kronodroid_npz(
    data_dir: Union[str, Path],
    train_name: str = "kronodroid_train.npz",
    test_name: str = "kronodroid_test.npz",
    in_channels: int = 3,
    image_side: int = 18,
) -> Tuple[KronoDroidNPZ, KronoDroidNPZ]:
    """
    Loads NPZ train/test with shared normalization stats (train mean/std).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / train_name
    test_path = data_dir / test_name

    train_ds = KronoDroidNPZ(
        train_path, compute_stats=True, in_channels=in_channels, image_side=image_side
    )
    test_ds = KronoDroidNPZ(
        test_path, stats=train_ds.stats, in_channels=in_channels, image_side=image_side
    )
    return train_ds, test_ds
