# ------------------------------
# CIFAR-10 utilities (Dirichlet Non-IID partition as in experiments)
# ------------------------------

import random
from typing import List, Sequence

import torch


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