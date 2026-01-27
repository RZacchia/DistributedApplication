from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"  # "cpu" also works

    # Data / FL
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # Non-IID
    dirichlet_alpha: float = 0.3  # lower => more non-IID

    # Split candidates (by layer index in models.py)
    cut_candidates: tuple = (1, 2, 3)  # early/mid/late cuts

    # Adaptive splitting "constraints"
    # We'll model each client with:
    # - uplink bandwidth (MB/s)
    # - compute speed scalar (higher=faster)
    # Adaptive policy picks a cut to minimize estimated time per step.
    simulate_network: bool = True

    # Prototype personalization
    use_prototypes: bool = True
    proto_momentum: float = 0.7
    proto_loss_weight: float = 0.1  # pull embeddings toward class prototypes
    proto_cond_head: bool = True     # condition classifier on prototypes (personalized)



@dataclass
class FedLFPConfig:
    # Paper hyperparameters
    T: int = 50            # total communication rounds (Algorithm 1, line 3)
    E: int = 2             # local epochs (Algorithm 1, line 20)
    gamma: float = 0.01    # local learning rate γ_i (Algorithm 1 input; line 27)
    K: int = 10            # target number of clusters on server (Algorithm 1 input; Eq. (5))
    lambda_cl: float = 0.1 # λ in Eq. (6)
    tau: float = 0.2       # temperature τ in Eq. (7)
    rho: float = 0.1       # ρ used for exp dissimilarity option (Section III-C-3)
    dissim_mode: str = "euclidean"  # {"euclidean", "exp"} (Eq. (9)-(10) discussion)
    device: str = "cuda"
    # Server KMeans options
    kmeans_iters: int = 25
    # Client optimizer
    momentum: float = 0.9
    class_weights: Optional[torch.Tensor] = None
