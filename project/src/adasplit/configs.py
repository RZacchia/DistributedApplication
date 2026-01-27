from dataclasses import dataclass
from typing import Optional

import torch



@dataclass
class FedLFPConfig:
    # Paper hyperparameters
    T: int = 20            # total communication rounds (Algorithm 1, line 3)
    E: int = 1             # local epochs (Algorithm 1, line 20)
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
