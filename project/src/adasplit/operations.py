import torch
import random

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize last dimension."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Inner-product similarity used in the paper's contrastive terms
    (see Eq. (7)-(10): sim(·,·) is inner product).
    """
    return (a * b).sum(dim=-1)