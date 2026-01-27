from fedLFP.configs import FedLFPConfig
from fedLFP.operations import normalize
import torch
from typing import Optional, List, Tuple



# ------------------------------
# Weighted K-Means (server-side)
# ------------------------------

def weighted_kmeans(
    X: torch.Tensor,
    W: torch.Tensor,
    K: int,
    num_iters: int = 25,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    Server-side weighted K-Means to produce GP (global prototypes).
    - Corresponds to Eq. (5): GP ← KMeans(LP, LW, K)
    - The paper states K-Means clustering on LP using clustering weights LW.

    Args:
        X: (N, d) local prototype matrix, i.e., LP = (l1,...,l|LP|).
        W: (N,) weights LW = (lw1,...,lw|LP|), Eq. (4).
        K: number of clusters/global prototypes.
        num_iters: max iterations.
        tol: relative center movement tolerance.

    Returns:
        centers: (K, d) centroid prototypes GP = (g1,...,gK).
    """
    assert X.ndim == 2 and W.ndim == 1 and X.shape[0] == W.shape[0]
    N, d = X.shape
    device = X.device

    # Initialize centers with weighted sampling (kmeans++-ish simplified)
    probs = (W / (W.sum() + 1e-12)).clamp_min(1e-12)
    init_idx = torch.multinomial(probs, num_samples=K, replacement=(N < K))
    centers = X[init_idx].clone()

    prev = centers.clone()
    for _ in range(num_iters):
        # Assign: nearest center in Euclidean distance
        # Note: prototypes are normalized per Eq. (2), so Euclidean is fine.
        dist2 = torch.cdist(X, centers, p=2) ** 2  # (N, K)
        assign = dist2.argmin(dim=1)  # (N,)

        # Update weighted centroids
        new_centers = torch.zeros((K, d), device=device, dtype=X.dtype)
        for k in range(K):
            mask = (assign == k)
            if mask.any():
                wk = W[mask].unsqueeze(1)  # (nk, 1)
                new_centers[k] = (wk * X[mask]).sum(dim=0) / (wk.sum() + 1e-12)
            else:
                # Empty cluster: re-sample a point by weight
                ridx = torch.multinomial(probs, num_samples=1).item()
                new_centers[k] = X[ridx]
        centers = new_centers

        # Convergence check
        move = (centers - prev).norm(p=2) / (prev.norm(p=2) + 1e-12)
        if move.item() < tol:
            break
        prev = centers.clone()

    # Optional: normalize GP to stabilize contrastive alignment (consistent with Eq. (2) normalization)
    centers = normalize(centers)
    return centers





# ------------------------------
# FedLFP Server
# ------------------------------

class FedLFPServer:
    """
    Server executes Algorithm 1, lines 1-18:
    - Receive {LPi, Qi, Si}
    - Aggregate to {LP, Q, S} (line 6; Section III-C-2)
    - Compute combined clustering weight LW (Eq. (4))
    - Cluster LP with weights and generate GP (Eq. (5))
    - Distribute GP
    """
    def __init__(self, cfg: FedLFPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.LP: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None
        self.LW: Optional[torch.Tensor] = None
        self.GP: Optional[torch.Tensor] = None


    def aggregate(self, client_payloads: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        """
        Aggregate label-free local prototype data from participating clients At:
        - Algorithm 1, line 5-6
        - Paper notation (Section III-C-2):
            LP = (LP1,...,LPM) = (l1,...,l_|LP|)
            Q  = (q1,...,q_|LP|)
            S  = (s1,...,s_|LP|)
        """
        LP_list, Q_list, S_list = [], [], []
        for LPi, Qi, Si in client_payloads:
            if LPi is None or LPi.numel() == 0:
                continue
            LP_list.append(LPi.detach().to(self.device))
            Q_list.append(Qi.detach().to(self.device))
            S_list.append(Si.detach().to(self.device))

        if len(LP_list) == 0:
            self.LP = self.Q = self.S = self.LW = self.GP = None
            return

        self.LP = torch.cat(LP_list, dim=0)  # (|LP|, d)
        self.Q = torch.cat(Q_list, dim=0)    # (|LP|,)
        self.S = torch.cat(S_list, dim=0)    # (|LP|,)

    def compute_LW(self) -> torch.Tensor:
        """
        Compute combined clustering weights LW (Eq. (4)):
            lwr = qwr + swr
                = |LP| * qr / sum_i qi + |LP| * sr / sum_i si
        This balances data quantity (Q_Weight) and confidence (S_Weight).
        """
        assert self.LP is not None and self.Q is not None and self.S is not None
        n = self.LP.shape[0]

        q_sum = self.Q.sum() + 1e-12
        s_sum = self.S.sum() + 1e-12

        qwr = n * self.Q / q_sum
        swr = n * self.S / s_sum
        self.LW = qwr + swr  # Eq. (4)
        return self.LW

    def compute_GP(self) -> Optional[torch.Tensor]:
        """
        Compute global prototypes GP (Algorithm 1 line 7-12; Eq. (5)).
        - If |LP| < K: GP = Null
        - Else: perform weighted K-Means with LW and output centroids.
        """
        if self.LP is None or self.LP.shape[0] < self.cfg.K:
            self.GP = None
            return None

        LW = self.compute_LW()
        self.GP = weighted_kmeans(
            X=normalize(self.LP),
            W=LW,
            K=self.cfg.K,
            num_iters=self.cfg.kmeans_iters,
        )
        return self.GP