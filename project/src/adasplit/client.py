from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from adasplit.configs import FedLFPConfig
from adasplit.models import LeNet5ClientHead, LeNet5FeatureExtractor
from adasplit.operations import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
# ------------------------------
# FedLFP Client
# ------------------------------

class FedLFPClient:
    """
    One client i, holding ω_i = {θ_i, Φ_i} (paper notation).
    - θ_i: feature extractor params for f(θ_i;·)
    - Φ_i: classifier params for h(Φ_i;·)

    Implements Algorithm 1: Clientupdate(ω_i, GP) and prototype upload.
    """
    def __init__(
        self,
        client_id: int,
        f_theta: nn.Module,
        h_phi: nn.Module,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        num_classes: int,
        cfg: FedLFPConfig,
    ):
        self.client_id = client_id
        self.f = f_theta
        self.h = h_phi
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.f.to(self.device)
        self.h.to(self.device)

        self.opt = torch.optim.SGD(
            list(self.f.parameters()) + list(self.h.parameters()),
            lr=cfg.gamma,
            momentum=cfg.momentum,
        )

        # Cached local prototype data to upload (LPi, Qi, Si)
        self.LPi: Optional[torch.Tensor] = None
        self.Qi: Optional[torch.Tensor] = None
        self.Si: Optional[torch.Tensor] = None

    def client_update(self, GP: Optional[torch.Tensor]) -> None:
        """
        Algorithm 1: procedure Clientupdate(ω_i, GP), lines 19-34.
        - If GP is Null: use cross-entropy only (line 22-24).
        - Else: use full loss Eq. (6) with (7)-(10) (line 25, via Eq. (9) in paper).
        """
        self.f.train()
        self.h.train()

        for _e in range(self.cfg.E):  # Algorithm 1, line 20
            for x, y in self.dataloader:  # Algorithm 1, line 21
                x = x.to(self.device)
                y = y.to(self.device)

                z = self.f(x)         # representations z_a
                logits = self.h(z)    # predictions for L_ce
                Lce = F.cross_entropy(logits, y)  # Eq. (6) L_ce

                if GP is None:
                    # Algorithm 1, line 22-24
                    L = Lce
                else:
                    # Eq. (7): supervised contrastive loss L_batch
                    Lbatch = Lbatch_supervised_contrastive(z, y, tau=self.cfg.tau)

                    # Compute local prototypes for this client (needed for Eq. (10))
                    # For efficiency, we compute per-mini-batch prototypes (approx) during training.
                    # In the paper, LP_i is computed after local training (Algorithm 1 line 30),
                    # but Eq. (10) conceptually needs local prototypes; batch-approx is common.
                    LPi_batch = self._batch_local_prototypes(z, y)

                    # Eq. (8)-(10): L_global (unsupervised with global prototypes)
                    Lglobal = Lglobal_unsupervised_contrastive(
                        Z=z,
                        LPi=LPi_batch,
                        GP=GP,
                        dissim_mode=self.cfg.dissim_mode,
                        rho=self.cfg.rho,
                    )

                    # Eq. (6): total loss
                    L = Lce + self.cfg.lambda_cl * (Lbatch + Lglobal)

                self.opt.zero_grad(set_to_none=True)
                L.backward()
                self.opt.step()

        # After local training in the round: compute {LPi, Qi, Si} (Algorithm 1 line 30-33)
        self.LPi, self.Qi, self.Si = compute_LP_Q_S(
            f_theta=self.f,
            h_phi=self.h,
            dataloader=self.dataloader,
            num_classes=self.num_classes,
            device=self.device,
        )

    @staticmethod
    def _batch_local_prototypes(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute per-class mean prototypes within a batch (normalized)."""
        protos = []
        for cls in y.unique().tolist():
            mask = (y == cls)
            protos.append(normalize(z[mask].mean(dim=0)))
        if len(protos) == 0:
            return z.new_zeros((0, z.shape[1]))
        return torch.stack(protos, dim=0)



# ------------------------------
# Client-side: prototypes and losses
# ------------------------------


def compute_LP_Q_S(
    f_theta: nn.Module,
    h_phi: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute label-free local prototype data {LP_i, Q_i, S_i} (Section III-C-1).
    - Eq. (2): local prototype l^i_k = normalize(mean_{(x,y) in D^i_k} f(θ_i; x))
    - Eq. (3): confidence score s^i_k = softmax(h(Φ_i; l^i_k))
    - Q_i: sample count q^i_k = |D^i_k| (described after Eq. (3))

    Returns:
        LPi: (Ci, d) local prototypes for classes present on this client
        Qi:  (Ci,) sample counts for those prototypes
        Si:  (Ci,) confidence scores for those prototypes (max prob)
    """
    f_theta.eval()
    h_phi.eval()

    # Accumulate sums and counts per class
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    with torch.no_grad():
        sums = {}
        counts = {}

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            z = f_theta(x)

            for cls in y.unique().tolist():
                mask = (y == cls)
                z_sum = z[mask].sum(dim=0)

                if cls not in sums:
                    sums[cls] = z_sum.detach().clone()
                    counts[cls] = int(mask.sum().item())
                else:
                    sums[cls] += z_sum
                    counts[cls] += int(mask.sum().item())

            classes = sorted(sums.keys())
    if len(classes) == 0:
        # No data
        return (torch.empty(0, 1, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device))

    # Eq. (2) prototypes (mean then normalize)
    LP, Q = [], []
    for cls in sums:
        proto = sums[cls] / counts[cls]
        proto = F.normalize(proto, dim=0)
        LP.append(proto)
        Q.append(counts[cls])

    LPi = torch.stack(LP, dim=0)                  # (Ci, d)
    Qi = torch.tensor(Q, device=device).float()   # (Ci,)

    # Eq. (3): confidence score from classifier on prototype; use max softmax prob
    logits = h_phi(LPi)                                # (Ci, C_global)
    probs = F.softmax(logits, dim=-1)
    Si = probs.max(dim=-1).values                      # (Ci,)

    return LPi, Qi, Si


def Lbatch_supervised_contrastive(
    Z: torch.Tensor,
    y: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Supervised contrastive learning loss L_batch (Eq. (7)).
    Paper notation:
      - batch samples a, b in B; positives are same label (Bi(a))
      - sim is inner product; temperature τ.

    Implementation:
      - Standard supervised contrastive loss from Khosla et al. (2020) style,
        but simplified to match Eq. (7) form.
    """
    Z = normalize(Z)
    N = Z.shape[0]
    if N <= 1:
        return Z.new_tensor(0.0)

    # Similarity matrix scaled by tau
    S = (Z @ Z.t()) / max(tau, 1e-12)  # (N, N)
    # Mask out self-contrast
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=Z.device)
    S = S.masked_fill(~logits_mask, float("-inf"))

    y = y.view(-1, 1)
    pos_mask = (y == y.t()) & logits_mask  # positives excluding self
    # For each anchor, compute log prob over all except self
    log_prob = S - torch.logsumexp(S, dim=1, keepdim=True)

    # Average over positives per anchor, then over anchors
    pos_counts = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_counts)
    return loss.mean()


def dissim_weight(
    A: torch.Tensor,
    B: torch.Tensor,
    mode: str = "euclidean",
    rho: float = 0.1,
) -> torch.Tensor:
    """
    Dissimilarity weight dissim(·,·) used in Eq. (9)-(10) discussion.
    Paper suggests:
      - Euclidean dissimilarity ||a - b||^2
      - or exponential dissimilarity exp(-ρ * sim(a,b))

    Returns:
        (N, K) weight matrix.
    """
    if mode == "euclidean":
        # ||A - B||^2
        # A: (N, d), B: (K, d)
        dist2 = torch.cdist(A, B, p=2) ** 2
        return dist2
    elif mode == "exp":
        # exp(-ρ * sim(a,b)), using inner product similarity (paper text below Eq. (9))
        A_n = normalize(A)
        B_n = normalize(B)
        s = A_n @ B_n.t()
        return torch.exp(-rho * s)
    else:
        raise ValueError(f"Unknown dissim_mode: {mode}")


def Lglobal_unsupervised_contrastive(
    Z: torch.Tensor,
    LPi: torch.Tensor,
    GP: torch.Tensor,
    dissim_mode: str,
    rho: float,
) -> torch.Tensor:
    """
    Unsupervised contrastive loss L_global = L_gp_emb + L_gp_lp (Eq. (8)-(10)).
    - Eq. (9): sample-level alignment between local representations z_a and global prototypes g_k
    - Eq. (10): prototype-level alignment between local prototypes l^i_a and global prototypes g_k
    - Both weighted by dissim(·,·) emphasizing harder pairs (paper text in Section III-C-3).

    Notes:
    - The paper writes exp(sim(·,·)) without explicit temperature. We follow that literally.
    - We keep everything normalized for stability.
    """
    if GP is None or GP.numel() == 0:
        return Z.new_tensor(0.0)

    Z = normalize(Z)
    GP = normalize(GP)

    # ----- L_gp_emb (Eq. (9)) -----
    # For each sample z_a, compute softmax over global prototypes.
    logits_emb = Z @ GP.t()  # (N, K) equals sim(z_a, g_k)
    prob_emb = torch.softmax(logits_emb, dim=1)  # (N, K)
    w_emb = dissim_weight(Z, GP, mode=dissim_mode, rho=rho)  # (N, K)
    Lgp_emb = (w_emb * prob_emb).sum(dim=1).mean()

    # ----- L_gp_lp (Eq. (10)) -----
    if LPi is None or LPi.numel() == 0:
        Lgp_lp = Z.new_tensor(0.0)
    else:
        LPi = normalize(LPi)
        logits_lp = LPi @ GP.t()   # (Ci, K)
        prob_lp = torch.softmax(logits_lp, dim=1)
        w_lp = dissim_weight(LPi, GP, mode=dissim_mode, rho=rho)
        Lgp_lp = (w_lp * prob_lp).sum(dim=1).mean()

    # Eq. (8)
    return Lgp_emb + Lgp_lp

