
"""
FedLFP: Communication-Efficient Personalized Federated Learning on Non-IID Data in MEC
====================================================================================

This file provides a faithful, *paper-naming-scheme* implementation of FedLFP
(Algorithm 1; Eqs. (2)-(10)).

Paper: "FedLFP: Communication-Efficient Personalized Federated Learning on Non-IID Data
in Mobile Edge Computing Environments", IEEE TMC, 2025.
- See Algorithm 1 and Eqs. (2)-(10) for the core method.

Design goals
------------
1) Keep variable names consistent with the paper: LP, Q, S, GP, LW, lwr, etc.
2) Include in-code comments pointing to the paper sections/equations.
3) Minimal assumptions: PyTorch is required; everything else is pure Python.

What this is (and isn't)
------------------------
- This is a research-grade reference implementation of the *FedLFP algorithmic core*.
- It is not a full benchmark runner (datasets, Non-IID partitioners, etc.),
  but includes a small demo and the hooks you'll need to plug in real data.

Dependencies
------------
- torch

Usage (quick demo)
------------------
python fedlfp.py --demo

To integrate with your own model/data, see:
- FedLFPClient: provide a DataLoader that yields (x, y)
- feature_extractor f(θ_i;·) and classifier h(Φ_i;·)

Copyright / License
-------------------
This code is provided for research/educational use. You are responsible for
ensuring compliance with the paper's license and your project's needs.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Model split: f(θ) and h(Φ)
# ------------------------------

class SimpleMLPFeatureExtractor(nn.Module):
    """A tiny feature extractor f(θ;·) for demos."""
    def __init__(self, in_dim: int, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleClassifier(nn.Module):
    """A tiny classifier head h(Φ;·) for demos."""
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


@dataclass
class FedLFPConfig:
    # Paper hyperparameters / knobs
    T: int = 50            # total communication rounds (Algorithm 1, line 3)
    E: int = 2             # local epochs (Algorithm 1, line 20)
    gamma: float = 0.01    # local learning rate γ_i (Algorithm 1 input; line 27)
    K: int = 10            # target number of clusters on server (Algorithm 1 input; Eq. (5))
    lambda_cl: float = 0.1 # λ in Eq. (6)
    tau: float = 0.2       # temperature τ in Eq. (7)
    rho: float = 0.1       # ρ used for exp dissimilarity option (Section III-C-3)
    dissim_mode: str = "euclidean"  # {"euclidean", "exp"} (Eq. (9)-(10) discussion)
    device: str = "cpu"
    # Server KMeans options
    kmeans_iters: int = 25
    # Client optimizer
    momentum: float = 0.9


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
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            z = f_theta(x)  # representations (Section III-C-1)
            for cls in y.unique().tolist():
                mask = (y == cls)
                zc = z[mask].mean(dim=0)
                if cls not in sums:
                    sums[cls] = zc.detach().clone()
                    counts[cls] = int(mask.sum().item())
                else:
                    # weighted sum: keep sum, not mean, for stable aggregation
                    sums[cls] += z[mask].sum(dim=0)
                    counts[cls] += int(mask.sum().item())

    classes = sorted(sums.keys())
    if len(classes) == 0:
        # No data
        return (torch.empty(0, 1, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device))

    # Eq. (2) prototypes (mean then normalize)
    LP_list = []
    Q_list = []
    for cls in classes:
        mean_vec = sums[cls] / max(counts[cls], 1)
        LP_list.append(normalize(mean_vec))
        Q_list.append(counts[cls])

    LPi = torch.stack(LP_list, dim=0)                  # (Ci, d)
    Qi = torch.tensor(Q_list, device=device).float()   # (Ci,)

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


# ------------------------------
# Orchestrator
# ------------------------------

class FedLFPTrainer:
    """
    End-to-end training driver matching Algorithm 1 structure.
    """
    def __init__(self, clients: List[FedLFPClient], server: FedLFPServer, cfg: FedLFPConfig):
        self.clients = clients
        self.server = server
        self.cfg = cfg

    def sample_clients(self, ratio_low: float = 0.6, ratio_high: float = 1.0) -> List[FedLFPClient]:
        """
        Paper's experimental setup samples a client selection ratio between 0.6 and 1 each round
        (Section V-B-2), but Algorithm 1 just says "Sample subset At".
        """
        M = len(self.clients)
        ratio = random.uniform(ratio_low, ratio_high)
        m = max(1, int(round(M * ratio)))
        return random.sample(self.clients, m)

    def fit(self) -> None:
        """
        Run T communication rounds (Algorithm 1, line 3).
        """
        for t in range(1, self.cfg.T + 1):
            At = self.sample_clients()  # Algorithm 1, line 4

            # --- Clients upload {LPi, Qi, Si} from previous state if exists ---
            # In the paper, upload happens after local training each round (Algorithm 1, line 33).
            # To keep logic simple, we do: clients first train with last GP, then upload for next GP.
            # Start with GP=None in round 1.
            GP = self.server.GP  # may be None at t=1

            # --- Client local updates (Algorithm 1, line 13-16) ---
            for c in At:
                c.client_update(GP=GP)

            # --- Server receives and aggregates (Algorithm 1, line 5-6) ---
            payloads = [(c.LPi, c.Qi, c.Si) for c in At]
            self.server.aggregate(payloads)

            # --- Server computes GP for next round (Algorithm 1, line 7-12) ---
            self.server.compute_GP()

            if t % max(1, self.cfg.T // 10) == 0 or t == 1:
                print(f"[Round {t:03d}] | clients={len(At)} | |LP|={(self.server.LP.shape[0] if self.server.LP is not None else 0)} | GP={'set' if self.server.GP is not None else 'None'}")



# ------------------------------
# Paper model: LeNet-5 split (feature extractor + client head)
# ------------------------------

class LeNet5FeatureExtractor(nn.Module):
    """
    LeNet-5 feature extractor f(θ_i;·) used in the paper (Section V-B-2 "Model and Training").
    Paper detail: "Each client utilizes the LeNet-5 architecture, where the first four layers
    are shared with the server for feature extraction, and the final fully connected (FC) layer
    acts as the client-specific decision layer." fileciteturn1file3

    We implement the standard LeNet-5 up to the 84-dim feature vector:
      conv1 -> pool -> conv2 -> pool -> fc1(120) -> fc2(84)
    The client-specific classifier head is a separate Linear(84, num_classes).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)   # layer 1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)      # layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # layer 3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)      # layer 4

        # For 32x32 CIFAR: output after pool2 is (16, 5, 5) => 400 features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)  # paper: FC layer input dimension = 84 for CIFAR-10 fileciteturn1file3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x  # 84-dim representation


class LeNet5ClientHead(nn.Module):
    """Client-specific decision layer h(Φ_i;·): Linear(84, C)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(84, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ------------------------------
# CIFAR-10 utilities (Dirichlet Non-IID partition as in experiments)
# ------------------------------

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



def build_cifar10_fedlfp(
    data_dir: str,
    num_clients: int = 20,
    alpha: float = 0.1,  # paper uses α=0.1 for CIFAR-10 setting (Section V-C-2) fileciteturn1file4
    batch_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[List[FedLFPClient], FedLFPServer, FedLFPConfig, torch.utils.data.DataLoader]:
    """
    Build a FedLFP setup on CIFAR-10 following the paper's defaults:
      - LeNet-5 split model
      - batch_size=64, local epochs=2, lr γ=0.01, momentum=0.9, rounds T=50 fileciteturn1file0turn1file3
      - K = #classes = 10 for CIFAR-10 fileciteturn1file3
      - Dirichlet Non-IID with α (default 0.1 for CIFAR-10) fileciteturn1file4

    Returns:
      clients, server, cfg, test_loader
    """
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError(
            "torchvision is required for CIFAR-10. Please install torchvision."
        ) from e

    set_seed(seed)

    # Standard CIFAR-10 normalization (common; paper does not specify exact mean/std)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Dirichlet partition on train targets
    client_splits = dirichlet_partition(train_set.targets, num_clients=num_clients, alpha=alpha, seed=seed)

    cfg = FedLFPConfig(
        T=50,            # paper default fileciteturn1file0turn1file3
        E=2,             # paper default fileciteturn1file0turn1file3
        gamma=0.01,      # paper default fileciteturn1file0turn1file3
        K=10,            # CIFAR-10 classes fileciteturn1file3
        lambda_cl=0.1,   # paper indicates best λ=0.1 in ablation fileciteturn1file14
        tau=0.2,         # paper default temperature (Algorithm input; see method section)
        rho=0.1,         # used in FedLFP_exp variant fileciteturn1file1
        dissim_mode="euclidean",
        device=device,
        momentum=0.9,    # paper default fileciteturn1file0turn1file3
    )

    clients: List[FedLFPClient] = []
    for i in range(num_clients):
        subset = torch.utils.data.Subset(train_set, client_splits[i])
        dl = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

        f = LeNet5FeatureExtractor(in_channels=3)
        h = LeNet5ClientHead(num_classes=10)

        clients.append(FedLFPClient(i, f, h, dl, num_classes=10, cfg=cfg))

    server = FedLFPServer(cfg)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    return clients, server, cfg, test_loader


@torch.no_grad()
def evaluate_accuracy(
    client: FedLFPClient,
    dataloader: torch.utils.data.DataLoader,
) -> float:
    """Compute Top-1 accuracy on a dataloader."""
    client.f.eval()
    client.h.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(client.device)
        y = y.to(client.device)
        z = client.f(x)
        logits = client.h(z)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


# ------------------------------
# Demo
# ------------------------------

class ToyDataset(torch.utils.data.Dataset):
    """Simple 2D gaussian blobs for a quick functional demo."""
    def __init__(self, n: int, num_classes: int, seed: int, class_bias: Optional[List[int]] = None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_classes = num_classes

        # Make class means on a circle
        means = []
        for k in range(num_classes):
            ang = 2 * math.pi * k / num_classes
            means.append(torch.tensor([math.cos(ang), math.sin(ang)]) * 4.0)
        means = torch.stack(means, dim=0)

        # Sample labels with bias (simulate Non-IID)
        if class_bias is None:
            probs = torch.ones(num_classes) / num_classes
        else:
            probs = torch.zeros(num_classes)
            probs[class_bias] = 1.0
            probs = probs / probs.sum()

        y = torch.multinomial(probs, n, replacement=True, generator=g)
        x = means[y] + torch.randn((n, 2), generator=g) * 0.8

        self.x = x.float()
        self.y = y.long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_demo(num_clients: int = 10, num_classes: int = 5, device: str = "cpu") -> Tuple[List[FedLFPClient], FedLFPServer, FedLFPConfig]:
    cfg = FedLFPConfig(
        T=20,
        E=2,
        gamma=0.03,
        K=num_classes,          # K at least number of classes (paper: K set to known classes)
        lambda_cl=0.1,
        tau=0.2,
        dissim_mode="euclidean",
        device=device,
    )

    clients: List[FedLFPClient] = []
    for i in range(num_clients):
        # Non-IID: each client sees only a subset of classes
        bias = [i % num_classes, (i + 1) % num_classes]
        ds = ToyDataset(n=256, num_classes=num_classes, seed=1000 + i, class_bias=bias)
        dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        f = SimpleMLPFeatureExtractor(in_dim=2, feat_dim=32)
        h = SimpleClassifier(feat_dim=32, num_classes=num_classes)

        clients.append(FedLFPClient(i, f, h, dl, num_classes, cfg))

    server = FedLFPServer(cfg)
    return clients, server, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run a small synthetic demo (2D blobs).")
    parser.add_argument("--cifar10", action="store_true", help="Run FedLFP on CIFAR-10 using the paper's LeNet-5 split model.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Where to download/store datasets (for --cifar10).")
    parser.add_argument("--num_clients", type=int, default=20, help="Number of clients M (paper commonly uses 20 for image tasks).")
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration parameter α for Non-IID partition (CIFAR-10 default 0.1 in paper).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N rounds (for --cifar10).")
    args = parser.parse_args()

    if args.demo:
        set_seed(0)
        clients, server, cfg = build_demo(device=args.device)
        trainer = FedLFPTrainer(clients, server, cfg)
        trainer.fit()
        print("Demo finished.")
        return

    if args.cifar10:
        clients, server, cfg, test_loader = build_cifar10_fedlfp(
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            alpha=args.alpha,
            device=args.device,
        )
        trainer = FedLFPTrainer(clients, server, cfg)

        # Track a representative client's test accuracy (client 0) for convenience.
        for t in range(1, cfg.T + 1):
            At = trainer.sample_clients()
            GP = server.GP
            for c in At:
                c.client_update(GP=GP)
            payloads = [(c.LPi, c.Qi, c.Si) for c in At]
            server.aggregate(payloads)
            server.compute_GP()

            if (t % args.eval_every == 0) or (t == 1) or (t == cfg.T):
                acc = evaluate_accuracy(clients[0], test_loader)
                print(f"[Round {t:03d}] clients={len(At)} GP={'set' if server.GP is not None else 'None'} | client0 test acc={acc*100:.2f}%")
        return

    print("This is a library-style reference implementation. Use --demo or --cifar10.")


if __name__ == "__main__":
    main()
