import argparse
from typing import List, Tuple

from adasplit.client import FedLFPClient
from adasplit.configs import FedLFPConfig
from adasplit.data import dirichlet_partition, load_kronodroid_npz
from adasplit.models import LeNet5ClientHead, LeNet5FeatureExtractor, init_head_small, init_lenet_tanh
from adasplit.operations import set_seed
from adasplit.orchestrator import FedLFPTrainer
from adasplit.server import FedLFPServer
import sys
import torch
import numpy as np

@torch.no_grad()
def majority_baseline_from_loader(test_loader) -> tuple[int, float, np.ndarray]:
    ys = []
    for _, y in test_loader:
        ys.append(y.cpu().numpy())
    y_all = np.concatenate(ys, axis=0)

    counts = np.bincount(y_all)
    majority_cls = int(counts.argmax())
    baseline_acc = float(counts[majority_cls] / counts.sum())
    return majority_cls, baseline_acc, counts

@torch.no_grad()
def evaluate_mean_client_accuracy(clients, test_loader):
    accs = []
    for c in clients:
        corr, n = evaluate_correct_and_total(c, test_loader)
        accs.append(corr / max(1, n))
    return float(sum(accs) / len(accs)), accs


def _server_aggregate_compat(server: FedLFPServer, clients: List[FedLFPClient]) -> None:
    """
    Compatibility wrapper because some codebases implement:
      - server.aggregate(payloads)
    others implement:
      - server.aggregate(clients)
    """
    payloads = [(c.LPi, c.Qi, c.Si) for c in clients]
    try:
        server.aggregate(payloads)  # type: ignore[arg-type]
    except TypeError:
        server.aggregate(clients)   # type: ignore[arg-type]

@torch.no_grad()
def evaluate_correct_and_total(
    client: FedLFPClient,
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[int, int]:
    """Return (correct, total) Top-1 over a dataloader."""
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
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct, total


@torch.no_grad()
def evaluate_overall_accuracy_paper_style(
    clients: List[FedLFPClient],
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float, int]:
    """
    Paper-style overall accuracy:
        total_correct / total_predictions
    where each client model is evaluated on the SAME global test set.
    """
    total_correct = 0
    total_n = 0
    for c in clients:
        corr, n = evaluate_correct_and_total(c, test_loader)
        total_correct += corr
        total_n += n
    return total_correct / max(1, total_n), total_n


def build_krono_droid_fedlfp(
    data_dir: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    seed: int,
    device: str,
    image_side: int = 18,
) -> Tuple[List[FedLFPClient], FedLFPServer, FedLFPConfig, torch.utils.data.DataLoader]:
    """
    KronoDroid FedLFP runner (paper-style evaluation):
      - train is Dirichlet-partitioned (non-IID)
      - test is ONE global time-based test set (no Dirichlet split for test)
      - overall accuracy is computed on that global test set
    """
    set_seed(seed)

    train_set, test_set = load_kronodroid_npz(
        data_dir=data_dir,
        in_channels=3,
        image_side=image_side,
    )

    train_splits = dirichlet_partition(
        targets=train_set.y.tolist(),
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    cfg = FedLFPConfig(
        T=20,
        E=2,
        gamma=0.01,
        momentum=0.9,
        K=10,              # benign + 12 families
        lambda_cl=0.1,
        tau=0.2,
        rho=0.1,
        dissim_mode="euclidean",
        device=device,
    )

    clients: List[FedLFPClient] = []
    for i in range(num_clients):
        subset = torch.utils.data.Subset(train_set, train_splits[i])
        train_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        f = LeNet5FeatureExtractor(in_channels=3, input_size=image_side)
        h = LeNet5ClientHead(num_classes=cfg.K)

        f.apply(init_lenet_tanh)
        h.apply(init_head_small)
        clients.append(FedLFPClient(i, f, h, train_loader, num_classes=cfg.K, cfg=cfg))

    server = FedLFPServer(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    x0, y0 = next(iter(clients[0].dataloader))
    print("Train batch tensor shape:", x0.shape, "labels:", y0.min().item(), y0.max().item())


    maj_cls, maj_acc, counts = majority_baseline_from_loader(test_loader)
    print(f"[Baseline] Majority class = {maj_cls}, count = {counts[maj_cls]}, "
        f"baseline acc = {maj_acc*100:.2f}% (N={counts.sum()})")
    print("[Baseline] Test class counts:", counts.tolist())


    counts = np.bincount(np.array(train_set.y))
    present = counts > 0
    weights = np.zeros_like(counts, dtype=np.float32)
    weights[present] = counts[present].sum() / counts[present].astype(np.float32)
    weights[present] /= weights[present].mean()  # normalize only over present classes

    cfg.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Train class counts:", counts.tolist())
    print("Class weights:", weights.tolist())
    

    return clients, server, cfg, test_loader


def build_cifar10_fedlfp(
    data_dir: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    seed: int,
    device: str,
) -> Tuple[List[FedLFPClient], FedLFPServer, FedLFPConfig, torch.utils.data.DataLoader]:
    """
    CIFAR-10 setup (kept simple):
      - train Dirichlet partition
      - test is global CIFAR-10 test set
      - overall accuracy computed paper-style on global test set
    """
    set_seed(seed)
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for CIFAR-10") from e

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_splits = dirichlet_partition(
        targets=train_set.targets,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    cfg = FedLFPConfig(
        T=100,
        E=2,
        gamma=0.01,
        momentum=0.9,
        K=10,
        lambda_cl=0.1,
        tau=0.2,
        rho=0.1,
        dissim_mode="euclidean",
        device=device,
    )

    clients: List[FedLFPClient] = []
    for i in range(num_clients):
        subset = torch.utils.data.Subset(train_set, train_splits[i])
        train_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        f = LeNet5FeatureExtractor(in_channels=3)
        h = LeNet5ClientHead(num_classes=10)

        clients.append(FedLFPClient(i, f, h, train_loader, num_classes=10, cfg=cfg))

    server = FedLFPServer(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return clients, server, cfg, test_loader


def quantize_int8(x: torch.Tensor):
    x = x.detach()
    maxv = x.abs().max()
    scale = (maxv / 127.0) if maxv > 0 else x.new_tensor(1.0)
    q = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
    return q, scale

def dequantize_int8(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale


def fit(
    clients: List[FedLFPClient],
    server: FedLFPServer,
    cfg: FedLFPConfig,
    test_loader: torch.utils.data.DataLoader,
    eval_every: int,
    quantize: bool = False
) -> float:
    """
    Runs one training experiment and returns:
      best_over_rounds_overall_accuracy (paper-style: best single-round accuracy in this run)
    """
    trainer = FedLFPTrainer(clients, server, cfg)
    best_acc = 0.0

    for t in range(1, cfg.T + 1):
        At = trainer.sample_clients()


        # client update with current global prototypes
        GP = server.GP
        if quantize and t > 1:
            GPq, GPs = quantize_int8(server.GP)
            GPd = dequantize_int8(q=GPq, scale=GPs)
            rel_err = (GP - GPd).norm() / (GP.norm() + 1e-12)
            print("GP rel quant error:", rel_err.item())
            GP = GPd
            print(f"Server Broadcasts GP ({sys.getsizeof((GPq, GPs))} Bytes) to {len(clients)} clients.")

        else:
            print(f"Server Broadcasts GP ({sys.getsizeof(GP)} Bytes) to {len(clients)} clients.")

        payloads = []
        payloads_q = []  
        for c in At:      
            c.client_update(GP=GP)
            if quantize:
                lpq, lps = quantize_int8(c.LPi)
                lpd = dequantize_int8(q=lpq, scale=lps)
                rel_err = (c.LPi - lpd).norm() / (c.LPi.norm() + 1e-12)
                payloads.append((lpd, c.Qi, c.Si))
                payloads_q.append((lpd, c.Qi, c.Si))
            else:
                payloads.append((c.LPi, c.Qi, c.Si))
                

        if quantize:
            print(f"{len(clients)} clients send back payloads ({sys.getsizeof(payloads_q)} Bytes) to server.")
        else:
            print(f"{len(clients)} clients send back payloads ({sys.getsizeof(payloads)} Bytes) to server.")
            

        server.aggregate(payloads)
        server.compute_GP()

        if (t % eval_every == 0) or (t == 1) or (t == cfg.T):
            mean_acc, accs = evaluate_mean_client_accuracy(clients, test_loader)
            print(f"[Round {t:03d}] Mean client test acc = {mean_acc*100:.2f}% (global test)")
            best_acc = max(accs)
            print(f"[Round {t:03d}] Best client test acc = {best_acc*100:.2f}% )")

    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3, help="paper-style: repeat and average best-over-rounds")
    ap.add_argument("--seed", type=int, default=0)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--krono", action="store_true")
    g.add_argument("--cifar10", action="store_true")

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available; using CPU")

    best_runs: List[float] = []

    for r in range(args.runs):
        seed = args.seed + r
        print(f"\n=== Run {r+1}/{args.runs} (seed={seed}) ===")

        if args.krono:
            clients, server, cfg, test_loader = build_krono_droid_fedlfp(
                data_dir=args.data_dir,
                num_clients=args.num_clients,
                alpha=args.alpha,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
                image_side=18,
            )
        else:
            clients, server, cfg, test_loader = build_cifar10_fedlfp(
                data_dir=args.data_dir,
                num_clients=args.num_clients,
                alpha=args.alpha,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
            )

        print("Train X shape:", np.load("./data/kronodroid_npz/kronodroid_train.npz")["X"].shape)

        best_acc = fit(
            clients=clients,
            server=server,
            cfg=cfg,
            test_loader=test_loader,
            eval_every=args.eval_every,
        )
        best_runs.append(best_acc)
        print(f"Best (this run): {best_acc*100:.2f}%")

    final_report = float(sum(best_runs) / max(1, len(best_runs)))
    print(f"\nFinal (paper-style): avg(best over rounds) over {args.runs} runs = {final_report*100:.2f}%")


if __name__ == "__main__":
    main()
