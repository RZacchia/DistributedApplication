import argparse
import datetime
from typing import List, Tuple

from fedLFP.client import FedLFPClient
from fedLFP.configs import FedLFPConfig
from fedLFP.data import dirichlet_partition, load_kronodroid_npz, majority_baseline_from_loader
from fedLFP.models import LeNet5ClientHead, LeNet5FeatureExtractor, init_head_small, init_lenet_tanh
from fedLFP.operations import set_seed
from fedLFP.orchestrator import FedLFPTrainer
from fedLFP.server import FedLFPServer
import torch
import numpy as np
from tqdm import tqdm




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
        T=50,
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
        h = LeNet5ClientHead(num_classes=cfg.K)

        clients.append(FedLFPClient(i, f, h, train_loader, num_classes=10, cfg=cfg))

    server = FedLFPServer(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
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




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3, help="paper-style: repeat and average best-over-rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-quantize", action="store_false", dest="quantize")
    ap.set_defaults(quantize=True)


    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--krono", action="store_true")
    g.add_argument("--cifar10", action="store_true")

    args = ap.parse_args()



    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available; using CPU")


    for r in tqdm(range(args.runs), desc="run"):
        seed = args.seed + r
        prefix = ""
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
            prefix = "krono_"
        else:
            clients, server, cfg, test_loader = build_cifar10_fedlfp(
                data_dir=args.data_dir,
                num_clients=args.num_clients,
                alpha=args.alpha,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
            )
            prefix = "cifar_10_"


        print("Train X shape:", np.load("./data/kronodroid_npz/kronodroid_train.npz")["X"].shape)
        trainer = FedLFPTrainer(
            clients=clients,
            server=server,
            cfg=cfg
            )
        stats = trainer.fit(test_loader=test_loader, quantize=args.quantize)
        filename = datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")
        
        if args.quantize:
            print("Quantized enabled")
            prefix = prefix + "q_"
        stats.to_csv(filepath=f"{prefix}{filename}.csv")
    


if __name__ == "__main__":
    main()
