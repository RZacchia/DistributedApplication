import argparse
from typing import List, Tuple

from adasplit.client import FedLFPClient
from adasplit.configs import FedLFPConfig
from adasplit.data import dirichlet_partition
from adasplit.models import LeNet5ClientHead, LeNet5FeatureExtractor
from adasplit.operations import set_seed
from adasplit.orchestrator import FedLFPTrainer
from adasplit.server import FedLFPServer
import torch


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
                print(f"client update")
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
