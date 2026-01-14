import argparse
import numpy as np
import torch

from .configs import Config
from .data import set_seed, load_cifar10, dirichlet_partition, make_loaders
from .models import SplitCNN
from .splitfed import SplitFedSimulator
from .proto import PrototypeBank

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="adaptive", choices=["fixed", "adaptive"])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--no-proto", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.no_proto:
        cfg.use_prototypes = False
        cfg.proto_cond_head = False

    set_seed(cfg.seed)

    train_ds, test_ds = load_cifar10()
    client_indices = dirichlet_partition(train_ds.targets, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)
    client_loaders, test_loader = make_loaders(train_ds, test_ds, client_indices, cfg.batch_size)

    full = SplitCNN(num_classes=10, emb_dim=128, proto_cond_head=cfg.proto_cond_head)

    proto = None
    if cfg.use_prototypes:
        proto = PrototypeBank(num_classes=10, emb_dim=128, momentum=cfg.proto_momentum, device=cfg.device)

    sim = SplitFedSimulator(full, cfg, proto_bank=proto)

    for r in range(cfg.rounds):
        rng = np.random.default_rng(cfg.seed + r)
        selected = rng.choice(cfg.num_clients, size=cfg.clients_per_round, replace=False).tolist()

        logs = sim.train_round(client_loaders, selected, mode=args.mode)
        ev = sim.evaluate(test_loader, cut_for_eval=cfg.cut_candidates[-1])

        # print a compact round summary
        cuts = [l["cut"] for l in logs]
        avg_act = sum(l["act_mb"] for l in logs) / len(logs)
        avg_acc = sum(l["acc"] for l in logs) / len(logs)

        print(f"[Round {r+1:03d}] mode={args.mode} "
              f"train_acc={avg_acc:.3f} avg_actMB={avg_act:.2f} "
              f"cuts={cuts} test_acc={ev['test_acc']:.3f}")

if __name__ == "__main__":
    main()
