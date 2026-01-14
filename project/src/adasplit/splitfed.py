import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from metrics import AvgMeter, accuracy, Timer
from models import ClientNet, ServerNet

def estimate_activation_mb(cut: int, batch_size: int):
    """
    Rough activation sizes for our SplitCNN after each block:
      cut=1 => [B,64,16,16]
      cut=2 => [B,128,8,8]
      cut=3 => [B,256,1,1] (after adaptive avg pool)
    """
    if cut == 1:
        shape = (batch_size, 64, 16, 16)
    elif cut == 2:
        shape = (batch_size, 128, 8, 8)
    elif cut == 3:
        shape = (batch_size, 256, 1, 1)
    else:
        raise ValueError("Unsupported cut")
    numel = np.prod(shape)
    bytes_ = numel * 4  # fp32
    return bytes_ / (1024**2)

def adaptive_cut_choice(cut_candidates, batch_size, uplink_mb_s, compute_speed):
    """
    Simple policy:
      time ≈ (activation_MB / uplink_MBps) + (client_compute_cost / compute_speed)

    We model client compute cost as increasing with later cuts (more layers on client).
    You can replace this with your own objective/constraints.
    """
    # relative client compute per cut (later cut => more client compute)
    compute_cost = {1: 1.0, 2: 1.8, 3: 2.6}
    best = None
    best_t = 1e9
    for cut in cut_candidates:
        comm_t = estimate_activation_mb(cut, batch_size) / max(1e-6, uplink_mb_s)
        comp_t = compute_cost[cut] / max(1e-6, compute_speed)
        t = comm_t + comp_t
        if t < best_t:
            best_t = t
            best = cut
    return best, best_t

class SplitFedSimulator:
    def __init__(self, full_model, cfg, proto_bank=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.full = full_model.to(self.device)
        self.proto = proto_bank

        # server holds a "reference" model; per cut we build ServerNet views
        self.server_state = copy.deepcopy(self.full.state_dict())

        # simulate heterogeneity
        rng = np.random.default_rng(cfg.seed)
        self.uplink = rng.uniform(2.0, 12.0, size=cfg.num_clients)  # MB/s
        self.compute = rng.uniform(0.7, 1.4, size=cfg.num_clients)  # speed scalar

    def _make_pair(self, cut: int):
        # create client/server nets sharing weights from self.full via state_dict load
        full = copy.deepcopy(self.full)
        full.load_state_dict(self.server_state)

        cnet = ClientNet(full, cut).to(self.device)
        snet = ServerNet(full, cut).to(self.device)
        return cnet, snet

    def _server_optimizer(self, snet: nn.Module):
        return torch.optim.SGD(
            snet.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay
        )
    
    def _compose_full_state(self, cnet, snet, cut: int):
        """
        Build a full SplitCNN-compatible state_dict by merging:
          - client blocks [0..cut-1] from cnet
          - server blocks [cut..end] from snet (with index remap)
          - head params (to_emb, classifier, film) from snet
        """
        full_state = copy.deepcopy(self.server_state)  # start from current global

        c_sd = cnet.state_dict()
        s_sd = snet.state_dict()

        # ---- client blocks: cnet.blocks.{i} -> full.blocks.{i}
        for k, v in c_sd.items():
            # k like: "blocks.0.0.weight" etc
            if not k.startswith("blocks."):
                continue
            full_k = "blocks." + k[len("blocks."):]  # same index
            full_state[full_k] = v.detach().cpu() if v.is_cuda else v.detach().clone()

        # ---- server conv blocks: snet.blocks.{i} corresponds to full.blocks.{cut+i}
        for k, v in s_sd.items():
            if k.startswith("blocks."):
                # k is like "blocks.0.0.weight" but should map to full.blocks.(cut+0)...
                rest = k[len("blocks."):]             # "0.0.weight"
                parts = rest.split(".", 1)            # ["0", "0.weight"]
                i = int(parts[0])
                tail = parts[1] if len(parts) > 1 else ""
                full_block_idx = cut + i
                full_k = f"blocks.{full_block_idx}.{tail}" if tail else f"blocks.{full_block_idx}"
                full_state[full_k] = v.detach().cpu() if v.is_cuda else v.detach().clone()

        # ---- heads (names match full model)
        for head_prefix in ["to_emb.", "classifier.", "film."]:
            for k, v in s_sd.items():
                if k.startswith(head_prefix):
                    full_state[k] = v.detach().cpu() if v.is_cuda else v.detach().clone()

        return full_state


    def train_round(self, client_loaders, client_ids, mode: str):
        """
        mode in {"fixed", "adaptive"}
        """
        cfg = self.cfg
        round_logs = []
        server_updates = []

        for cid in client_ids:
            loader = client_loaders[cid]

            # choose cut
            if mode == "fixed":
                cut = cfg.cut_candidates[-1]  # e.g., latest cut as baseline
                est_t = None
            elif mode == "adaptive":
                cut, est_t = adaptive_cut_choice(
                    cfg.cut_candidates, cfg.batch_size, self.uplink[cid], self.compute[cid]
                )
            else:
                raise ValueError(mode)

            cnet, snet = self._make_pair(cut)
            snet.train(); cnet.train()

            opt_s = self._server_optimizer(snet)
            opt_c = torch.optim.SGD(
                cnet.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay
            )

            loss_m = AvgMeter()
            acc_m = AvgMeter()

            for _ in range(cfg.local_epochs):
                for x, y in loader:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    # client forward
                    with Timer() as t_client_fwd:
                        z = cnet(x)

                    # (optional) "simulate" comm delay (no sleep; just log estimate)
                    act_mb = estimate_activation_mb(cut, x.size(0))

                    # server forward/backward
                    z = z.detach().requires_grad_(True)
                    proto_ctx = None
                    if self.proto and cfg.use_prototypes and cfg.proto_cond_head:
                        proto_ctx = self.proto.proto_context(cid, y)

                    logits, emb = snet(z, proto_ctx=proto_ctx)

                    ce = F.cross_entropy(logits, y)

                    # prototype pull loss (optional)
                    if self.proto and cfg.use_prototypes:
                        with torch.no_grad():
                            # if no prototypes yet, pull toward global (which is 0 initially)
                            target = self.proto.proto_context(cid, y)
                        pl = F.mse_loss(emb, target)
                        loss = ce + cfg.proto_loss_weight * pl
                    else:
                        loss = ce

                    opt_s.zero_grad()
                    opt_c.zero_grad()
                    loss.backward()

                    # server sends grad wrt activation back to client
                    grad_z = z.grad.detach()
                    z_client = cnet(x)
                    z_client.backward(grad_z)

                    opt_s.step()
                    opt_c.step()

                    loss_m.update(loss.item(), x.size(0))
                    acc_m.update(accuracy(logits.detach(), y), x.size(0))

                    # update client prototypes using embedding produced on server
                    if self.proto and cfg.use_prototypes:
                        self.proto.update_client(cid, emb.detach(), y)

            # collect "server-side" updated weights (SplitFed style: only aggregate server model)
            full_update = self._compose_full_state(cnet, snet, cut)
            server_updates.append(full_update)
            round_logs.append({
                "cid": cid,
                "cut": cut,
                "loss": loss_m.avg,
                "acc": acc_m.avg,
                "act_mb": act_mb,
                "uplink_mb_s": float(self.uplink[cid]),
                "compute": float(self.compute[cid]),
                "est_time": float(est_t) if est_t is not None else None
            })

        # aggregate server models (FedAvg over participating clients)
        new_state = copy.deepcopy(server_updates[0])
        for k in new_state.keys():
            new_state[k] = sum(u[k] for u in server_updates) / len(server_updates)
        self.server_state = new_state

        # aggregate prototypes to global
        if self.proto and cfg.use_prototypes:
            self.proto.aggregate_to_global()

        return round_logs

    @torch.no_grad()
    def evaluate(self, test_loader, cut_for_eval: int = 3):
        # evaluate using a fixed cut for simplicity (client->server)
        cnet, snet = self._make_pair(cut_for_eval)
        cnet.eval(); snet.eval()

        acc_m = AvgMeter()
        loss_m = AvgMeter()

        for x, y in test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            z = cnet(x)
            proto_ctx = None
            if self.proto and self.cfg.use_prototypes and self.cfg.proto_cond_head:
                # label-independent context: global mean prototype
                mean_proto = self.proto.global_proto.mean(dim=0, keepdim=True)  # [1, D]
                proto_ctx = mean_proto.repeat(y.size(0), 1)                     # [B, D]

            logits, _ = snet(z, proto_ctx=proto_ctx)

            loss = F.cross_entropy(logits, y)
            loss_m.update(loss.item(), x.size(0))
            acc_m.update(accuracy(logits, y), x.size(0))

        return {"test_loss": loss_m.avg, "test_acc": acc_m.avg}
