import torch

class PrototypeBank:
    """
    Maintains per-client and global class prototypes in embedding space.
    """
    def __init__(self, num_classes: int, emb_dim: int, momentum: float, device: str):
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.momentum = momentum
        self.device = device

        self.global_proto = torch.zeros(num_classes, emb_dim, device=device)
        self.global_count = torch.zeros(num_classes, device=device)

        # per-client prototypes stored in python dict: client_id -> (proto, count)
        self.client = {}

    def _get_client(self, cid: int):
        if cid not in self.client:
            self.client[cid] = (
                torch.zeros(self.num_classes, self.emb_dim, device=self.device),
                torch.zeros(self.num_classes, device=self.device)
            )
        return self.client[cid]

    @torch.no_grad()
    def update_client(self, cid: int, emb: torch.Tensor, y: torch.Tensor):
        proto, cnt = self._get_client(cid)
        for k in range(self.num_classes):
            m = (y == k)
            if m.any():
                mean_k = emb[m].mean(dim=0)
                proto[k] = self.momentum * proto[k] + (1 - self.momentum) * mean_k
                cnt[k] += m.sum().float()

    @torch.no_grad()
    def aggregate_to_global(self):
        # weighted average of client prototypes
        gp = torch.zeros_like(self.global_proto)
        gc = torch.zeros_like(self.global_count)
        for (proto, cnt) in self.client.values():
            gp += proto * cnt.unsqueeze(-1)
            gc += cnt
        denom = gc.clamp_min(1.0).unsqueeze(-1)
        self.global_proto = gp / denom
        self.global_count = gc

    def proto_context(self, cid: int, y: torch.Tensor):
        """
        Create a context vector for conditioning.
        Here: use the client's prototype for the ground-truth class if available,
        otherwise fall back to global prototype.
        """
        proto, cnt = self._get_client(cid)
        ctx = []
        for yi in y.tolist():
            if cnt[yi] > 0:
                ctx.append(proto[yi])
            else:
                ctx.append(self.global_proto[yi])
        return torch.stack(ctx, dim=0)
