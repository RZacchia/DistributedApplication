import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitCNN(nn.Module):
    """
    A small CNN broken into "blocks". Cut points refer to block index.
    blocks[0..cut-1] live on client, blocks[cut..] on server.
    """
    def __init__(self, num_classes=10, emb_dim=128, proto_cond_head=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(  # block 0
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(  # block 1
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(  # block 2
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            ),
        ])

        self.to_emb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True),
        )

        self.proto_cond_head = proto_cond_head
        if proto_cond_head:
            # FiLM-style conditioning using a "proto context" vector
            self.film = nn.Linear(emb_dim, emb_dim * 2)  # gamma,beta
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward_from_cut(self, z, proto_ctx=None):
        """
        Server-side forward starting at representation z produced by client blocks.
        z shape depends on cut.
        """
        x = z
        # If z is an activation map, run remaining conv blocks + head
        if x.dim() == 4:
            # determine which conv block we are at by channel count
            # We'll just run "rest of blocks" outside; splitfed passes correct remainder.
            raise RuntimeError("Use ServerNet wrapper for remaining blocks.")

        # If z already embedding, just head
        emb = x
        if self.proto_cond_head and proto_ctx is not None:
            gb = self.film(proto_ctx)
            gamma, beta = gb.chunk(2, dim=-1)
            emb = emb * (1 + torch.tanh(gamma)) + beta
        logits = self.classifier(emb)
        return logits, emb

class ClientNet(nn.Module):
    def __init__(self, full: SplitCNN, cut: int):
        super().__init__()
        self.cut = cut
        self.blocks = nn.Sequential(*list(full.blocks[:cut]))

    def forward(self, x):
        return self.blocks(x)

class ServerNet(nn.Module):
    def __init__(self, full: SplitCNN, cut: int):
        super().__init__()
        self.cut = cut
        self.blocks = nn.Sequential(*list(full.blocks[cut:]))
        self.to_emb = full.to_emb
        self.proto_cond_head = full.proto_cond_head
        self.film = full.film if full.proto_cond_head else None
        self.classifier = full.classifier

    def forward(self, z, proto_ctx=None):
        x = z
        if x.dim() == 4:
            x = self.blocks(x)
            emb = self.to_emb(x)
        else:
            emb = x

        if self.proto_cond_head and proto_ctx is not None:
            gb = self.film(proto_ctx)
            gamma, beta = gb.chunk(2, dim=-1)
            emb = emb * (1 + torch.tanh(gamma)) + beta

        logits = self.classifier(emb)
        return logits, emb
