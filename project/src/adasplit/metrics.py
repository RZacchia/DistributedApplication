import time
import torch

class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.n += n

    @property
    def avg(self):
        return self.sum / max(1, self.n)

def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
