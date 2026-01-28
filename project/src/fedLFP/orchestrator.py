# ------------------------------
# Orchestrator
# ------------------------------

import csv
from dataclasses import dataclass
from itertools import zip_longest
import random
import sys
from typing import List, Tuple

from fedLFP.client import FedLFPClient
from fedLFP.configs import FedLFPConfig
from fedLFP.server import FedLFPServer
import torch
from tqdm import tqdm

@dataclass
class Statistics:
    upload_to_server: List[int]
    download_from_server: List[int]
    normed_relative_q_error: List[float]
    mean_accuracies: List[float]

    def to_csv(self, filepath: str = "stats.csv"):
        headers = [
            "upload_to_server",
            "download_from_server",
            "normed_relative_q_error",
            "mean_accuracies",
        ]


        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(zip_longest(
            self.upload_to_server,
            self.download_from_server,
            self.normed_relative_q_error,
            self.mean_accuracies,
            fillvalue=""
        ))


class FedLFPTrainer:
    """
    End-to-end training driver matching Algorithm 1 structure.
    """
    def __init__(self, clients: List[FedLFPClient], server: FedLFPServer, cfg: FedLFPConfig):
        self.clients = clients
        self.server = server
        self.cfg = cfg

    def quantize_int8(x: torch.Tensor):
        x = x.detach()
        maxv = x.abs().max()
        scale = (maxv / 127.0) if maxv > 0 else x.new_tensor(1.0)
        q = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
        return q, scale

    def dequantize_int8(q: torch.Tensor, scale: torch.Tensor):
        return q.to(torch.float32) * scale

    def sample_clients(self, ratio_low: float = 0.6, ratio_high: float = 1.0) -> List[FedLFPClient]:
        """
        Paper's experimental setup samples a client selection ratio between 0.6 and 1 each round
        (Section V-B-2), but Algorithm 1 just says "Sample subset At".
        """
        M = len(self.clients)
        ratio = random.uniform(ratio_low, ratio_high)
        m = max(1, int(round(M * ratio)))
        return random.sample(self.clients, m)
    

    @torch.no_grad()
    def evaluate_correct_and_total(
        self,
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
    def evaluate_mean_client_accuracy(self, clients, test_loader):
        accs = []
        for c in clients:
            corr, n = self.evaluate_correct_and_total(c, test_loader)
            accs.append(corr / max(1, n))
        return float(sum(accs) / len(accs))

    def fit(
        self,
        test_loader: torch.utils.data.DataLoader,
        quantize: bool = False
    ) -> Statistics:
        """
        Runs one training experiment and returns:
        best_over_rounds_overall_accuracy (paper-style: best single-round accuracy in this run)
        """

        stats = Statistics([],[],[],[])

        for t in tqdm(range(1, self.cfg.T + 1), desc="round"):
            At = self.sample_clients()


            # client update with current global prototypes
            GP = self.server.GP
            if quantize and t > 1:
                GPq, GPs = self.quantize_int8(self.server.GP)
                GPd = self.dequantize_int8(q=GPq, scale=GPs)
                GP = GPd
                stats.download_from_server.append(sys.getsizeof((GPq, GPs)))
            elif t == 1:
                stats.download_from_server.append(sys.getsizeof(0))
            else:
                stats.download_from_server.append(sys.getsizeof(GP))

            payloads = []
            for c in At:      
                c.client_update(GP=GP)
                if quantize:
                    lpq, lps = self.quantize_int8(c.LPi)
                    lpd = self.dequantize_int8(q=lpq, scale=lps)
                    payloads.append((lpd, c.Qi, c.Si))
                else:
                    payloads.append((c.LPi, c.Qi, c.Si))

            
            self.server.aggregate(payloads)
            self.server.compute_GP()

            stats.upload_to_server.append({sys.getsizeof(payloads)})
            stats.mean_accuracies.append(self.evaluate_mean_client_accuracy(self.clients, test_loader))
            

        return stats



