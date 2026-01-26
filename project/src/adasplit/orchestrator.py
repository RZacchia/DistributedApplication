# ------------------------------
# Orchestrator
# ------------------------------

import random
from typing import List

from adasplit.client import FedLFPClient
from adasplit.configs import FedLFPConfig
from adasplit.server import FedLFPServer


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



