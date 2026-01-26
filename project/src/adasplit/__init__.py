"""
Top-level package for WFM (Workflow Manager).
"""

from .client import FedLFPClient
from .configs import FedLFPConfig
from .data import dirichlet_partition
from .models import LeNet5ClientHead, LeNet5FeatureExtractor
from .operations import normalize, set_seed, sim
from .orchestrator import FedLFPTrainer
from .server import FedLFPServer

__all__ = [
    "FedLFPClient",
    "FedLFPConfig",
    "dirichlet_partition",
    "LeNet5ClientHead",
    "LeNet5FeatureExtractor",
    "normalize",
    "set_seed",
    "sim",
    "FedLFPTrainer",
    "FedLFPServer"    
]