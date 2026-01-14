import pytest
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adasplit import Timer, AvgMeter, accuracy


def test_accuracy_calc():
    x = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6])
    y = 5

    assert accuracy(x,y) == 1
    pass