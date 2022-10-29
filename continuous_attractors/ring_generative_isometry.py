import numpy as np
import torch
from torch import nn


in_dim = 1
out_dim = 3

model = nn.Sequential(
    torch.nn.Linear(1, 50),
    torch.nn.Sigmoid(),
    torch.nn.Linear(50, 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(100, 20),
    torch.nn.Sigmoid(),
    torch.nn.Linear(20, 3))

n_train = 10000
in_samples = np.random.uniform(-np.pi, np.pi, n_train)
angle_batch = np.random.uniform(-np.pi, np.pi, (5, 10))
