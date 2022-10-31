import numpy as np
import torch
from torch import nn

from continuous_attractors import generative_isometry_util


in_dim = 1
out_dim = 3
hidden_dim=100
n_hidden=1


layers = []
layers.append(nn.Linear(in_dim, int(hidden_dim/2)))
layers.append(nn.Sigmoid())
for _ in range(n_hidden):
    layers.append(nn.LazyLinear(hidden_dim))
    layers.append(nn.Sigmoid())
layers.append(nn.Linear(hidden_dim, int(hidden_dim / 2)))
layers.append(nn.Sigmoid())
layers.append(nn.Linear(int(hidden_dim / 2), out_dim))
net = nn.Sequential(*layers)
opt = torch.optim.Adam(net.parameters())
params = list(net.parameters())

batch_size = 20
n_points_compare = 30
n_epochs = 500
losses = []
for _ in range(n_epochs):
    angle_batch = torch.tensor(np.random.uniform(-np.pi, np.pi, (batch_size, n_points_compare)), dtype=torch.get_default_dtype())
    angle_metric = generative_isometry_util.angle_metric_1d(angle_batch)
    scaled_angle_metric = angle_metric/torch.mean(angle_metric)
    opt.zero_grad()
    forward_points = net.forward(torch.unsqueeze(torch.flatten(angle_batch), -1))
    forward_points_reshaped = torch.reshape(forward_points, [*angle_batch.size(), -1])
    forward_distances = generative_isometry_util.point_metric(forward_points_reshaped)
    scaled_forward_distances = forward_distances/torch.mean(forward_distances)
    loss = torch.sum(torch.square(scaled_forward_distances - scaled_angle_metric))
    losses.append(loss.detach().numpy())
    print(loss)
    loss.backward()
    opt.step()

import matplotlib.pyplot as plt
loss_fig, loss_axs = plt.subplots()
angles = np.arange(start=-np.pi, stop=np.pi, step=0.01)
with torch.no_grad():
    test_angles = torch.unsqueeze(torch.tensor(angles, dtype=torch.get_default_dtype()), -1)
    forward_pred = net.forward(test_angles)
forward_pred = forward_pred.detach().numpy()
loss_axs.plot(losses)

proj_fig = plt.figure()
proj_axs = proj_fig.add_subplot(projection="3d")
proj_axs.scatter(forward_pred[:, 0], forward_pred[:, 1], forward_pred[:, 2],  cmap="hsv", c=angles)
plt.show()
print("")

