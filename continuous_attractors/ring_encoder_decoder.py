import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from continuous_attractors import generative_isometry_util


in_dim = 1
out_dim = 3
encoder_hidden_dim=100
encoder_n_hidden = 1
decoder_hidden_dim = 100
decoder_n_hidden = 1

encoder_layers = []
encoder_layers.append(nn.Linear(in_dim, int(encoder_hidden_dim/2)))
encoder_layers.append(nn.Sigmoid())
for _ in range(encoder_n_hidden):
    encoder_layers.append(nn.LazyLinear(encoder_hidden_dim))
    encoder_layers.append(nn.Sigmoid())
encoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
encoder_layers.append(nn.Sigmoid())
encoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), out_dim))
encoder_net = nn.Sequential(*encoder_layers)


decoder_layers = []
decoder_layers.append(nn.Linear(out_dim, int(encoder_hidden_dim/2)))
decoder_layers.append(nn.Sigmoid())
for _ in range(encoder_n_hidden):
    decoder_layers.append(nn.LazyLinear(encoder_hidden_dim))
    decoder_layers.append(nn.Sigmoid())
decoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
decoder_layers.append(nn.Sigmoid())
decoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), in_dim))
decoder_net = nn.Sequential(*decoder_layers)

opt = torch.optim.Adam(encoder_net.parameters())

batch_size = 100
n_points_compare = 100
n_epochs = 2000
losses = []
for _ in range(n_epochs):
    angle_batch = torch.tensor(np.sort(np.random.uniform(-np.pi, np.pi, (batch_size, n_points_compare))), dtype=torch.get_default_dtype())
    angle_metric = generative_isometry_util.linear_angle_metric(angle_batch)
    scaled_angle_metric = angle_metric/torch.mean(angle_metric)
    opt.zero_grad()
    encoded_points = encoder_net.forward(torch.unsqueeze(torch.flatten(angle_batch), -1))
    encoded_points_reshaped = torch.reshape(encoded_points, [*angle_batch.size(), -1])
    encoded_distances = generative_isometry_util.linear_point_metric(encoded_points_reshaped)
    scaled_encoded_distances = encoded_distances/torch.mean(encoded_distances)
    decoded_points = decoder_net.forward(encoded_points)
    decoded_points_reshaped = torch.reshape(decoded_points, [*angle_batch.size()])
    loss_encoding = torch.sum(torch.square(scaled_encoded_distances - scaled_angle_metric))
    loss_decoding = torch.sum(torch.square(decoded_points_reshaped - angle_batch))
    loss = loss_encoding + loss_decoding
    losses.append(loss.detach().numpy())
    print(loss)
    loss.backward()
    opt.step()

angles = np.arange(start=-np.pi, stop=np.pi, step=0.01)
with torch.no_grad():
    test_angles = torch.unsqueeze(torch.tensor(angles, dtype=torch.get_default_dtype()), -1)
    forward_pred = encoder_net.forward(test_angles)
forward_pred = forward_pred.detach().numpy()


if out_dim == 2:
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(losses)
    axs[1].scatter(forward_pred[:, 0], forward_pred[:, 1],  cmap="hsv", c=angles)
elif out_dim == 3:
    loss_fig, loss_axs = plt.subplots()
    loss_axs.plot(losses)
    proj_fig = plt.figure()
    proj_axs = proj_fig.add_subplot(projection="3d")
    proj_axs.scatter(forward_pred[:, 0], forward_pred[:, 1], forward_pred[:, 2], cmap="hsv", c=angles)

plt.show()
print("")

