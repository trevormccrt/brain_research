import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from continuous_attractors import generative_isometry_util


in_dim = 2
out_dim = 3
encoder_hidden_dim=1000
encoder_n_hidden = 2
decoder_hidden_dim = encoder_hidden_dim
decoder_n_hidden = encoder_n_hidden
n_resample = 50

encoder_layers = []
encoder_layers.append(nn.Linear(in_dim, int(encoder_hidden_dim/2)))
encoder_layers.append(nn.ReLU())
for _ in range(encoder_n_hidden):
    encoder_layers.append(nn.LazyLinear(encoder_hidden_dim))
    encoder_layers.append(nn.ReLU())
encoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
encoder_layers.append(nn.ReLU())
encoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), out_dim))
encoder_net = nn.Sequential(*encoder_layers)

decoder_layers = []
decoder_layers.append(nn.Linear(out_dim, int(encoder_hidden_dim/2)))
decoder_layers.append(nn.ReLU())
for _ in range(encoder_n_hidden):
    decoder_layers.append(nn.LazyLinear( encoder_hidden_dim))
    decoder_layers.append(nn.ReLU())
decoder_layers.append(nn.Linear(encoder_hidden_dim, int(encoder_hidden_dim / 2)))
decoder_layers.append(nn.ReLU())
decoder_layers.append(nn.Linear(int(encoder_hidden_dim / 2), in_dim))
decoder_net = nn.Sequential(*decoder_layers)
params = list(encoder_net.parameters()) + list(decoder_net.parameters())
opt = torch.optim.Adam(params)
torch.autograd.set_detect_anomaly(True)
batch_size = 20
n_points_compare = 20
n_epochs = 200
encoder_losses = []
decoder_losses = []
for _ in range(n_epochs):
    angles_numpy = np.sort(np.random.uniform(-np.pi, np.pi, (batch_size, n_points_compare)))
    angles_span = generative_isometry_util.densely_sample_angles(angles_numpy, n_resample)
    mapped_init_angles = torch.tensor(generative_isometry_util.angles_to_ring(angles_numpy), dtype=torch.get_default_dtype())
    mapped_span = torch.tensor(generative_isometry_util.angles_to_ring(angles_span), dtype=torch.get_default_dtype())
    angle_batch = torch.tensor(angles_span, dtype=torch.get_default_dtype())
    angle_metric = torch.tensor(generative_isometry_util.integrated_angle_metric(angles_numpy), dtype=torch.get_default_dtype())
    scaled_angle_metric = angle_metric/torch.mean(angle_metric)
    opt.zero_grad()
    encoded_points = encoder_net.forward(torch.reshape(mapped_span, [-1, mapped_span.size(-1)]))
    encoded_points_reshaped = torch.reshape(encoded_points, [*angle_batch.size(), -1])
    encoded_distances = generative_isometry_util.integrated_point_metric(encoded_points_reshaped)
    scaled_encoded_distances = encoded_distances/torch.mean(encoded_distances)
    first_encoded_points = encoded_points_reshaped[:, :, 0, :]
    flat_first_encoded_points = torch.reshape(first_encoded_points, [-1, encoded_points.size(-1)])
    decoded_points = decoder_net.forward(flat_first_encoded_points)
    decoded_points_reshaped = torch.reshape(decoded_points, mapped_init_angles.size())
    loss_encoding = torch.sum(torch.square(scaled_encoded_distances - scaled_angle_metric))
    loss_decoding = torch.sum(torch.square(decoded_points_reshaped - mapped_init_angles))
    loss = loss_encoding + loss_decoding
    encoder_losses.append(loss_encoding.detach().numpy())
    decoder_losses.append(loss_decoding.detach().numpy())
    print("encoding loss: {}, decoding loss: {}".format(loss_encoding.detach().numpy(), loss_decoding.detach().numpy()))
    loss.backward()
    opt.step()

angles = np.arange(start=-np.pi, stop=np.pi, step=0.01)
with torch.no_grad():
    test_points = torch.tensor(generative_isometry_util.angles_to_ring(angles), dtype=torch.get_default_dtype())
    forward_pred = encoder_net.forward(test_points)
    decoder_pred = decoder_net.forward(forward_pred)
forward_pred = forward_pred.detach().numpy()
decoder_pred = decoder_pred.detach().numpy()
print("")

if out_dim == 2:
    fig, axs = plt.subplots(nrows=1, ncols=4)
    axs[0].plot(encoder_losses)
    axs[1].plot(decoder_losses)
    axs[2].scatter(forward_pred[:, 0], forward_pred[:, 1],  cmap="hsv", c=angles)
    axs[3].scatter(decoder_pred[:, 0], decoder_pred[:, 1], cmap="hsv", c=angles)
    axs[3].plot(np.cos(angles), np.sin(angles), color="black", linestyle="--")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Encoder Loss")
    axs[0].set_yscale("log")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Decoder Loss")
    axs[1].set_yscale("log")
    axs[2].set_title("Encoded Representation")
    axs[3].set_title("Decoder Output")
    fig.tight_layout()

elif out_dim == 3:
    loss_fig, loss_axs = plt.subplots(ncols=2)
    loss_axs[0].plot(encoder_losses)
    loss_axs[1].plot(decoder_losses)
    loss_axs[0].set_xlabel("Epochs")
    loss_axs[0].set_ylabel("Encoder Loss")
    loss_axs[0].set_yscale("log")
    loss_axs[1].set_xlabel("Epochs")
    loss_axs[1].set_ylabel("Decoder Loss")
    loss_axs[1].set_yscale("log")
    proj_fig = plt.figure()
    proj_axs = proj_fig.add_subplot(projection="3d")
    proj_axs.scatter(forward_pred[:, 0], forward_pred[:, 1], forward_pred[:, 2], cmap="hsv", c=angles)
    proj_axs.set_title("Encoded Representation")
    decoder_fig, decoder_axs = plt.subplots()
    decoder_axs.scatter(decoder_pred[:, 0], decoder_pred[:, 1], cmap="hsv", c=angles)
    decoder_axs.plot(np.cos(angles), np.sin(angles), color="black", linestyle="--")
    decoder_axs.set_title("Decoder Output")

else:
    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].plot(encoder_losses)
    axs[1].plot(decoder_losses)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Encoder Loss")
    axs[0].set_yscale("log")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Decoder Loss")
    axs[1].set_yscale("log")
    axs[2].scatter(decoder_pred[:, 0], decoder_pred[:, 1], cmap="hsv", c=angles)
    axs[2].plot(np.cos(angles), np.sin(angles), color="black", linestyle="--")
    axs[2].set_title("Decoder Output")
plt.show()
print("")

