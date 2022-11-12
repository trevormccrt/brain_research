import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from spectral_methods import chebychev, colocation_core, torch_nn_newtons

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class SimpleNN(nn.Module):
    def __init__(self, weights, biases, nonlin=torch.sigmoid):
        super().__init__()
        self.nonlin = nonlin
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, _, y):
        return -y + self.nonlin(torch.einsum("ij, ...j -> ...i", self.weights, y) + self.biases)


n_neurons = 128
weight_mat = torch.tensor(np.random.uniform(-1, 1, (n_neurons, n_neurons)), requires_grad=True).to(device)
b = torch.tensor(np.random.uniform(-1, 1, n_neurons)).to(device)
final_time = 100.0
init_state = np.random.uniform(-1, 1, n_neurons)

all_step_grads = []
all_global_grads = []
for i in range(10):
    # timestepping solve
    dynamics = SimpleNN(weight_mat, b, torch.sigmoid)
    timestep_soln = odeint(dynamics, torch.tensor(init_state).to(device), torch.tensor([0, final_time]).to(device))
    loss_step = timestep_soln[-1, 0] ** 2
    step_grad = torch.autograd.grad(loss_step, dynamics.weights)
    # global solve
    N = 64
    init_u = 25 * np.random.uniform(0, 1, N * n_neurons)
    init_u[-n_neurons:] = init_state
    init_u = torch.from_numpy(init_u).to(device)

    grid = chebychev.extrema_grid(N)
    grid_t = 1 / 2 * final_time * (1 + grid)
    diff_mat = torch.tensor(2 / final_time * colocation_core.cheb_extrema_diff_mat(grid)).to(device)
    diff_op = torch_nn_newtons.time_matmul(diff_mat, n_neurons, device=device)
    weight_op = torch_nn_newtons.space_matmul(weight_mat, N, device=device)
    global_soln, _ = torch_nn_newtons.iterate(init_u, weight_op, b, torch.sigmoid, torch_nn_newtons.deriv_sigmoid, diff_op, n_neurons, N, device=device)
    global_soln_grid = torch_nn_newtons.vec_to_grid(global_soln, n_neurons)
    loss_global = global_soln_grid[0, 0] ** 2
    print("done soln")
    global_grad = torch.autograd.grad(loss_global, weight_mat)
    print(global_grad)
    all_step_grads.append(step_grad)
    all_global_grads.append(global_grad)
    print("")
