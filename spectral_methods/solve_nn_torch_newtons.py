import numpy as np
import torch

from spectral_methods import chebychev, colocation_core, torch_nn_newtons

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

N = 64
n_neurons = 128
init_u = torch.from_numpy(np.random.uniform(-1, 1, N * n_neurons)).to(device)
grid = chebychev.extrema_grid(N)
final_time = 10
diff_mat = torch.from_numpy(2 / final_time * colocation_core.cheb_extrema_diff_mat(grid))
diff_op = torch_nn_newtons.time_matmul(diff_mat, n_neurons).to(device)
weight_mat = torch.tensor(np.random.uniform(-1, 1, (n_neurons, n_neurons)), requires_grad=True)
weight_op = torch_nn_newtons.space_matmul(weight_mat, N).to(device)
b = torch.from_numpy(np.random.uniform(-1, 1, n_neurons)).to(device)
soln, resids = torch_nn_newtons.iterate(init_u, weight_op, b, torch.sigmoid, torch_nn_newtons.deriv_sigmoid, diff_op, n_neurons, N, device=device)
soln_grid = torch_nn_newtons.vec_to_grid(soln, n_neurons)
loss = soln_grid[0, 0] ** 2
print("done soln")
grad = torch.autograd.grad(loss, weight_mat)
print(grad)
print("")
