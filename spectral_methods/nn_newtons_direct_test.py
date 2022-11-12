import matplotlib.pyplot as plt
import numpy as np

from ctep import nonlinearities
from spectral_methods import nn_newtons_direct, chebychev, colocation_core


def test_time_space_matmul():
    n_grid = 64
    n_neurons = 128
    space_mat = np.random.uniform(-1, 1, (n_neurons, n_neurons))
    time_mat = np.random.uniform(-1, 1, (n_grid, n_grid))
    x_test_grid = np.random.uniform(-1, 1, (n_grid, n_neurons))
    flat_grid = x_test_grid.flatten()
    space_actual = np.einsum("ij, tj -> ti", space_mat, x_test_grid)
    time_actual = np.einsum("ij, jg -> ig", time_mat, x_test_grid)
    space_sparse = nn_newtons_direct.space_matmul(space_mat, n_grid)
    time_sparse = nn_newtons_direct.time_matmul(time_mat, n_neurons)
    space_test = nn_newtons_direct.vec_to_grid(space_sparse.dot(flat_grid), n_neurons)
    time_test = nn_newtons_direct.vec_to_grid(time_sparse.dot(flat_grid), n_neurons)
    np.testing.assert_allclose(space_actual, space_test)
    np.testing.assert_allclose(time_actual, time_test)


def test_newtons():
    N = 64
    n_neurons = 256
    init_u = np.random.uniform(-1, 1, N * n_neurons)
    grid = chebychev.extrema_grid(N)
    final_time = 100
    diff_mat = 2 / final_time * colocation_core.cheb_extrema_diff_mat(grid)
    diff_op = nn_newtons_direct.time_matmul(diff_mat, n_neurons)
    weight_mat = np.random.uniform(-1, 1, (n_neurons, n_neurons))
    weight_mat = weight_mat
    weight_op = nn_newtons_direct.space_matmul(weight_mat, N)
    b = np.random.uniform(-1, 1, n_neurons)
    soln, resids = nn_newtons_direct.iterate(init_u, weight_op, b, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid, diff_op, n_neurons, N)
    soln_grid = nn_newtons_direct.vec_to_grid(soln, n_neurons)
    grid_t = 1 / 2 * final_time * (1 + grid)
    plt.plot(grid_t, soln_grid)
    plt.show()

