import numpy as np

from spectral_methods import chebychev, nn_newtons, colocation_core
from ctep import nonlinearities

def test_time_matmul():
    N = 128
    n_neurons = 30
    test_x = np.random.uniform(-1, 1, (N, n_neurons))
    diff_mat = np.random.uniform(-1, 1, (N, N))
    diff_op = nn_newtons.TimeMatmul(n_neurons, diff_mat)
    batch_deriv = diff_op.gridify_vec(diff_op.dot(diff_op.flatten_vec(test_x)))
    batch_deriv_gridded = diff_op.gridded_matvec(test_x)
    true_deriv = np.einsum("ij, jn -> in", diff_mat, test_x)
    np.testing.assert_allclose(batch_deriv, true_deriv)
    np.testing.assert_allclose(batch_deriv, batch_deriv_gridded)


def test_space_matmul():
    N = 128
    n_neurons = 30
    test_x = np.random.uniform(-1, 1, (N, n_neurons))
    weight_mat = np.random.uniform(-1, 1, (n_neurons, n_neurons))
    weight_op = nn_newtons.SpaceMatmul(N, weight_mat)
    batch_w = weight_op.gridify_vec(weight_op.dot(weight_op.flatten_vec(test_x)))
    batch_w_gridded = weight_op.gridded_matvec(test_x)
    true_w = np.einsum("ij, tj -> ti", weight_mat, test_x)
    np.testing.assert_allclose(batch_w, true_w)
    np.testing.assert_allclose(batch_w, batch_w_gridded)


def test_elementwise():
    N = 128
    n_neurons = 30
    test_x = np.random.uniform(-1, 1, (N, n_neurons))
    test_mul = np.random.uniform(-1, 1, (N, n_neurons))
    result = test_mul * test_x
    mul_op = nn_newtons.ElementWiseMultiplication(test_mul)
    result_op = mul_op.gridded_matvec(test_x)
    np.testing.assert_allclose(result, result_op)


def test_boundary():
    N = 128
    n_neurons = 128
    test_x = np.random.uniform(-1, 1, (N, n_neurons))
    weight_mat = np.random.uniform(-1, 1, (n_neurons, n_neurons))
    weight_op = nn_newtons.SpaceMatmul(N, weight_mat)
    boundary_op = nn_newtons.BoundaryBordering(n_neurons, N, -1, 1, weight_op)
    a = boundary_op.gridded_matvec(test_x)
    np.testing.assert_allclose(a[-1, :], np.ones_like(a[-1, :]))


def test_newton():
    N = 128
    n_neurons = 128
    np.random.seed(1)
    init_u = np.random.uniform(-1, 1, N * n_neurons)
    grid = chebychev.extrema_grid(N)
    final_time = 100
    diff_mat = 2/final_time * colocation_core.cheb_extrema_diff_mat(grid)
    diff_op =  nn_newtons.TimeMatmul(n_neurons, diff_mat)
    weight_mat = -1 * np.ones((n_neurons, n_neurons))
    weight_op = nn_newtons.SpaceMatmul(N, weight_mat)
    soln, resids = nn_newtons.iterate(init_u, weight_op, np.ones(n_neurons), nonlinearities.sigmoid, nonlinearities.deriv_sigmoid, diff_op, tol=1e-3)
    print(resids)

test_newton()
