import numpy as np

from scipy import sparse
from scipy.sparse import linalg as splinalg


class SpectralNNLinearOperator(splinalg.LinearOperator):
    def __init__(self, n_neurons, n_grid):
        self.n_neurons = n_neurons
        self.n_grid = n_grid
        shape = n_neurons * n_grid
        super().__init__(float, (shape, shape))

    def gridify_vec(self, x):
        return np.reshape(x, (self.n_grid, self.n_neurons))

    def flatten_vec(self, x):
        return x.flatten()

    def gridded_matvec(self, x):
        return self.gridify_vec(self.dot(self.flatten_vec(x)))


class TimeMatmul(SpectralNNLinearOperator):
    def __init__(self, n_neurons, differentiation_matrix):
        n_grid = np.shape(differentiation_matrix)[0]
        self.differentiation_matrix = differentiation_matrix
        super().__init__(n_neurons, n_grid)

    def _matvec(self, x):
        return self.flatten_vec(np.einsum("it, tn -> in", self.differentiation_matrix, self.gridify_vec(x)))


class SpaceMatmul(SpectralNNLinearOperator):
    def __init__(self, n_grid, weight_matrix):
        n_neurons = np.shape(weight_matrix)[0]
        self.weight_matrix = weight_matrix
        super().__init__(n_neurons, n_grid)

    def _matvec(self, x):
        return self.flatten_vec(np.einsum("in, tn -> ti", self.weight_matrix, self.gridify_vec(x)))


class ElementWiseMultiplication(SpectralNNLinearOperator):
    def __init__(self, to_mul):
        self.to_mul = to_mul
        self.flat_to_mul = self.flatten_vec(to_mul)
        super().__init__(*np.shape(to_mul)[::-1])

    def _matvec(self, x):
        return self.flat_to_mul * x


class BoundaryBordering(SpectralNNLinearOperator):
    def __init__(self, n_neurons, n_grid, boundary, boundary_value, op_to_border):
        super().__init__(n_neurons, n_grid)
        self.boundary = boundary
        self.boundary_value = boundary_value
        self.op_to_border = op_to_border

    def _matvec(self, x):
        x_grid_orig = self.gridify_vec(x)
        x_forward = self.op_to_border.dot(x)
        gridded_x = self.gridify_vec(x_forward)
        gridded_x[self.boundary, :] = self.boundary_value * x_grid_orig[self.boundary, :]
        return self.flatten_vec(gridded_x)


def newton_iteration(u_prev, w_op: SpectralNNLinearOperator, b, nonlin, nonlin_deriv, diff_op: SpectralNNLinearOperator):
    u_prev_grid = diff_op.gridify_vec(u_prev)
    n_grid = u_prev_grid.shape[0]
    n_neurons = u_prev_grid.shape[1]
    deriv_activ_prev = nonlin_deriv(w_op.gridded_matvec(u_prev_grid) + b)
    el_op = ElementWiseMultiplication(deriv_activ_prev)
    spatial_id = SpaceMatmul(n_grid, np.eye(n_neurons))
    lhs_op = diff_op + spatial_id - (el_op.dot(w_op))
    lhs_with_boundaries = BoundaryBordering(n_neurons, n_grid, -1, 1, lhs_op)
    a = diff_op.dot(u_prev)
    rhs = -a - u_prev + nonlin(w_op.flatten_vec(w_op.gridded_matvec(u_prev_grid) + b))
    soln = splinalg.cgs(lhs_with_boundaries, rhs)
    if soln[1] > 0:
        raise ValueError("CGS Failed to Converge")
    return soln[0]


def iterate(init_u, w_op, b, nonlin, nonlin_deriv, diff_op, tol=1e-5, max_iter=100000):
    this_u = init_u
    all_resids = []
    for i in range(max_iter):
        this_v = newton_iteration(this_u, w_op, b, nonlin, nonlin_deriv, diff_op)
        resid = np.max(np.abs(this_v)) / np.max(np.abs(this_u))
        all_resids.append(resid)
        this_u = this_u + this_v
        print(resid)
        if resid < tol:
            break
    return this_u, all_resids
