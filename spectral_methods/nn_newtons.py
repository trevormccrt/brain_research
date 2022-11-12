import numpy as np

from scipy import sparse
from scipy.sparse import linalg as splinalg


class SpectralNNLinearOperator(splinalg.LinearOperator):
    def __init__(self, n_neurons, n_grid):
        self.n_neurons = n_neurons
        self.n_grid = n_grid
        shape = n_neurons * n_grid
        super().__init__(None, (shape, shape))

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


def newton_iteration(u_prev, w_op: SpectralNNLinearOperator, b, nonlin, nonlin_deriv, diff_op: SpectralNNLinearOperator, init_cond):
    u_prev_grid = diff_op.gridify_vec(u_prev)
    deriv_activ_prev = nonlin_deriv(w_op.gridded_matvec(u_prev_grid) + b)
    el_op = ElementWiseMultiplication(deriv_activ_prev)
    lhs_op = diff_op + sparse.identity(w_op.shape[0]) - (el_op * w_op)
    a = lhs_op[:-1, :-1]
    rhs = -diff_op.dot(u_prev) - u_prev + nonlin(w_op.flatten_vec(w_op.gridded_matvec(u_prev_grid) + b))
