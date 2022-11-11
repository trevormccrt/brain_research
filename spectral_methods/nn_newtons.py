import numpy as np
from scipy.sparse import linalg as splinalg


class SpectralNNLinearOperator(splinalg.LinearOperator):
    def __init__(self, n_neurons, n_grid):
        self.n_neurons = n_neurons
        self.n_grid = n_grid
        super().__init__(None, n_neurons * n_grid)


class DifferentiationOperator(SpectralNNLinearOperator):
    def __init__(self, n_neurons, differentiation_matrix):
        n_grid = np.shape(differentiation_matrix)[0]
        self.differentiation_matrix = differentiation_matrix
        super().__init__(n_neurons, n_grid)

    def _matvec(self, x):
        return np.einsum(
            "it, tn -> in", self.differentiation_matrix, np.reshape(x, (self.n_grid, self.n_neurons))).flatten()


class WeightOperator(SpectralNNLinearOperator):
    def __init__(self, n_grid, weight_matrix):
        n_neurons = np.shape(weight_matrix)[0]
        self.weight_matrix = weight_matrix
        super().__init__(n_neurons * n_grid)

    def _matvec(self, x):
        return np.einsum(
            "in, tn -> ti", self.weight_matrix, np.reshape(x, (self.n_grid, self.n_neurons))).flatten()

def solve_newton_update(u_prev, w, b, nonlin, nonlin_deriv, differentiation_matrix, init_cond):
    w_u = np.einsum("ij, tj -> ti", w, u_prev)


