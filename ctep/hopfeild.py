import numpy as np


def hopfield_energy(state, weights, biases, nonlinearity):
    non_diag_weights = np.copy(weights)
    np.fill_diagonal(non_diag_weights, 0)
    activated_state = nonlinearity(state)
    return 1/2 * np.einsum("i..., i... -> ...", state, state) - \
           1/2 * np.einsum("i..., ij, j... -> ...", activated_state, non_diag_weights, activated_state) -\
           np.einsum("i, i... -> ...", biases, activated_state)


def hopfeild_deriv(state, weights, biases, nonlinearity, nonlinearity_deriv):
    non_diag_weights = np.copy(weights)
    np.fill_diagonal(non_diag_weights, 0)
    activated_state = nonlinearity(state)
    deriv_activated_state = nonlinearity_deriv(state)
    inner_sum = np.transpose(np.transpose(np.einsum("ij..., j... -> i...", non_diag_weights, activated_state)) + biases)
    return -np.einsum("i..., i... -> i...", deriv_activated_state, inner_sum) + state


