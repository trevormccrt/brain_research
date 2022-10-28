import numpy as np
from scipy import integrate

import hopfeild


def _assemble_weight_matrix(weights):
    matrix_dim = int(1/2 * (1 + np.sqrt(1 + 8 * len(weights))))
    out = np.zeros((matrix_dim, matrix_dim))
    out[np.triu_indices(matrix_dim, k=1)] = weights
    return out + np.transpose(out)


def _flatten_weight_matrix(weight_matrix):
    return weight_matrix[np.triu_indices(np.shape(weight_matrix)[0], k=1)]


def _reshape_params(x, n_state):
    state_vars = x[:n_state]
    biases = x[n_state: 2 * n_state + 1]
    weights = x[2 * n_state + 1:]
    weight_matrix = _assemble_weight_matrix(weights)
    return state_vars, biases, weight_matrix


def _flatten_params(state_vars, biases, weight_matrix):
    return np.concatenate([state_vars, biases, _flatten_weight_matrix(weight_matrix)])


def fully_connected_learning_dynamics(state, inputs, ground_truth, cost_deriv, cost_weights,
                                      beta, eta, weight_matrix, biases, nonlinearity, nonlinearity_deriv):
    full_state = np.concatenate([state, inputs], axis=0)
    state_deriv = hopfeild.hopfeild_deriv(full_state, weight_matrix, biases, nonlinearity,
                                        nonlinearity_deriv)
    state_deriv[np.shape(state)[0]:] = 0
    nonlin_state = nonlinearity(full_state)
    nonlin_derivs = nonlinearity_deriv(full_state)
    weight_derivs = -1/2 * (np.einsum("i..., i..., j... -> ij...", nonlin_derivs, state_deriv, nonlin_state) +
                            np.einsum("i..., j..., j... -> ij...", nonlin_state, nonlin_derivs, state_deriv))
    bias_derivs = -1 * nonlin_derivs * state_deriv
    np.fill_diagonal(weight_derivs, 0)
    return -1 * state_deriv[:np.shape(state)[0]] - beta * cost_deriv(cost_weights, state, ground_truth), \
    -eta/beta * bias_derivs, -eta/beta * weight_derivs


def run_cycle(init_state, inputs, ground_truth, cost_deriv, cost_weights,
                                      beta, eta, init_weight_matrix, init_biases, nonlinearity, nonlinearity_deriv,
              free_time, forced_time):
    def free_dynamics(_, y):
        full_state = np.concatenate([y, inputs])
        return -1 * hopfeild.hopfeild_deriv(full_state, init_weight_matrix, init_biases, nonlinearity, nonlinearity_deriv)[:-1]

    free_soln = integrate.solve_ivp(free_dynamics, [0, free_time], init_state)

    def forced_dynamics(_, y):
        state, biases, weight_matrix = _reshape_params(y, np.shape(init_state)[0])
        return _flatten_params(*fully_connected_learning_dynamics(state, inputs, ground_truth, cost_deriv,
                                                                     cost_weights, beta, eta, weight_matrix,
                                                                     biases, nonlinearity, nonlinearity_deriv))

    init_forced_state = _flatten_params(free_soln.y[:, -1], init_biases, init_weight_matrix)
    forced_soln = integrate.solve_ivp(forced_dynamics, [0, forced_time], init_forced_state)
    final_state, final_biases, final_weights = _reshape_params(forced_soln.y[:, -1], np.shape(init_state)[0])
    return free_soln.y[:, -1], final_state, final_biases, final_weights

