import numpy as np


def state_quadratic_cost(cost_weights, state, ground_truth):
    return np.sum(cost_weights * (state - ground_truth)**2)


def state_quadratic_cost_deriv(cost_weights, state, ground_truth):
    return 2 * cost_weights * (state - ground_truth)