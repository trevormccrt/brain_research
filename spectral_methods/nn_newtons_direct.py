import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


def vec_to_grid(x, n_neurons):
    return np.reshape(x, (-1, n_neurons))


def space_matmul(space_mat, time_dim):
    return sparse.kron(sparse.identity(time_dim), space_mat)


def time_matmul(time_mat, space_dim):
    return sparse.kron(time_mat, sparse.eye(space_dim, space_dim)).tocsr()


def newton_iteration(u_prev, w_op, b, nonlin, nonlin_deriv, diff_op, n_neurons, n_grid):
    b_tile = np.kron(np.ones(n_grid), b)
    prev_forward = w_op.dot(u_prev) + b_tile
    deriv_prev_forward = nonlin_deriv(prev_forward)
    a = sparse.diags([deriv_prev_forward], [0]).dot(w_op)
    lhs = diff_op + sparse.identity(u_prev.shape[0]) - a
    lhs[-n_neurons:, :] = 0
    for i in range(n_neurons):
        lhs[-(i+1), -(i+1)] = 1
    rhs = -diff_op.dot(u_prev) - u_prev + nonlin(prev_forward)
    rhs[-n_neurons:] = 0
    return splinalg.spsolve(lhs, rhs)


def iterate(init_u, w_op, b, nonlin, nonlin_deriv, diff_op, n_neurons, n_grid, tol=1e-5, max_iter=100000):
    this_u = init_u
    all_resids = []
    for i in range(max_iter):
        this_v = newton_iteration(this_u, w_op, b, nonlin, nonlin_deriv, diff_op, n_neurons, n_grid)
        resid = np.max(np.abs(this_v)) / np.max(np.abs(this_u))
        all_resids.append(resid)
        this_u = this_u + this_v
        print(resid)
        if resid < tol:
            break
    return this_u, all_resids
