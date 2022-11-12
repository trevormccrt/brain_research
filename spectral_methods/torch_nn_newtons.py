import torch


def deriv_sigmoid(x):
    return torch.exp(x)/(1 + torch.exp(x))**2


def vec_to_grid(x, n_neurons):
    return torch.reshape(x, (-1, n_neurons))


def space_matmul(space_mat, time_dim, device="cpu"):
    return torch.kron(torch.eye(time_dim).to(device), space_mat)


def time_matmul(time_mat, space_dim, device="cpu"):
    return torch.kron(time_mat, torch.eye(space_dim).to(device))


def newton_iteration(u_prev, w_op, b, nonlin, nonlin_deriv, diff_op, n_neurons, n_grid, device="cpu"):
    b_tile = torch.kron(torch.ones(n_grid).to(device), b)
    prev_forward = torch.matmul(w_op, u_prev) + b_tile
    deriv_prev_forward = nonlin_deriv(prev_forward)
    a = torch.matmul(torch.diag(deriv_prev_forward).to(device), w_op)
    lhs = diff_op + torch.eye(u_prev.size(0)).to(device) - a
    lhs[-n_neurons:, :] = 0
    for i in range(n_neurons):
        lhs[-(i+1), -(i+1)] = 1
    rhs = -torch.matmul(diff_op, u_prev) - u_prev + nonlin(prev_forward)
    rhs[-n_neurons:] = 0
    return torch.linalg.solve(lhs, rhs)


def iterate(init_u, w_op, b, nonlin, nonlin_deriv, diff_op, n_neurons, n_grid, tol=1e-5, max_iter=100000, device="cpu"):
    this_u = init_u
    all_resids = []
    for i in range(max_iter):
        this_v = newton_iteration(this_u, w_op, b, nonlin, nonlin_deriv, diff_op, n_neurons, n_grid, device=device)
        resid = torch.max(torch.abs(this_v)) / torch.max(torch.abs(this_u))
        all_resids.append(resid)
        this_u = this_u + this_v
        print(resid)
        if resid < tol:
            break
    return this_u, all_resids
