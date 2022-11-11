import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from spectral_methods import chebychev, colocation_core
from ctep import nonlinearities


def solve_newton_update(u_prev, w, b, nonlin, nonlin_deriv, differentiation_matrix, init_cond):
    lhs = differentiation_matrix + np.diag(1 - w * nonlin_deriv(w * u_prev + b))
    lhs[-1, :] = np.zeros_like(u_prev)
    lhs[-1, -1] = 1
    rhs = - differentiation_matrix.dot(u_prev) - u_prev  + nonlin(w * u_prev + b)
    rhs[-1] = init_cond
    return np.linalg.solve(lhs, rhs)


N = 2**10
final_time = 100000
grid = chebychev.extrema_grid(N)
d_mat = 2/final_time * colocation_core.cheb_extrema_diff_mat(grid)
w = -1
b = 1
tol = 1e-5

init_u = np.random.uniform(-1, 1, len(grid))

this_u = init_u
init_cond = init_u[-1]
print(init_cond)
all_resids = []
for i in range(100000):
    this_v = solve_newton_update(this_u, w, b, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid, d_mat, 0)
    resid = np.max(np.abs(this_v))/np.max(np.abs(this_u))
    all_resids.append(resid)
    this_u = this_u + this_v
    if resid < tol:
        break


def dynamics(y, w, b, nonlin_fn):
    return -y + nonlin_fn(w * y + b)


soln = integrate.solve_ivp(lambda _, y: dynamics(y, w, b, nonlinearities.sigmoid), [0, final_time], [init_cond], rtol=tol)

num_t = len(soln.t)
print(num_t)
grid_t = 1/2 * final_time * (1 + grid)

fig, axs = plt.subplots(ncols=2)
axs[0].plot(all_resids)
axs[0].set_yscale("log")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Residual")
axs[1].plot(grid_t, this_u)
axs[1].plot(soln.t, soln.y[0, :])
#axs[1].legend()
axs[1].set_xlabel("x")
axs[1].set_ylabel("u(x)")
plt.show()
