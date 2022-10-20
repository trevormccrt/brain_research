import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def parameterized_dynamics(_, y, phi, x, beta):
    r, theta = y
    dr = (2 - np.cos(theta)) * (-r * (r**2 - phi**2) - beta * (r - x))
    dtheta = r
    return np.array([dr, dtheta])


def lyapunov_fn(r, phi, x, beta):
    return 1/4 * r * (r**3 - 4 * x * beta + 2 * r * (beta - phi**2))


phi = 1
x = 2
beta = 0.1
t_solve = np.arange(start=0, stop=10, step=0.05)
dynamics_fn = lambda _, y: parameterized_dynamics(_, y, phi, x, beta)
soln = integrate.solve_ivp(dynamics_fn, [np.min(t_solve), np.max(t_solve)], [0.1, 0.1], t_eval=t_solve)
lyap_vals = lyapunov_fn(soln.y[0, :], phi, x, beta)
fig, axs = plt.subplots(nrows=1, ncols=2)

x_solns = soln.y[0, :] * np.cos(soln.y[1, :])
y_solns = soln.y[0, :] * np.sin(soln.y[1, :])
axs[0].plot(x_solns, y_solns)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].plot(soln.t, lyap_vals)
axs[1].set_xlabel("t")
axs[1].set_ylabel("V")
fig.tight_layout()
plt.show()
print("")
