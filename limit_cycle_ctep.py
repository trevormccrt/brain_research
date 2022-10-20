import numpy as np
from scipy import integrate

def main_dynamics(r, theta, phi, x, beta):
    dr = (2 - np.cos(theta)) * (-r * (r**2 - phi) - beta * (r - x))
    dtheta = r + 1
    return np.array([dr, dtheta])


def phi_dynamics(r, dr, eta_beta):
    return eta_beta * r * dr


def run_cycle(init_state, cycle_time, init_phi, x, beta, eta_beta):
    free_dynamics = lambda _, y: main_dynamics(*y, init_phi, x, 0)
    free_soln = integrate.solve_ivp(free_dynamics, [0, cycle_time], init_state)
    final_free_pos = free_soln.y[:, -1]

    def clamped_dynamics(_, y):
        r, theta, phi = y
        dr, dtheta = main_dynamics(r, theta, phi, x, beta)
        dphi = phi_dynamics(r, dr, eta_beta)
        return np.array([dr, dtheta, dphi])

    clamped_soln = integrate.solve_ivp(clamped_dynamics, [0, cycle_time], np.concatenate([final_free_pos, [init_phi]]))
    return free_soln.y[:, -1], clamped_soln.y[:, -1]


n_cycles = 60
init_phi = 1
x = 0.2
beta = 0.3
eta_beta = 0.3
init_r = 0.1
init_theta = 0.1
cycle_time = 100


phi = init_phi
free_r_vals = []
for _ in range(n_cycles):
    free_soln, clamped_soln = run_cycle(np.array([init_r, init_theta]), cycle_time, phi, x, beta, eta_beta)
    r = free_soln[0]
    phi = clamped_soln[-1]
    free_r_vals.append(r)
    print(r)
    print(phi)


def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

t_solve_display = np.arange(start=0, stop=cycle_time, step=0.02)

init_free_dynamics = lambda _, y: main_dynamics(*y, init_phi, x, 0)
init_free_soln = integrate.solve_ivp(init_free_dynamics, [0, cycle_time], [init_r, init_theta], t_eval=t_solve_display)
init_x, init_y = polar_to_cartesian(init_free_soln.y[0, :], init_free_soln.y[1, :])


final_free_dynamics = lambda _, y: main_dynamics(*y, phi, x, 0)
final_free_soln = integrate.solve_ivp(final_free_dynamics, [0, cycle_time], [init_r, init_theta], t_eval=t_solve_display)
final_x, final_y = polar_to_cartesian(final_free_soln.y[0, :], final_free_soln.y[1, :])

desired_cycle_theta = np.arange(start=0, stop=2 * np.pi, step=0.05)
desired_cycle_r = np.ones_like(desired_cycle_theta) * x
desired_x, desired_y = polar_to_cartesian(desired_cycle_r, desired_cycle_theta)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(init_x, init_y, label="initial trajectory", color="C0")
axs[0].plot(final_x, final_y, label="final trajectory", color="C1")
axs[0].plot(desired_x, desired_y, label="desired orbit", color="black", linestyle="--")
axs[0].legend()
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].plot(free_r_vals, color="C2")
line_x = np.arange(start=0, stop=len(free_r_vals), step=0.1)
line_y = np.ones_like(line_x) * x
axs[1].plot(line_x, line_y, color="black", linestyle="--")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Equalibrium Cycle Radius")
fig.tight_layout()
fig.show()
plt.show()
