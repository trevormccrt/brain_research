import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import integrate

import numpy as np

import hopfeild, nonlinearities


weights = np.random.uniform(-100, 100, (2, 2))
weights = weights + np.transpose(weights)

biases = np.random.uniform(-100, 100, 2)

grid_vals = np.arange(start=-20, stop=20, step=1)
s1_grid, s2_grid = np.meshgrid(grid_vals, grid_vals)
stacked_grid = np.stack([s1_grid, s2_grid])
energy = hopfeild.hopfield_energy(stacked_grid, weights, biases, nonlinearities.sigmoid)
deriv = hopfeild.hopfeild_deriv(stacked_grid, weights, biases, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid)


def dynamics_fn(t, y):
    y_up = y
    if len(np.shape(y)) == 1:
        y_up = np.expand_dims(y_up, -1)
    return np.reshape(-1 * hopfeild.hopfeild_deriv(y_up, weights, biases, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid), np.shape(y))


soln = integrate.solve_ivp(dynamics_fn, (0, 20), np.array([0, 0]))

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.contour(s1_grid, s2_grid, energy, levels=10)
axs.quiver(s1_grid, s2_grid, deriv[0], deriv[1], units='xy', color='gray', scale=10)
axs.plot(soln.y[0], soln.y[1])
# arrow = FancyArrowPatch((35, 35), (35+34*0.2, 35+0), arrowstyle='simple',
#                         color='r', mutation_scale=10)
# axs.add_patch(arrow)
plt.show()
print("")