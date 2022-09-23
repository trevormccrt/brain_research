import numpy as np

import hopfeild
import hopfeild_ctep, costs, nonlinearities

network_units = 2

weights = np.random.uniform(-100, 100, (network_units+1, network_units+1))
weights = weights + np.transpose(weights)
np.fill_diagonal(weights, 0)
biases = np.random.uniform(-100, 100, network_units+1)

init_state = [0, 0]

cost_weights = np.zeros(network_units)
cost_weights[-1] = 1
target_y_value = 1
ground_truths = np.zeros(network_units)
ground_truths[-1] = target_y_value


t_ev = 20

all_free_end_points = []
final_weights = []
final_biases = []
iterations = 50
beta_vals =  [1, 2, 5]
for beta in beta_vals:
    this_end_points = []
    eta = beta/2
    update_weights = weights
    update_biases = biases
    for i in range(iterations):
        free_finish, _, update_biases, update_weights = hopfeild_ctep.run_cycle(
            init_state, [target_y_value], ground_truths, costs.state_quadratic_cost_deriv,
            cost_weights, beta, eta, update_weights, update_biases, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid, t_ev, t_ev)
        this_end_points.append(free_finish[-1])
    final_weights.append(update_weights)
    final_biases.append(update_biases)
    all_free_end_points.append(this_end_points)

import matplotlib.pyplot as plt


fig, axs = plt.subplots(nrows=1, ncols=1)
for beta, traj in zip(beta_vals, all_free_end_points):
    axs.plot(traj, label="Learned, β={}".format(beta))
axs.set_xlabel("# of Instruction Cycles")
axs.set_ylabel("Free Evolution Steady y Value")
axs.plot([0, iterations], [target_y_value, target_y_value], linestyle="--", color="black", label="Target")
axs.legend()

grid_vals = np.arange(start=-5, stop=5, step=0.5)
s1_grid, s2_grid = np.meshgrid(grid_vals, grid_vals)
stacked_grid = np.stack([s1_grid, s2_grid, np.ones_like(s1_grid) * target_y_value])

init_energy = hopfeild.hopfield_energy(stacked_grid, weights, biases, nonlinearities.sigmoid)
init_deriv = hopfeild.hopfeild_deriv(stacked_grid, weights, biases, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid)
final_energy = hopfeild.hopfield_energy(stacked_grid, final_weights[-1], final_biases[-1], nonlinearities.sigmoid)
final_deriv = hopfeild.hopfeild_deriv(stacked_grid, final_weights[-1], final_biases[-1], nonlinearities.sigmoid, nonlinearities.deriv_sigmoid)

fig2, axs2 = plt.subplots(nrows=1, ncols=2)
axs2[0].contour(s1_grid, s2_grid, init_energy)
axs2[0].quiver(s1_grid, s2_grid, init_deriv[0], init_deriv[1], units='xy', color='gray', scale=10)
axs2[1].contour(s1_grid, s2_grid, final_energy)
axs2[1].quiver(s1_grid, s2_grid, final_deriv[0], final_deriv[1], units='xy', color='gray', scale=10)

plt.show()
print("")