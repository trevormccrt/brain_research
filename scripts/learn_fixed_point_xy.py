import matplotlib.pyplot as plt
import numpy as np

import hopfeild
import hopfeild_ctep, costs, nonlinearities

network_units = 2

weights = np.random.uniform(-100, 100, (network_units+1, network_units+1))
weights = weights + np.transpose(weights)
np.fill_diagonal(weights, 0)
biases = np.random.uniform(-100, 100, network_units+1)

init_state = [0, 0]

cost_weights = np.ones(network_units)
target_x_value = 1.5
target_y_value = 2
ground_truths = [target_x_value, target_y_value]


t_ev = 20

all_free_end_points = []
final_weights = []
final_biases = []
iterations = 400
beta_vals =  [0.5]
for beta in beta_vals:
    this_end_points = []
    eta = 0.5 * beta
    update_weights = weights
    update_biases = biases
    for i in range(iterations):
        free_finish, _, update_biases, update_weights = hopfeild_ctep.run_cycle(
            init_state, [target_y_value], ground_truths, costs.state_quadratic_cost_deriv,
            cost_weights, beta, eta, update_weights, update_biases, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid, t_ev, t_ev)
        this_end_points.append(free_finish)
    final_weights.append(update_weights)
    final_biases.append(update_biases)
    all_free_end_points.append(this_end_points)


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
for beta, traj in zip(beta_vals, all_free_end_points):
    traj = np.transpose(traj)
    axs[0].plot(traj[0, :])
    axs[1].plot(traj[1, :], label="Learned, β={}".format(beta))
axs[1].set_xlabel("Iterations")
axs[0].set_ylabel("Steady x Value")
axs[1].set_ylabel("Steady y Value")
axs[0].plot([0, iterations], [target_x_value, target_x_value], linestyle="--", color="black", label="Target")
axs[1].plot([0, iterations], [target_y_value, target_y_value], linestyle="--", color="black", label="Target")
axs[1].legend()

grid_vals = np.arange(start=-5, stop=5, step=0.5)
s1_grid, s2_grid = np.meshgrid(grid_vals, grid_vals)
stacked_grid = np.stack([s1_grid, s2_grid, np.ones_like(s1_grid) * target_y_value])

init_energy = hopfeild.hopfield_energy(stacked_grid, weights, biases, nonlinearities.sigmoid)
init_deriv = hopfeild.hopfeild_deriv(stacked_grid, weights, biases, nonlinearities.sigmoid, nonlinearities.deriv_sigmoid)
final_energy = hopfeild.hopfield_energy(stacked_grid, final_weights[-1], final_biases[-1], nonlinearities.sigmoid)
final_deriv = hopfeild.hopfeild_deriv(stacked_grid, final_weights[-1], final_biases[-1], nonlinearities.sigmoid, nonlinearities.deriv_sigmoid)

fig2, axs2 = plt.subplots(nrows=1, ncols=2)
xlims = [np.min(s1_grid), np.max(s1_grid)]
ylims = [np.min(s2_grid), np.max(s2_grid)]
axs2[0].contour(s1_grid, s2_grid, init_energy)
axs2[0].quiver(s1_grid, s2_grid, init_deriv[0], init_deriv[1], units='xy', color='gray', scale=10)
axs2[1].contour(s1_grid, s2_grid, final_energy)
axs2[1].quiver(s1_grid, s2_grid, final_deriv[0], final_deriv[1], units='xy', color='gray', scale=10)
axs2[0].plot([target_x_value, target_x_value], ylims, linestyle="--", color="black")
axs2[0].plot(xlims, [target_y_value, target_y_value], linestyle="--", color="black", label="target")
axs2[1].plot([target_x_value, target_x_value], ylims, linestyle="--", color="black")
axs2[1].plot(xlims, [target_y_value, target_y_value], linestyle="--", color="black", label="target")
axs2[1].legend()
axs2[0].set_title("Initial Phase Portrait")
axs2[1].set_title("Final Phase Portrait")

plt.show()
print("")
