import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "/home/trevor/brain_research/continuous_attractors/data/ring_attractor_eq/2022-11-08-19-22-47"
data_path = os.path.join(data_dir, "data.npy")

data = np.load(data_path)
fig, axs = plt.subplots()
axs.imshow(data, aspect="auto")

N = np.shape(data)[1]
n_projections = 4

random_projections = np.random.normal(0, 1, (n_projections, 3, N))
col_norm = np.einsum("ijk, ijk -> ik", random_projections, random_projections)
normed_proj = np.einsum("ijk, ik -> ijk", random_projections, 1/np.sqrt(col_norm))
projected_data = np.einsum("bk, ijk -> bij", data, normed_proj)

max_pos = (np.argmax(data, axis=1)/N) * 2 * np.pi


for i in range(n_projections):
    proj_fig = plt.figure()
    proj_axs = proj_fig.add_subplot(projection="3d")
    proj_axs.scatter(projected_data[:, i, 0], projected_data[:, i, 1], projected_data[:, i, 2] , c=max_pos)
plt.show()

out_dir = os.path.join(os.getenv("HOME"), "manifold_test_data/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
np.save(os.path.join(out_dir, "true_phases.npy"), max_pos)
np.save(os.path.join(out_dir, "encoded_points.npy"), data/np.mean(np.abs(data)))
print("")
