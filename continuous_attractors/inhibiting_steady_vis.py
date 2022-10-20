import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, linalg as sclinalg

from continuous_attractors import ring_attractor
import nonlinearities

w_0 = 1
w_1 = 1

N = 2 ** 7

kernel = ring_attractor.cosine_kernel(w_0, w_1, N)

kernel_mat = sclinalg.circulant(kernel)
dft_mat = sclinalg.dft(N)
vals = np.diag(dft_mat.dot(kernel_mat).dot(np.linalg.inv(dft_mat)))

kernel_fig, kernel_axs = plt.subplots(ncols=3)
kernel_axs[0].plot(kernel)
kernel_axs[1].plot(vals)
kernel_axs[2].imshow(kernel_mat)

init_state = np.random.uniform(-0.1, 0.1, N)
soln = integrate.solve_ivp(lambda _, y: ring_attractor.dynamics(y, kernel, np.zeros(N), nonlinearities.sigmoid), [0, 20], init_state, vectorized=True)

fig, axs = plt.subplots()
axs.imshow(soln.y, aspect="auto")
plt.show()

