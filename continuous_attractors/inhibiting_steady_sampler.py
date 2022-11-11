import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from scipy import integrate, linalg as sclinalg

from continuous_attractors import ring_attractor
from ctep import nonlinearities

N = 2 ** 7
n_samples = 1000

init_conds = np.random.uniform(-1, 1, (n_samples, N))

w_0 = 1
w_1 = 1
kernel = ring_attractor.cosine_kernel(w_0, w_1, N)


def run_equilibriation(init_conds):
    soln = integrate.solve_ivp(lambda _, y: ring_attractor.dynamics(y, kernel, np.zeros(N), nonlinearities.sigmoid),
                               [0, 20], init_conds, vectorized=True)
    return soln.y[:, -1]


p = mp.Pool()
solns = np.array(p.map(run_equilibriation, init_conds))
out_dir = "data/ring_attractor_eq/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
os.makedirs(out_dir)
out_file = os.path.join(out_dir, "data.npy")
np.save(out_file, solns)
