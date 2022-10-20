import numpy as np


def conv_circ(signal, ker):
    return np.fft.ifft(np.einsum("ij, j -> ij", np.fft.fft(signal, axis=1), np.fft.fft(ker)), axis=1)


def cosine_kernel(w_0, w_1, N):
    step = 2/(N)
    grid = np.arange(start=0, stop=2, step=step) * np.pi
    weights = -w_0 + w_1 * np.cos(grid)
    return weights


def dynamics(state, kernel, bias_vec, nonlin_fn):
    if len(np.shape(state)) == 1:
        state = np.expand_dims(state, -1)
    state = np.transpose(state)
    return np.transpose(-state + nonlin_fn(conv_circ(state, kernel) + bias_vec))
