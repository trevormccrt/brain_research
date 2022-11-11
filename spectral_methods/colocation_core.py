import numpy as np


def cheb_extrema_diff_mat(grid):
    N = len(grid)
    grid_idx = np.arange(start=0, stop=N, step=1)
    i_mat = np.tile(np.expand_dims(grid_idx, -1), [1, N])
    j_mat = np.tile(np.expand_dims(grid_idx, 0), [N, 1])
    x_i = np.tile(np.expand_dims(grid, -1), [1, N])
    x_j = np.tile(np.expand_dims(grid, 0), [N, 1])
    c_i = np.ones_like(i_mat)
    c_i[0, :] += 1
    c_i[-1, :] += 1
    c_j = np.transpose(c_i)
    d = (-1) ** (i_mat + j_mat) * c_i/(c_j * (x_i - x_j))
    diag = np.zeros_like(grid)
    diag[1:-1] = -grid[1:-1]/(2 * (1 - grid[1:-1]**2))
    diag[0] = (1 + 2 * (N-1)**2)/6
    diag[-1] = -(1 + 2 * (N-1)**2)/6
    np.fill_diagonal(d, diag)
    return d
