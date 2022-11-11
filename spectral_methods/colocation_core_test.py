import numpy as np

from spectral_methods import chebychev, colocation_core


def test_diff_mat_structure():
    N = 16
    grid = chebychev.extrema_grid(N)
    dmat = colocation_core.cheb_extrema_diff_mat(grid)
    for i in range(N):
        for j in range(N):
            if i == 0 and j == 0:
                expected = (1 + 2 * (N-1)**2)/6
            elif i == N-1 and j == N-1:
                expected = -(1 + 2 * (N - 1) ** 2) / 6
            elif i == j:
                expected = -grid[i]/(2 * (1 - grid[i]**2))
            else:
                c_i = 1
                if i == 0:
                    c_i += 1
                elif i == N-1:
                    c_i += 1
                c_j = 1
                if j == 0:
                    c_j += 1
                elif j == N - 1:
                    c_j += 1
                expected = (-1)**(i+j) * (c_i)/(c_j * (grid[i] - grid[j]))
            actual = dmat[i, j]
            np.testing.assert_allclose(expected, actual)


def test_diff_mat():
    N = 128
    grid = chebychev.extrema_grid(N)
    quad_coeff = np.random.uniform(1, 2)
    sin_coeff = np.random.uniform(-1, 1)
    f = (quad_coeff * (grid ** 2)) + (sin_coeff * np.sin(grid))
    deriv_mat = colocation_core.cheb_extrema_diff_mat(grid)
    deriv = deriv_mat.dot(f)
    second_deriv = deriv_mat.dot(deriv_mat).dot(f)
    expected = (2 * quad_coeff * grid) + (sin_coeff * np.cos(grid))
    expected_second = (2 * quad_coeff) - (sin_coeff * np.sin(grid))
    np.testing.assert_allclose(deriv, expected)
    np.testing.assert_allclose(second_deriv, expected_second, rtol=1e-4)
