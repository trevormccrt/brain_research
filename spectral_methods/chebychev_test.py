import numpy as np
from scipy import integrate

from spectral_methods import chebychev


def test_inverse():
    N = 20
    vals = np.random.uniform(-1, 1, (1, N))
    forward = chebychev.cheb(vals)
    backward = chebychev.icheb(forward)
    np.testing.assert_allclose(vals, backward)


def test_delta():
    N = 2**5
    grid = chebychev.extrema_grid(N)
    for m in range(N):
        input = np.zeros((1, N))
        input[0, m] = 1
        out = chebychev.icheb(input)
        scale = 1
        if m == 0 or m == N-1:
            scale = 1/2
        desired = scale * np.cos(m * np.arccos(grid))
        np.testing.assert_allclose(out[0, :], desired)


def test_convergence():
    def f_test(x):
        return x**2 + np.tanh(x)

    def integrand(x, m):
        return 2/np.pi * f_test(x) * np.cos(m * np.arccos(x)) * 1/np.sqrt(1 - x**2)

    N = 2**10
    m_test = 1
    grid = np.expand_dims(chebychev.extrema_grid(N), 0)
    f_sampled = f_test(grid)
    f_trns = chebychev.cheb(f_sampled)
    exact = integrate.quad(lambda x: integrand(x, m_test), -1, 1)[0]
    fast = f_trns[0, m_test]
    np.testing.assert_allclose(exact, fast, rtol=1e-5)
