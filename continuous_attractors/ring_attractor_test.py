import numpy as np
from scipy import linalg as sclinalg


from continuous_attractors import ring_attractor


def test_circular_conv():
    N = 2**4
    signal = np.random.uniform(-1, 1, N)
    kernel = np.random.uniform(-1, 1, N)
    mat = sclinalg.circulant(kernel)
    conv_matmul = mat.dot(signal)
    conv_fft = ring_attractor.conv_circ(np.expand_dims(signal, 0), kernel)
    np.testing.assert_allclose(conv_matmul, conv_fft[0, :])


def test_cosine_kernel():
    w_0 = 1
    w_1 = 1
    N = 2 ** 5
    kernel = ring_attractor.cosine_kernel(w_0, w_1, N)
    kernel_mat = sclinalg.circulant(kernel)
    np.testing.assert_allclose(kernel_mat, np.transpose(kernel_mat))

