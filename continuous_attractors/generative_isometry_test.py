import numpy as np
import torch

from continuous_attractors import generative_isometry_util


def test_point_metric():
    points = np.random.uniform(-1, 1, (1, 5, 3))
    point_list_len = np.shape(points)[1]
    distance_mat = np.zeros((point_list_len, point_list_len))
    for i in range(point_list_len):
        for j in range(point_list_len):
            distance_mat[i, j] = np.sum(np.square(points[0, i, :] - points[0, j, :]))
    torch_dist_mat = generative_isometry_util.point_metric(torch.from_numpy(points)).detach().numpy()
    np.testing.assert_allclose(distance_mat, torch_dist_mat[0, :, :])


def test_angle_metric():
    angles = np.random.uniform(-np.pi, np.pi, (1, 10))
    x_vals = np.cos(angles)
    y_vals = np.sin(angles)
    points = np.transpose(np.stack([x_vals, y_vals]), [1, 2, 0])
    angle_dist = generative_isometry_util.angle_metric_1d(torch.from_numpy(angles)).detach().numpy()
    point_dist = generative_isometry_util.point_metric(torch.from_numpy(points)).detach().numpy()
    np.testing.assert_allclose(angle_dist, point_dist)


def test_linear_angle_metric():
    angles = np.random.uniform(-np.pi, np.pi, (1, 10))
    x_vals = np.cos(angles)
    y_vals = np.sin(angles)
    points = np.transpose(np.stack([x_vals, y_vals]), [1, 2, 0])
    angle_dist = generative_isometry_util.linear_angle_metric(torch.from_numpy(angles)).detach().numpy()
    point_dist = generative_isometry_util.linear_point_metric(torch.from_numpy(points)).detach().numpy()
    np.testing.assert_allclose(angle_dist, point_dist)
