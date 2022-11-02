import numpy as np
import torch


def angle_metric_1d(angles):
    angle_horiz = torch.tile(torch.unsqueeze(angles, 1), [1, angles.size(-1), 1])
    angles_vert = torch.tile(torch.unsqueeze(angles, -1), [1, 1, angles.size(-1)])
    return 2 * (1 - torch.cos(angles_vert - angle_horiz))


def point_metric(points):
    points_horiz = torch.tile(torch.unsqueeze(points, 1), [1, points.size(1), 1, 1])
    points_vert = torch.tile(torch.unsqueeze(points, 2), [1, 1, points.size(1), 1])
    dists = torch.sum(torch.square(points_horiz - points_vert), -1)
    return dists


def linear_point_metric(points):
    points_rolled = torch.roll(points, 1, dims=1)
    return torch.sum(torch.square(points - points_rolled), -1)


def linear_angle_metric(angles):
    angles_rolled = torch.roll(angles, 1, dims=-1)
    return 2 * (1 - torch.cos(angles - angles_rolled))


def integrated_angle_metric(angles):
    angles_rolled = np.roll(angles, 1, axis=-1)
    return np.abs(angles - angles_rolled)


def integrated_point_metric(points):
    rolled_points = torch.roll(points, 1, dims=-2)
    return torch.sum(torch.sqrt(torch.sum(torch.square(points[:, :, 1:, :] - rolled_points[:, :, 1:, :]), dim=-1) + 1e-13), dim=-1)


def densely_sample_angles(angles, n_samples):
    angles_rolled = np.roll(angles, 1, axis=1)
    return np.linspace(start=angles, stop=angles_rolled, num=n_samples, axis=-1)


def angles_to_ring(angles):
    points_x = np.cos(angles)
    points_y = np.sin(angles)
    return np.stack([points_x, points_y], axis=-1)
