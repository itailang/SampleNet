"""PyTorch implementation of the Soft Projection block."""

import torch
import torch.nn as nn
import numpy as np

from knn_cuda import KNN
from pointnet2.utils.pointnet2_utils import grouping_operation as group_point


def knn_point(group_size, point_cloud, query_cloud):
    knn_obj = KNN(k=group_size, transpose_mode=False)
    dist, idx = knn_obj(point_cloud, query_cloud)
    return dist, idx


def _axis_to_dim(axis):
    """Translate Tensorflow 'axis' to corresponding PyTorch 'dim'"""
    return {0: 0, 1: 2, 2: 3, 3: 1}.get(axis)


class SoftProjection(nn.Module):
    def __init__(
        self,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-4,
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, 3, num_orig_points), database point cloud.
            query_cloud: A `Tensor` of shape (batch_size, 3, num_query_points), query items to project or propogate to.
            point_features [optional]: A `Tensor` of shape (batch_size, num_features, num_orig_points), features to propagate.
            action [optional]: 'project', 'propagate' or 'project_and_propagate'.
        Outputs:
            Depending on 'action':
            propagated_features: A `Tensor` of shape (batch_size, num_features, num_query_points)
            projected_points: A `Tensor` of shape (batch_size, 3, num_query_points)
        """

        super().__init__()
        self._group_size = group_size

        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                requires_grad=is_temperature_trainable,
                dtype=torch.float32,
            )
        )

        self._min_sigma = torch.tensor(min_sigma, dtype=torch.float32)

    def forward(self, point_cloud, query_cloud, point_features=None, action="project"):
        point_cloud = point_cloud.contiguous()
        query_cloud = query_cloud.contiguous()

        if action == "project":
            return self.project(point_cloud, query_cloud)
        elif action == "propagate":
            return self.propagate(point_cloud, point_features, query_cloud)
        elif action == "project_and_propagate":
            return self.project_and_propagate(point_cloud, point_features, query_cloud)
        else:
            raise ValueError(
                "action should be one of the following: 'project', 'propagate', 'project_and_propagate'"
            )

    def _group_points(self, point_cloud, query_cloud, point_features=None):
        group_size = self._group_size

        # find nearest group_size neighbours in point_cloud
        dist, idx = knn_point(group_size, point_cloud, query_cloud)

        # self._dist = dist.unsqueeze(1).permute(0, 1, 3, 2) ** 2

        idx = idx.permute(0, 2, 1).type(
            torch.int32
        )  # index should be Batch x QueryPoints x K
        grouped_points = group_point(point_cloud, idx)  # B x 3 x QueryPoints x K
        grouped_features = (
            None if point_features is None else group_point(point_features, idx)
        )  # B x F x QueryPoints x K
        return grouped_points, grouped_features

    def _get_distances(self, grouped_points, query_cloud):
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        dist = torch.sum(deltas ** 2, dim=_axis_to_dim(3), keepdim=True) / self.sigma()
        return dist

    def sigma(self):
        device = self._temperature.device
        return torch.max(self._temperature ** 2, self._min_sigma.to(device))

    def project_and_propagate(self, point_cloud, point_features, query_cloud):
        # group into (batch_size, num_query_points, group_size, 3),
        # (batch_size, num_query_points, group_size, num_features)
        grouped_points, grouped_features = self._group_points(
            point_cloud, query_cloud, point_features
        )
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))

        # get weighted average of grouped_points
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        propagated_features = torch.sum(
            grouped_features * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return (projected_points, propagated_features)

    def propagate(self, point_cloud, point_features, query_cloud):
        grouped_points, grouped_features = self._group_points(
            point_cloud, query_cloud, point_features
        )
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))

        # get weighted average of grouped_points
        propagated_features = torch.sum(
            grouped_features * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return propagated_features

    def project(self, point_cloud, query_cloud, hard=False):
        grouped_points, _ = self._group_points(point_cloud, query_cloud)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        if hard:
            raise NotImplementedError

        # get weighted average of grouped_points
        weights = weights.repeat(1, 3, 1, 1)
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        return projected_points


"""SoftProjection test"""


if __name__ == "__main__":
    k = 3
    propagator = SoftProjection(k, initial_temperature=1.0)
    query_cloud = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [5, 4, 4],
            [4, 5, 4],
            [4, 4, 5],
            [8, 7, 7],
            [7, 8, 7],
            [7, 7, 8],
        ]
    )
    point_cloud = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 5, 5], [7, 7, 8], [7, 7, 8.5]]
    )
    point_features = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
        ]
    )
    expected_nn_cloud = np.array(
        [
            [0.333, 0.333, 0.333],
            [1, 0, 0],
            [1, 0, 0],
            [4.333, 4.333, 4.333],
            [7, 7, 8],
            [7, 7, 8],
        ]
    )
    expected_features_nn_1 = np.array(
        [
            [6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [16, 17, 18, 19, 20],
            [16, 17, 18, 19, 20],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [21, 22, 23, 24, 25],
            [21, 22, 23, 24, 25],
        ]
    )
    expected_features_nn_3 = np.array(
        [
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [2.459, 3.459, 4.459, 5.459, 6.459],
            [2.459, 3.459, 4.459, 5.459, 6.459],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [22.113, 23.113, 24.113, 25.113, 26.113],
            [22.113, 23.113, 24.113, 25.113, 26.113],
            [23.189, 24.189, 25.189, 26.189, 27.189],
        ]
    )

    if k == 3:
        expected_features_nn = expected_features_nn_3
    elif k == 1:
        expected_features_nn = expected_features_nn_1
    else:
        assert False, "Non valid value of k"

    # expend to batch_size = 1
    point_cloud = np.expand_dims(point_cloud, axis=0)
    point_features = np.expand_dims(point_features, axis=0)
    query_cloud = np.expand_dims(query_cloud, axis=0)
    expected_features_nn = np.transpose(
        np.expand_dims(expected_features_nn, axis=0), (0, 2, 1)
    )
    expected_nn_cloud = np.transpose(
        np.expand_dims(expected_nn_cloud, axis=0), (0, 2, 1)
    )

    point_cloud_pl = (
        torch.tensor(point_cloud, dtype=torch.float32).permute(0, 2, 1).cuda()
    )
    point_features_pl = (
        torch.tensor(point_features, dtype=torch.float32).permute(0, 2, 1).cuda()
    )
    query_cloud_pl = (
        torch.tensor(query_cloud, dtype=torch.float32).permute(0, 2, 1).cuda()
    )

    propagator.cuda()
    # projected_points, propagated_features = propagator.project_and_propagate(point_cloud_pl, point_features_pl, query_cloud_pl)
    propagated_features = propagator.propagate(
        point_cloud_pl, point_features_pl, query_cloud_pl
    )
    propagated_features = propagated_features.cpu().detach().numpy()

    # replace Query and Point roles, reduce temperature:
    state_dict = propagator.state_dict()
    state_dict['_temperature'] = torch.tensor(0.1, dtype=torch.float32)
    propagator.load_state_dict(state_dict)
    projected_points = propagator.project(query_cloud_pl, point_cloud_pl)
    projected_points = projected_points.cpu().detach().numpy()

    print("propagated features:")
    print(propagated_features)

    print("projected points:")
    print(projected_points)

    expected_features_nn = expected_features_nn.squeeze()
    expected_nn_cloud = expected_nn_cloud.squeeze()
    propagated_features = propagated_features.squeeze()
    projected_points = projected_points.squeeze()

    mse_feat = np.mean(
        np.sum((propagated_features - expected_features_nn) ** 2, axis=1)
    )
    mse_points = np.mean(np.sum((projected_points - expected_nn_cloud) ** 2, axis=1))
    print("propagated features vs. expected NN features mse:")
    print(mse_feat)
    print("projected points vs. expected NN points mse:")
    print(mse_points)
