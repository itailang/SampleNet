from __future__ import print_function

# import system modules
from builtins import object
import os.path as osp
import sys

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from external.grouping.tf_grouping import group_point, knn_point
import tensorflow as tf
import numpy as np


class SoftProjection(object):
    def __init__(
        self, group_size, initial_temperature=1.0, is_temperature_trainable=True
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, num_in_points, 3), original point cloud.
            query_cloud: A `Tensor` of shape (batch_size, num_out_points, 3), generated point cloud
        Outputs:
            projected_point_cloud: A `Tensor` of shape (batch_size, num_out_points, 3),
                the query_cloud projected onto its group_size nearest neighborhood,
                controlled by the learnable temperature parameter.
            weights: A `Tensor` of shape (batch_size, num_out_points, group_size, 1),
                the projection weights of the query_cloud onto its group_size nearest neighborhood
            dist: A `Tensor` of shape (batch_size, num_out_points, group_size, 1),
                the square distance of each query point from its neighbors divided by squared temperature parameter
        """

        self._group_size = group_size

        # create temperature variable
        self._temperature = tf.get_variable(
            "temperature",
            initializer=tf.constant(initial_temperature, dtype=tf.float32),
            trainable=is_temperature_trainable,
            dtype=tf.float32,
        )

        self._temperature_safe = tf.maximum(self._temperature, 1e-2)

        # sigma is exposed for loss calculation
        self.sigma = self._temperature_safe ** 2

    def __call__(self, point_cloud, query_cloud, hard=False):
        return self.project(point_cloud, query_cloud, hard)

    def _group_points(self, point_cloud, query_cloud):
        group_size = self._group_size
        _, num_out_points, _ = query_cloud.shape

        # find nearest group_size neighbours in point_cloud
        _, idx = knn_point(group_size, point_cloud, query_cloud)
        grouped_points = group_point(point_cloud, idx)
        return grouped_points

    def _get_distances(self, grouped_points, query_cloud):
        group_size = self._group_size

        # remove centers to get absolute distances
        deltas = grouped_points - tf.tile(
            tf.expand_dims(query_cloud, 2), [1, 1, group_size, 1]
        )
        dist = tf.reduce_sum(deltas ** 2, axis=3, keepdims=True) / self.sigma
        return dist

    def project(self, point_cloud, query_cloud, hard):
        grouped_points = self._group_points(
            point_cloud, query_cloud
        )  # (batch_size, num_out_points, group_size, 3)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = tf.nn.softmax(-dist, axis=2)
        if hard:
            # convert softmax weights to one_hot encoding
            weights = tf.one_hot(tf.argmax(weights, axis=2), depth=self._group_size)
            weights = tf.transpose(weights, perm=[0, 1, 3, 2])

        # get weighted average of grouped_points
        projected_point_cloud = tf.reduce_sum(
            grouped_points * weights, axis=2
        )  # (batch_size, num_out_points, 3)
        return projected_point_cloud, weights, dist


"""SoftProjection test"""
if __name__ == "__main__":
    tf.enable_eager_execution()
    projector = SoftProjection(3, initial_temperature=0.01)
    sigma = projector.sigma
    point_cloud = np.array(
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
    query_cloud = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 5, 5], [7, 7, 8], [7, 7, 8.5]]
    )
    expected_cloud_soft = np.array(
        [
            [0.333, 0.333, 0.333],
            [1, 0, 0],
            [1, 0, 0],
            [4.333, 4.333, 4.333],
            [7, 7, 8],
            [7, 7, 8],
        ]
    )

    expected_cloud_hard = np.array(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [5, 4, 4], [7, 7, 8], [7, 7, 8]]
    )

    # expend to batch_size = 2
    point_cloud = np.stack([point_cloud, point_cloud * 3], axis=0)
    query_cloud = np.stack([query_cloud, query_cloud * 3], axis=0)
    expected_cloud_soft = np.stack(
        [expected_cloud_soft, expected_cloud_soft * 3], axis=0
    )
    expected_cloud_hard = np.stack(
        [expected_cloud_hard, expected_cloud_hard * 3], axis=0
    )

    point_cloud_pl = tf.convert_to_tensor(point_cloud, dtype=tf.float32)
    query_cloud_pl = tf.convert_to_tensor(query_cloud, dtype=tf.float32)

    soft_projected_points, soft_projection_weights, dist = projector(
        point_cloud_pl, query_cloud_pl
    )
    hard_projected_points, hard_projection_weights, _ = projector(
        point_cloud_pl, query_cloud_pl, hard=True
    )

    soft_projected_points = soft_projected_points.numpy()
    soft_projection_weights = soft_projection_weights.numpy()
    hard_projected_points = hard_projected_points.numpy()
    hard_projection_weights = hard_projection_weights.numpy()

    expected_cloud_soft = expected_cloud_soft.squeeze()
    soft_projected_points = soft_projected_points.squeeze()
    soft_projection_weights = soft_projection_weights.squeeze()
    hard_projected_points = hard_projected_points.squeeze()
    hard_projection_weights = hard_projection_weights.squeeze()

    print("soft_projection_weights:")
    print(soft_projection_weights)

    mse = np.mean(np.sum((soft_projected_points - expected_cloud_soft) ** 2, axis=1))
    print("mean soft error:")
    print(mse)

    mse = np.mean(np.sum((hard_projected_points - expected_cloud_hard) ** 2, axis=1))
    print("mean hard error:")
    print(mse)
