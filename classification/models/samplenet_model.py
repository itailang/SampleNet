from builtins import range
import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import tf_util
from structural_losses.tf_nndistance import nn_distance
from structural_losses.tf_approxmatch import approx_match


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(
    point_cloud, is_training, num_output_points, bottleneck_size, bn_decay=None
):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(
        input_image,
        64,
        [1, 3],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv1",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        64,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv2",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        64,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv3",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        128,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv4",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        bottleneck_size,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv5",
        bn_decay=bn_decay,
    )

    net = tf_util.max_pool2d(net, [num_point, 1], padding="VALID", scope="maxpool")

    net = tf.reshape(net, [batch_size, -1])

    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope="fc11b", bn_decay=bn_decay
    )
    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope="fc12b", bn_decay=bn_decay
    )
    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope="fc13b", bn_decay=bn_decay
    )
    net = tf_util.fully_connected(
        net,
        3 * num_output_points,
        bn=True,
        is_training=is_training,
        scope="fc14b",
        bn_decay=bn_decay,
        activation_fn=None,
    )

    out_point_cloud = tf.reshape(net, [batch_size, -1, 3])

    return out_point_cloud


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def fps_from_given_pc(pts, k, given_pc):
    farthest_pts = np.zeros((k, 3))
    t = np.size(given_pc) // 3
    farthest_pts[0:t] = given_pc

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    for i in range(t, k):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def unique(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def nn_matching(full_pc, idx, k, complete_fps=True):
    batch_size = np.size(full_pc, 0)
    out_pc = np.zeros((full_pc.shape[0], k, 3))
    for ii in range(0, batch_size):
        best_idx = idx[ii]
        if complete_fps:
            best_idx = unique(best_idx)
            out_pc[ii] = fps_from_given_pc(full_pc[ii], k, full_pc[ii][best_idx])
        else:
            out_pc[ii] = full_pc[ii][best_idx]
    return out_pc[:, 0:k, :]


def emd_matching(full_pc, gen_pc, sess):
    batch_size = np.size(full_pc, 0)
    k = np.size(gen_pc, 1)
    out_pc = np.zeros_like(gen_pc)

    match_mat_tensor = approx_match(
        tf.convert_to_tensor(full_pc), tf.convert_to_tensor(gen_pc)
    )
    pc1_match_idx_tensor = tf.cast(tf.argmax(match_mat_tensor, axis=2), dtype=tf.int32)

    pc1_match_idx = pc1_match_idx_tensor.eval(session=sess)

    for ii in range(0, batch_size):
        best_idx = unique(pc1_match_idx[ii])
        out_pc[ii] = fps_from_given_pc(full_pc[ii], k, full_pc[ii][best_idx])

    return out_pc


def get_nn_indices(ref_pc, samp_pc):
    _, idx, _, _ = nn_distance(samp_pc, ref_pc)
    return idx


def get_simplification_loss(ref_pc, samp_pc, pc_size, gamma=1, delta=0):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(samp_pc, ref_pc)
    max_cost = tf.reduce_max(cost_p1_p2, axis=1)
    max_cost = tf.reduce_mean(max_cost)
    cost_p1_p2 = tf.reduce_mean(cost_p1_p2)
    cost_p2_p1 = tf.reduce_mean(cost_p2_p1)
    loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1

    tf.summary.scalar("cost_p1_p2", cost_p1_p2)
    tf.summary.scalar("cost_p2_p1", cost_p2_p1)
    tf.summary.scalar("max_cost", max_cost)

    return loss
