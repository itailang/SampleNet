"""Utility functions for SampleNet that can be shared between PyTorch and Tensorflow implementations"""

import numpy as np
import argparse


def _calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def _fps_from_given_pc(pts, k, given_pc):
    farthest_pts = np.zeros((k, 3))
    t = np.size(given_pc) // 3
    farthest_pts[0:t] = given_pc

    distances = _calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = np.minimum(distances, _calc_distances(farthest_pts[i], pts))

    for i in range(t, k):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, _calc_distances(farthest_pts[i], pts))
    return farthest_pts


def _unique(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def nn_matching(full_pc, idx, k, complete_fps=True):
    batch_size = np.size(full_pc, 0)
    out_pc = np.zeros((full_pc.shape[0], k, 3))
    for ii in range(0, batch_size):
        best_idx = idx[ii]
        if complete_fps:
            best_idx = _unique(best_idx)
            out_pc[ii] = _fps_from_given_pc(full_pc[ii], k, full_pc[ii][best_idx])
        else:
            out_pc[ii] = full_pc[ii][best_idx]
    return out_pc[:, 0:k, :]


# fmt: off
def get_parser():
    parser = argparse.ArgumentParser("SampleNet: Differentiable Point Cloud Sampling")

    parser.add_argument("--skip-projection", action="store_true", help="Do not project points in training")

    parser.add_argument("-in", "--num-in-points", type=int, default=1024, help="Number of input Points [default: 1024]")
    parser.add_argument("-out", "--num-out-points", type=int, default=64, help="Number of output points [2, 1024] [default: 64]")
    parser.add_argument("--bottleneck-size", type=int, default=128, help="bottleneck size [default: 128]")
    parser.add_argument("--alpha", type=float, default=0.01, help="Simplification regularization loss weight [default: 0.01]")
    parser.add_argument("--gamma", type=float, default=1, help="Lb constant regularization loss weight [default: 1]")
    parser.add_argument("--delta", type=float, default=0, help="Lb linear regularization loss weight [default: 0]")

    # projection arguments
    parser.add_argument("-gs", "--projection-group-size", type=int, default=8, help='Neighborhood size in Soft Projection [default: 8]')
    parser.add_argument("--lmbda", type=float, default=0.01, help="Projection regularization loss weight [default: 0.01]")

    return parser
# fmt: on
