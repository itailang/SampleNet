"""
Created on November 26, 2017

@author: optas

Edited by itailang
"""

from builtins import range
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def rand_rotation_matrix(deflection=1.0, seed=None):
    """Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def get_complementary_points(pcloud, idx):
    dim_num = len(pcloud.shape)
    n = pcloud.shape[dim_num - 2]
    k = idx.shape[dim_num - 2]

    if dim_num == 2:
        comp_idx = get_complementary_idx(idx, n)
        comp_points = pcloud[comp_idx, :]
    else:
        n_example = pcloud.shape[0]
        comp_points = np.zeros([n_example, n - k, pcloud.shape[2]])
        comp_idx = np.zeros([n_example, n - k], dtype=int)

        for i in range(n_example):
            comp_idx[i, :] = get_complementary_idx(idx[i, :], n)
            comp_points[i, :, :] = pcloud[i, comp_idx[i, :], :]

    return comp_points, comp_idx


def get_complementary_idx(idx, n):
    range_n = np.arange(n, dtype=int)
    comp_indicator = np.full(n, True)

    comp_indicator[idx] = False
    comp_idx = range_n[comp_indicator]

    return comp_idx


def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch = batch.copy()

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment["mu"]
        sigma = conf.gauss_augment["sigma"]
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch = batch.dot(r_rotation)
    return batch


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def plot_3d_point_cloud(
    pc,
    show=True,
    show_axis=True,
    in_u_sphere=True,
    marker=".",
    c="b",
    s=8,
    alpha=0.8,
    figsize=(5, 5),
    elev=10,
    azim=240,
    miv=None,
    mav=None,
    squeeze=0.7,
    axis=None,
    title=None,
    *args,
    **kwargs
):
    x, y, z = (pc[:, 0], pc[:, 1], pc[:, 2])

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, c=c, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        miv = -0.5
        mav = 0.5
    else:
        if miv is None:
            miv = squeeze * np.min(
                [np.min(x), np.min(y), np.min(z)]
            )  # Multiply with 'squeeze' to squeeze free-space.
        if mav is None:
            mav = squeeze * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis("off")

    if "c" in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig, miv, mav
