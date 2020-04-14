"""
Created on September 25th, 2019

@author: itailang
"""
from __future__ import print_function

# import system modules
from builtins import str
from builtins import range
import os.path as osp
import sys
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from reconstruction.src.autoencoder import Configuration as Conf
from reconstruction.src.pointnet_ae import PointNetAE

from reconstruction.src.in_out import (
    snc_category_to_synth_id,
    create_dir,
    load_and_split_all_point_clouds_under_folder,
)

from reconstruction.src.tf_utils import reset_tf_graph
from reconstruction.src.general_utils import plot_3d_point_cloud

# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--use_fps', type=int, default=0, help='1: FPS sampling before autoencoder, 0: do not sample [default: 0]')
parser.add_argument('--n_sample_points', type=int, default=2048, help='Number of sample points of the input [default: 2048]')
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--train_folder', type=str, default='log/autoencoder', help='Folder for saving data form the training [default: log/autoencoder]')
parser.add_argument('--restore_epoch', type=int, default=500, help='Restore epoch of a saved autoencoder model [default: 500]')
parser.add_argument('--visualize_results', action='store_true', help='Visualize results [default: False]')
flags = parser.parse_args()
# fmt: on

print(("Evaluation flags:", flags))

# define basic parameters
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
top_in_dir = osp.join(
    project_dir, "data", "shape_net_core_uniform_samples_2048"
)  # top-dir of where point-clouds are stored.
top_out_dir = osp.join(project_dir)  # use to save Neural-Net check-points etc.

if flags.object_class == "multi":
    class_name = ["chair", "table", "car", "airplane"]
else:
    class_name = [str(flags.object_class)]

# load Point-Clouds
syn_id = snc_category_to_synth_id()[class_name[0]]
class_dir = osp.join(top_in_dir, syn_id)
_, _, pc_data_test = load_and_split_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending=".ply", verbose=True
)

for i in range(1, len(class_name)):
    syn_id = snc_category_to_synth_id()[class_name[i]]
    class_dir = osp.join(top_in_dir, syn_id)
    _, _, pc_data_test_curr = load_and_split_all_point_clouds_under_folder(
        class_dir, n_threads=8, file_ending=".ply", verbose=True
    )
    pc_data_test.merge(pc_data_test_curr)

# load train configuration
train_dir = create_dir(osp.join(top_out_dir, flags.train_folder))
conf = Conf.load(train_dir + "/configuration")
conf.use_fps = flags.use_fps
conf.n_sample_points = flags.n_sample_points

# build AE Model
reset_tf_graph()
ae = PointNetAE(conf.experiment_name, conf)

# reload a saved model
ae.restore_model(train_dir, epoch=flags.restore_epoch, verbose=True)

n_sample_points = flags.n_sample_points

# create evaluation dir
eval_dir = create_dir(osp.join(train_dir, "eval"))

# sample point clouds
if flags.use_fps:
    sampled_pc, sample_idx = ae.get_samples(
        pc_data_test.point_clouds
    )  # FPS sampled points

    # Save sampled point cloud and sample index
    file_name = (
        "_".join(
            [
                "sampled_pc",
                "test_set",
                flags.object_class,
                "fps",
                "%04d" % n_sample_points,
            ]
        )
        + ".npy"
    )
    file_path = osp.join(eval_dir, file_name)
    np.save(file_path, sampled_pc)

    file_name = (
        "_".join(
            [
                "sample_idx",
                "test_set",
                flags.object_class,
                "fps",
                "%04d" % n_sample_points,
            ]
        )
        + ".npy"
    )
    file_path = osp.join(eval_dir, file_name)
    np.save(file_path, sample_idx)

# reconstruct point clouds
reconstructions = ae.get_reconstructions(pc_data_test.point_clouds)

# save reconstructions
file_name_wo_fps = (
    "_".join(["reconstructions", "test_set", flags.object_class]) + ".npy"
)
if flags.use_fps:
    file_name = (
        "_".join(
            [
                "reconstructions",
                "test_set",
                flags.object_class,
                "fps",
                "%04d" % n_sample_points,
            ]
        )
        + ".npy"
    )
else:
    file_name = file_name_wo_fps
file_path = osp.join(eval_dir, file_name)
np.save(file_path, reconstructions)

# compute loss per point cloud
loss_per_pc = ae.get_loss_per_pc(pc_data_test.point_clouds)

# save loss per point cloud
file_name_wo_fps = "_".join(["ae_loss", "test_set", flags.object_class]) + ".npy"
if flags.use_fps:
    file_name = (
        "_".join(
            ["ae_loss", "test_set", flags.object_class, "fps", "%04d" % n_sample_points]
        )
        + ".npy"
    )
else:
    file_name = file_name_wo_fps

file_path = osp.join(eval_dir, file_name)
np.save(file_path, loss_per_pc)

# save log file
log_file_name_wo_fps = "_".join(["eval_stats", "test_set", flags.object_class]) + ".txt"
if flags.use_fps:
    log_file_name = (
        "_".join(
            [
                "eval_stats",
                "test_set",
                flags.object_class,
                "fps",
                "%04d" % n_sample_points,
            ]
        )
        + ".txt"
    )
else:
    log_file_name = log_file_name_wo_fps

log_file = open(osp.join(eval_dir, log_file_name), "w", 1)
log_file.write("Mean ae loss: %.9f\n" % loss_per_pc.mean())

# compute normalized reconstruction error
file_path_ref = osp.join(eval_dir, file_name_wo_fps)
if osp.exists(file_path_ref):
    loss_per_pc_ref = np.load(file_path_ref)

    nre_per_pc = np.divide(loss_per_pc, loss_per_pc_ref)
    log_file.write("Normalized reconstruction error: %.3f\n" % nre_per_pc.mean())
log_file.close()

# use any plotting mechanism, such as matplotlib, to visualize the results
if flags.visualize_results:
    i = 0
    plot_3d_point_cloud(
        pc_data_test.point_clouds[i],
        in_u_sphere=True,
        title="Complete input point cloud",
    )
    if flags.use_fps:
        plot_3d_point_cloud(
            sampled_pc[i],
            in_u_sphere=True,
            title="FPS sampled points"
        )
    reconstruction_title = "Reconstruction from FPS points" if flags.use_fps else "Reconstruction from complete input"
    plot_3d_point_cloud(
        reconstructions[i], in_u_sphere=True, title=reconstruction_title
    )
