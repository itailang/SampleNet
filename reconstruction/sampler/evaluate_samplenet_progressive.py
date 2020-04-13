"""
Created on November 21st, 2019

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
from reconstruction.src.samplenet_progressive_pointnet_ae import (
    SampleNetProgressivePointNetAE,
)

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
parser.add_argument('--n_sample_points', type=int, default=64, help='Number of sample points (for evluation) [default: 64]')
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--train_folder', type=str, default='log/SampleNetProgressive', help='Folder for loading data form the training [default: log/SampleNetProgressive]')
parser.add_argument('--restore_epoch', type=int, default=400, help='Restore epoch of a saved sampler model [default: 400]')
parser.add_argument('--hard_projection', type=int, default=1, help='1: Replace soft projected points with first neatesr neighbor, 0: do not replace [default: 1]')
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

# load configuration
train_dir = osp.join(top_out_dir, flags.train_folder)
conf = Conf.load(osp.join(train_dir, "configuration"))

# update configuration
conf.use_batch_size_for_place_holder = False
conf.encoder_args["return_layer_before_symmetry"] = False
conf.hard_projection = flags.hard_projection
conf.pc_size = [flags.n_sample_points]
conf.n_samp = [flags.n_sample_points, 3]

# reload a saved model
reset_tf_graph()
ae = SampleNetProgressivePointNetAE(conf.experiment_name, conf)
ae.restore_model(train_dir, epoch=flags.restore_epoch, verbose=True)

n_input_points = conf.n_input[0]
n_sample_points = conf.n_samp[0]

# create evaluation dir
eval_dir = create_dir(osp.join(train_dir, "eval"))

# sample point clouds (if not exist)
file_name = (
    "_".join(["sampled_pc", "test_set", flags.object_class, "%04d" % n_input_points])
    + ".npy"
)
file_path = osp.join(eval_dir, file_name)
if not osp.exists(file_path):
    # sample point clouds
    sampled_pc, sample_idx = ae.get_samples(
        pc_data_test.point_clouds
    )  # Complete point cloud, ordered by SampleNetProgressive

    # save sampled point clouds
    np.save(file_path, sampled_pc)

    # save sample index
    file_name = (
        "_".join(
            ["sample_idx", "test_set", flags.object_class, "%04d" % n_input_points]
        )
        + ".npy"
    )
    file_path = osp.join(eval_dir, file_name)
    np.save(file_path, sample_idx)
else:
    # load sampled point clouds
    sampled_pc = np.load(file_path)

# reconstruct point clouds from sampled point clouds
reconstructions = ae.get_reconstructions_from_sampled(sampled_pc)

# save reconstructions
file_name = (
    "_".join(
        ["reconstructions", "test_set", flags.object_class, "%04d" % n_sample_points]
    )
    + ".npy"
)
file_path = osp.join(eval_dir, file_name)
np.save(file_path, reconstructions)

# compute loss per sampled point cloud
ae_loss_per_pc = ae.get_loss_ae_per_pc(pc_data_test.point_clouds, sampled_pc)

# save loss per pc
file_name = (
    "_".join(["ae_loss", "test_set", flags.object_class, "%04d" % n_sample_points])
    + ".npy"
)
file_path = osp.join(eval_dir, file_name)
np.save(file_path, ae_loss_per_pc)

# save log file
log_file_name = (
    "_".join(["eval_stats", "test_set", flags.object_class, "%04d" % n_sample_points])
    + ".txt"
)
log_file = open(osp.join(eval_dir, log_file_name), "w", 1)
log_file.write("Evaluation flags: %s\n" % flags)
log_file.write("Mean ae loss: %.9f\n" % ae_loss_per_pc.mean())

# compute normalized reconstruction error
file_name_ref = "_".join(["ae_loss", "test_set", flags.object_class]) + ".npy"
file_path_ref = osp.join(conf.ae_dir, "eval", file_name_ref)
if osp.exists(file_path_ref):
    ae_loss_per_pc_ref = np.load(file_path_ref)

    nre_per_pc = np.divide(ae_loss_per_pc, ae_loss_per_pc_ref)
    log_file.write("Normalized reconstruction error: %.9f\n" % nre_per_pc.mean())
log_file.close()

# use any plotting mechanism, such as matplotlib, to visualize the results
if flags.visualize_results:
    i = 0
    plot_3d_point_cloud(
        pc_data_test.point_clouds[i],
        in_u_sphere=True,
        title="Complete input point cloud",
    )
    plot_3d_point_cloud(
        sampled_pc[i][:n_sample_points], in_u_sphere=True, title="SampleNetProgressive sampled points"
    )
    plot_3d_point_cloud(
        reconstructions[i],
        in_u_sphere=True,
        title="Reconstruction from SampleNetProgressive points",
    )
