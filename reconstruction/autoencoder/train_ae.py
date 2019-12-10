"""
Created on September 5th, 2018

@author: itailang
"""
from __future__ import print_function

# import system modules
from builtins import str
from builtins import range
import os.path as osp
import sys
import argparse

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from reconstruction.src.ae_templates import (
    mlp_architecture_ala_iclr_18,
    default_train_params,
)
from reconstruction.src.autoencoder import Configuration as Conf
from reconstruction.src.pointnet_ae import PointNetAE

from reconstruction.src.in_out import (
    snc_category_to_synth_id,
    create_dir,
    PointCloudDataSet,
    load_and_split_all_point_clouds_under_folder,
)

from reconstruction.src.tf_utils import reset_tf_graph

# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--use_fps', type=int, default=0, help='1: FPS sampling before autoencoder, 0: do not sample [default: 0]')
parser.add_argument('--n_sample_points', type=int, default=2048, help='Number of sample points of the input [default: 2048]')
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--train_folder', type=str, default='log/autoencoder', help='Folder for saving data form the training [default: log/autoencoder]')
parser.add_argument('--training_epochs', type=int, default=500, help='Number of training epochs [default: 500]')
flags = parser.parse_args()
# fmt: on

print(("Train flags:", flags))

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

experiment_name = "autoencoder"
n_pc_points = 2048  # Number of points per model
bneck_size = 128  # Bottleneck-AE size
ae_loss = "chamfer"  # Loss to optimize

# load Point-Clouds
syn_id = snc_category_to_synth_id()[class_name[0]]
class_dir = osp.join(top_in_dir, syn_id)
pc_data_train, pc_data_val, _ = load_and_split_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending=".ply", verbose=True
)

for i in range(1, len(class_name)):
    syn_id = snc_category_to_synth_id()[class_name[i]]
    class_dir = osp.join(top_in_dir, syn_id)
    (
        pc_data_train_curr,
        pc_data_val_curr,
        _,
    ) = load_and_split_all_point_clouds_under_folder(
        class_dir, n_threads=8, file_ending=".ply", verbose=True
    )
    pc_data_train.merge(pc_data_train_curr)
    pc_data_val.merge(pc_data_val_curr)

if flags.object_class == "multi":
    pc_data_train.shuffle_data(seed=55)
    pc_data_val.shuffle_data(seed=55)

# load default training parameters
train_params = default_train_params()

encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(
    n_pc_points, bneck_size
)

train_dir = create_dir(osp.join(top_out_dir, flags.train_folder))

conf = Conf(
    n_input=[n_pc_points, 3],
    loss=ae_loss,
    training_epochs=train_params["training_epochs"],
    batch_size=train_params["batch_size"],
    denoising=train_params["denoising"],
    learning_rate=train_params["learning_rate"],
    train_dir=train_dir,
    loss_display_step=train_params["loss_display_step"],
    saver_step=train_params["saver_step"],
    z_rotate=train_params["z_rotate"],
    encoder=encoder,
    decoder=decoder,
    encoder_args=enc_args,
    decoder_args=dec_args,
)
conf.experiment_name = experiment_name
conf.held_out_step = 5  # how often to evaluate/print(out loss on)
# held_out data (if they are provided in ae.train()).
conf.class_name = class_name
conf.use_fps = flags.use_fps
conf.n_sample_points = flags.n_sample_points
conf.n_samp_out = [2048, 3]
conf.training_epochs = flags.training_epochs
conf.save(osp.join(train_dir, "configuration"))

# build AE Model
reset_tf_graph()
ae = PointNetAE(conf.experiment_name, conf)

# train the AE (save output to train_stats.txt)
buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, "train_stats.txt"), "a", buf_size)
train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)
fout.close()
