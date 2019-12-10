"""
Created on November 1st, 2019

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
from reconstruction.src.samplers import sampler_with_convs_and_symmetry_and_fc
from reconstruction.src.autoencoder import Configuration as Conf
from reconstruction.src.samplenet_progressive_pointnet_ae import (
    SampleNetProgressivePointNetAE,
)

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
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate [default: 0.0005]')
parser.add_argument('--training_epochs', type=int, default=400, help='Number of training epochs [default: 400]')
parser.add_argument('--restore_ae', type=bool, default=True, help='Restore a trained autoencoder model [default: True]')
parser.add_argument('--ae_restore_epoch', type=int, default=500, help='Epoch to restore a trained autoencoder model [default: 500]')
parser.add_argument('--fixed_ae', type=bool, default=True, help='Fixed autoencoder model [default: True]')
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--ae_folder', type=str, default='log/autoencoder', help='Folder for loading a trained autoencoder model [default: log/autoencoder]')

# sampler arguments
parser.add_argument('--n_sample_points', type=int, default=64, help='Number of sample points (for evluation) [default: 64]')
parser.add_argument('--alpha', type=float, default=0.01, help='Sampling regularization loss weight [default: 0.01]')
parser.add_argument('--train_folder', type=str, default='log/SampleNetProgressive', help='Folder for saving data form the training [default: log/SampleNetProgressive]')

# projection arguments
parser.add_argument('--projection_group_size', type=int, default=16, help='Neighborhood size in Soft Projection [default: 16]')
parser.add_argument('--lmbda', type=float, default=0.0001, help='Temperature regularization loss weight [default: 0.0001]')
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

# load autoencoder configuration
ae_dir = osp.join(top_out_dir, flags.ae_folder)
conf = Conf.load(osp.join(ae_dir, "configuration"))

# update autoencoder configuration
conf.ae_dir = ae_dir
conf.ae_name = "autoencoder"
conf.restore_ae = flags.restore_ae
conf.ae_restore_epoch = flags.ae_restore_epoch
conf.fixed_ae = flags.fixed_ae
if conf.fixed_ae:
    conf.encoder_args[
        "b_norm_decay"
    ] = 1.0  # for avoiding the update of batch normalization moving_mean and moving_variance parameters
    conf.decoder_args[
        "b_norm_decay"
    ] = 1.0  # for avoiding the update of batch normalization moving_mean and moving_variance parameters
    conf.decoder_args[
        "b_norm_decay_finish"
    ] = 1.0  # for avoiding the update of batch normalization moving_mean and moving_variance parameters

conf.use_batch_size_for_place_holder = True

# sampler configuration
conf.experiment_name = "sampler"
conf.pc_size = [2 ** i for i in range(4, 12)]  # Different sample sizes (for training)
conf.n_samp = [
    flags.n_sample_points,
    3,
]  # Dimensionality of sampled points (for evaluation)
conf.sampler = sampler_with_convs_and_symmetry_and_fc
conf.alpha = flags.alpha
conf.learning_rate = flags.learning_rate
conf.training_epochs = flags.training_epochs

# projection configuration
conf.lmbda = flags.lmbda
conf.projection_group_size = flags.projection_group_size
conf.hard_projection = 0

train_dir = create_dir(osp.join(top_out_dir, flags.train_folder))
conf.train_dir = train_dir

conf.save(osp.join(train_dir, "configuration"))

# build Sampler and AE Model
reset_tf_graph()
ae = SampleNetProgressivePointNetAE(conf.experiment_name, conf)

# train the sampler (save output to train_stats.txt)
buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, "train_stats.txt"), "a", buf_size)
train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)
fout.close()
