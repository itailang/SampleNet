from __future__ import print_function
from builtins import str
from builtins import range
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import sys

from soft_projection import SoftProjection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
import provider
import data_prep_util

# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--sampler_model', default='samplenet_model', help='Sampler model name [default: samplenet_model]')
parser.add_argument('--sampler_model_path', default='log/SampleNetProgressive/model.ckpt')
parser.add_argument('--use_restore_epoch', action='store_true', help='Add restore_epoch to sampler_model_path')
parser.add_argument('--restore_epoch', type=int, default=500, help='Epoch for model restore [default: 500]')
parser.add_argument('--num_in_points', type=int, default=1024, help='Number of input points [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size during inference [default: 1]')
parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck size [default: 128]')
parser.add_argument('--num_simplified_points', type=int, default=1024, help='Number of simplified points [default: 1024]')
parser.add_argument('--num_out_points', type=int, default=1024, help='Number of output points [default: 1024]')
parser.add_argument('--dump_dir', default='log/SampleNetProgressive', help='dump folder path')

# projection arguments
parser.add_argument("--projection_group_size", type=int, default=7, help='Neighborhood size in Soft Projection [default: 7]')
FLAGS = parser.parse_args()
# fmt: on

GPU_INDEX = FLAGS.gpu
SAMPLER_MODEL = importlib.import_module(FLAGS.sampler_model)  # import network module
SAMPLER_MODEL_PATH = FLAGS.sampler_model_path
USE_RESTORE_EPOCH = FLAGS.use_restore_epoch
RESTORE_EPOCH = FLAGS.restore_epoch
NUM_IN_POINTS = FLAGS.num_in_points
BATCH_SIZE = FLAGS.batch_size
BOTTLENECK_SIZE = FLAGS.bottleneck_size
NUM_SIMPLIFIED_POINTS = FLAGS.num_simplified_points
NUM_OUT_POINTS = FLAGS.num_out_points
DUMP_DIR = FLAGS.dump_dir

# projection configuration
PROJECTION_GROUP_SIZE = FLAGS.projection_group_size

if USE_RESTORE_EPOCH:
    SAMPLER_MODEL_PATH += "-" + str(int(RESTORE_EPOCH))

model_path, model_file_name = os.path.split(SAMPLER_MODEL_PATH)
OUT_DATA_PATH = model_path

if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, "log_evaluate.txt"), "w")
LOG_FOUT.write(str(FLAGS) + "\n")
data_dtype = "float32"
label_dtype = "uint8"
NUM_CLASSES = 40
SHAPE_NAMES = [
    line.rstrip()
    for line in open(
        os.path.join(BASE_DIR, "data/modelnet40_ply_hdf5_2048/shape_names.txt")
    )
]
HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, "data/modelnet40_ply_hdf5_2048/train_files.txt")
)
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, "data/modelnet40_ply_hdf5_2048/test_files.txt")
)

INFER_FILES = TEST_FILES


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = SAMPLER_MODEL.placeholder_inputs(
            BATCH_SIZE, NUM_IN_POINTS
        )
        is_training_pl = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope("sampler"):
            simplified_points = SAMPLER_MODEL.get_model(
                pointclouds_pl, is_training_pl, NUM_SIMPLIFIED_POINTS, BOTTLENECK_SIZE
            )

            projector = SoftProjection(PROJECTION_GROUP_SIZE)
            hard_projected_points, _, _ = projector.project(
                pointclouds_pl, simplified_points, hard=True
            )
            soft_projected_points, _, _ = projector.project(
                pointclouds_pl, simplified_points, hard=False
            )

        idx = SAMPLER_MODEL.get_nn_indices(pointclouds_pl, simplified_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, SAMPLER_MODEL_PATH)
    log_string("Model restored.")

    ops = {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "simplified_points": simplified_points,
        "soft_projected_points": soft_projected_points,
        "hard_projected_points": hard_projected_points,
        "idx": idx,
    }

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    total_seen = 0
    for fn in range(len(INFER_FILES)):
        log_string(
            "---- file number "
            + str(fn + 1)
            + " out of "
            + str(len(INFER_FILES))
            + " files ----"
        )
        current_data, current_label = provider.loadDataFile(INFER_FILES[fn])
        current_data = current_data[:, 0:NUM_IN_POINTS, :]
        out_data_simplified = np.zeros(
            (current_data.shape[0], NUM_SIMPLIFIED_POINTS, current_data.shape[2])
        )
        out_data_soft_projected = np.zeros(
            (current_data.shape[0], NUM_SIMPLIFIED_POINTS, current_data.shape[2])
        )
        out_data_hard_projected = np.zeros(
            (current_data.shape[0], NUM_SIMPLIFIED_POINTS, current_data.shape[2])
        )

        out_data_sampled = np.zeros(
            (current_data.shape[0], NUM_OUT_POINTS, current_data.shape[2])
        )

        current_label_orig = current_label
        current_label = np.squeeze(current_label)
        print(current_data.shape)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        for batch_idx in range(num_batches):
            print(str(batch_idx) + "/" + str(num_batches - 1))
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {
                ops["pointclouds_pl"]: current_data[start_idx:end_idx],
                ops["labels_pl"]: current_label[start_idx:end_idx],
                ops["is_training_pl"]: is_training,
            }
            (
                simplified_points,
                soft_projected_points,
                hard_projected_points,
                nn_indices,
            ) = sess.run(
                [
                    ops["simplified_points"],
                    ops["soft_projected_points"],
                    ops["hard_projected_points"],
                    ops["idx"],
                ],
                feed_dict=feed_dict,
            )

            out_data_simplified[start_idx:end_idx, :, :] = simplified_points
            out_data_soft_projected[start_idx:end_idx, :, :] = soft_projected_points
            out_data_hard_projected[start_idx:end_idx, :, :] = hard_projected_points

            outcloud_sampled = SAMPLER_MODEL.nn_matching(
                current_data[start_idx:end_idx], nn_indices, NUM_OUT_POINTS
            )
            out_data_sampled[start_idx:end_idx, :, :] = outcloud_sampled[
                :, 0:NUM_OUT_POINTS, :
            ]

            total_seen += BATCH_SIZE

        file_name = os.path.split(INFER_FILES[fn])
        if not os.path.exists(OUT_DATA_PATH + "/simplified/"):
            os.makedirs(OUT_DATA_PATH + "/simplified/")
        data_prep_util.save_h5(
            OUT_DATA_PATH + "/simplified/" + file_name[1],
            out_data_simplified,
            current_label_orig,
            data_dtype,
            label_dtype,
        )

        if not os.path.exists(OUT_DATA_PATH + "/soft_projected/"):
            os.makedirs(OUT_DATA_PATH + "/soft_projected/")
        data_prep_util.save_h5(
            OUT_DATA_PATH + "/soft_projected/" + file_name[1],
            out_data_soft_projected,
            current_label_orig,
            data_dtype,
            label_dtype,
        )

        if not os.path.exists(OUT_DATA_PATH + "/hard_projected/"):
            os.makedirs(OUT_DATA_PATH + "/hard_projected/")
        data_prep_util.save_h5(
            OUT_DATA_PATH + "/hard_projected/" + file_name[1],
            out_data_hard_projected,
            current_label_orig,
            data_dtype,
            label_dtype,
        )

        if not os.path.exists(OUT_DATA_PATH + "/sampled/"):
            os.makedirs(OUT_DATA_PATH + "/sampled/")
        data_prep_util.save_h5(
            OUT_DATA_PATH + "/sampled/" + file_name[1],
            out_data_sampled,
            current_label_orig,
            data_dtype,
            label_dtype,
        )


if __name__ == "__main__":
    evaluate()
    LOG_FOUT.close()
