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

# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--classifier_model', default='pointnet_cls', help='Classifier model name [pointnet_cls/pointnet_cls_basic] [default:pointnet_cls]')
parser.add_argument('--sampler_model', default='samplenet_model', help='Sampler model name: [default: samplenet_model]')
parser.add_argument('--sampler_model_path', default='log/SampleNet32/model.ckpt', help='Path to model.ckpt file of SampleNet')
parser.add_argument('--use_restore_epoch', action='store_true', help='Add restore_epoch to sampler_model_path')
parser.add_argument('--restore_epoch', type=int, default=500, help='Epoch for model restore [default: 500]')
parser.add_argument('--infer_set', default='test', help='Data set for inference (train or test) [default: test]')
parser.add_argument('--num_in_points', type=int, default=1024, help='Number of input Points [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during evaluation [default: 1]')
parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck size [default: 128]')
parser.add_argument('--match_output', type=int, default=1, help='Matching flag: 1 - match, 0 - do not match [default:1]')
parser.add_argument('--dump_dir', default='log/SampleNet32/eval', help='dump folder path')
parser.add_argument('--num_out_points', type=int, default=32, help='Number of output points [2,4,...,1024] [default: 32]')

# projection arguments
parser.add_argument("--projection_group_size", type=int, default=7, help='Neighborhood size in Soft Projection [default: 7]')
FLAGS = parser.parse_args()
# fmt: on

GPU_INDEX = FLAGS.gpu
CLASSIFIER_MODEL = importlib.import_module(
    FLAGS.classifier_model
)  # import network module
SAMPLER_MODEL = importlib.import_module(FLAGS.sampler_model)  # import network module
SAMPLER_MODEL_PATH = FLAGS.sampler_model_path
USE_RESTORE_EPOCH = FLAGS.use_restore_epoch
INFER_SET = FLAGS.infer_set
RESTORE_EPOCH = FLAGS.restore_epoch
NUM_IN_POINTS = FLAGS.num_in_points
BATCH_SIZE = FLAGS.batch_size
BOTTLENECK_SIZE = FLAGS.bottleneck_size
MATCH_OUTPUT = FLAGS.match_output
DUMP_DIR = FLAGS.dump_dir
NUM_OUT_POINTS = FLAGS.num_out_points

# projection configuration
PROJECTION_GROUP_SIZE = FLAGS.projection_group_size

if USE_RESTORE_EPOCH:
    SAMPLER_MODEL_PATH += "-" + str(int(RESTORE_EPOCH))

if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, "log_evaluate.txt"), "w")
LOG_FOUT.write(str(FLAGS) + "\n")

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

if INFER_SET == "train":
    INFER_FILES = TRAIN_FILES
else:
    INFER_FILES = TEST_FILES


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = CLASSIFIER_MODEL.placeholder_inputs(
            BATCH_SIZE, NUM_IN_POINTS
        )
        is_training_pl = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope("sampler"):
            simplified_points = SAMPLER_MODEL.get_model(
                pointclouds_pl, is_training_pl, NUM_OUT_POINTS, BOTTLENECK_SIZE
            )

            projector = SoftProjection(PROJECTION_GROUP_SIZE)
            hard_projected_points, _, _ = projector.project(
                pointclouds_pl, simplified_points, hard=True
            )
            soft_projected_points, _, _ = projector.project(
                pointclouds_pl, simplified_points, hard=False
            )

        idx = SAMPLER_MODEL.get_nn_indices(pointclouds_pl, simplified_points)

        outcloud = simplified_points
        pred, end_points = CLASSIFIER_MODEL.get_model(outcloud, is_training_pl)

        loss = CLASSIFIER_MODEL.get_loss(pred, labels_pl, end_points)

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
        "pred": pred,
        "loss": loss,
        "simplified_points": simplified_points,
        "soft_projected_points": soft_projected_points,
        "hard_projected_points": hard_projected_points,
        "idx": idx,
        "outcloud": outcloud,
    }

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_unique_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, "pred_label.txt"), "w")
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

        current_label = np.squeeze(current_label)
        print(current_data.shape)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        for batch_idx in range(num_batches):

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # Aggregating BEG
            batch_loss_sum = 0  # sum of losses for the batch
            batch_pred_sum = np.zeros(
                (cur_batch_size, NUM_CLASSES)
            )  # score for classes
            batch_pred_classes = np.zeros(
                (cur_batch_size, NUM_CLASSES)
            )  # 0/1 for classes
            curr_data = current_data[start_idx:end_idx, :, :]
            feed_dict = {
                ops["pointclouds_pl"]: curr_data,
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

            sampled_points = SAMPLER_MODEL.nn_matching(
                curr_data, nn_indices, NUM_OUT_POINTS
            )

            if MATCH_OUTPUT:
                outcloud = sampled_points
            else:
                outcloud = simplified_points

            for ii in range(0, BATCH_SIZE):
                num_unique_idx += np.size(np.unique(nn_indices[ii]))

            feed_dict = {
                ops["pointclouds_pl"]: curr_data,
                ops["outcloud"]: outcloud,
                ops["labels_pl"]: current_label[start_idx:end_idx],
                ops["is_training_pl"]: is_training,
            }
            loss_val, pred_val = sess.run(
                [ops["loss"], ops["pred"]], feed_dict=feed_dict
            )

            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += loss_val * cur_batch_size

            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END

            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += pred_val[i - start_idx] == l
                fout.write("%d, %d\n" % (pred_val[i - start_idx], l))

    log_string("eval mean loss: %f" % (loss_sum / float(total_seen)))
    log_string("eval accuracy: %f" % (total_correct / float(total_seen)))
    log_string(
        "eval avg class acc: %f"
        % (
            np.mean(
                np.array(total_correct_class)
                / np.array(total_seen_class, dtype=np.float)
            )
        )
    )
    log_string("total_seen: %f" % (total_seen))

    class_accuracies = np.array(total_correct_class) / np.array(
        total_seen_class, dtype=np.float
    )
    for i, name in enumerate(SHAPE_NAMES):
        log_string("%10s:\t%0.3f" % (name, class_accuracies[i]))


if __name__ == "__main__":
    evaluate()
    LOG_FOUT.close()
