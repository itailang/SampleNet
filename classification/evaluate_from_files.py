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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
import provider

# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--classifier_model', default='pointnet_cls_basic', help='Classifier model name (pointnet_cls or pointnet_cls_basic) [default: pointnet_cls_basic]')
parser.add_argument('--classifier_model_path', default='log/PointNetVanilla1024/model.ckpt', help='Path to model.ckpt file of a classifier')
parser.add_argument('--data_path', default='log/SampleNetProgressive/sampled', help='path to a folder containing .h5 files with sampled point clouds')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size during evaluation [default: 1]')
parser.add_argument('--dense_eval', type=int, default=0, help='0 - evaluate only for sample sizes that are powers of 2, 1 - evaluate for every sample size [default: 0]')
parser.add_argument('--min_pc_size', type=int, default=8, help='Minimal sample size to evaluate on [default: 8]')
parser.add_argument('--max_pc_size', type=int, default=1024, help='Maximal sample size to evaluate on [default: 1024]')
parser.add_argument('--dump_dir', default='log/SampleNetProgressive/eval/sampled', help='dump folder path')
FLAGS = parser.parse_args()
# fmt: on

GPU_INDEX = FLAGS.gpu
CLASSIFIER_MODEL = importlib.import_module(
    FLAGS.classifier_model
)  # import network module
CLASSIFIER_MODEL_PATH = FLAGS.classifier_model_path
DATA_PATH = FLAGS.data_path
BATCH_SIZE = FLAGS.batch_size
DENSE_EVAL = FLAGS.dense_eval
MIN_PC_SIZE = FLAGS.min_pc_size
MAX_PC_SIZE = FLAGS.max_pc_size

DUMP_DIR = FLAGS.dump_dir
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


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def evaluate(pc_size):
    with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = CLASSIFIER_MODEL.placeholder_inputs(
            BATCH_SIZE, pc_size
        )
        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, end_points = CLASSIFIER_MODEL.get_model(pointclouds_pl, is_training_pl)
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
    saver.restore(sess, CLASSIFIER_MODEL_PATH)

    ops = {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "pred": pred,
        "loss": loss,
    }

    eval_one_epoch(sess, ops, pc_size)


def eval_one_epoch(sess, ops, pc_size, topk=1):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, "pred_label.txt"), "w")
    for fn in range(len(TEST_FILES)):
        filename = os.path.split(TEST_FILES[fn])
        file_path = DATA_PATH + "/" + filename[1]
        current_data, current_label = provider.loadDataFile(file_path)
        current_data = current_data[:, 0:pc_size, :]
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
            feed_dict = {
                ops["pointclouds_pl"]: current_data[start_idx:end_idx, :, :],
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
            batch_loss_sum += loss_val * cur_batch_size / float(1)
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

    log_string(
        "sample size: %d, eval accuracy: %f"
        % (pc_size, total_correct / float(total_seen))
    )


if __name__ == "__main__":
    if DENSE_EVAL:
        pc_sizes = reversed(list(range(MIN_PC_SIZE, MAX_PC_SIZE + 1)))
    else:
        min_size = 2 ** (MIN_PC_SIZE - 1).bit_length()
        pc_sizes = np.array([], "int32")
        j = MIN_PC_SIZE
        while j <= MAX_PC_SIZE:
            pc_sizes = np.append(pc_sizes, int(j))
            j = j * 2

        pc_sizes = pc_sizes[::-1]

    for pc_size in pc_sizes:
        evaluate(pc_size)
        tf.reset_default_graph()
    LOG_FOUT.close()
