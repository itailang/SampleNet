"""
Created on February 2, 2017

@author: optas

Edited by itailang
"""
from __future__ import print_function

from builtins import next
from builtins import str
from builtins import range
from builtins import object
import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from tflearn import is_training

from .in_out import create_dir, pickle_data, unpickle_data
from .general_utils import apply_augmentations, iterate_in_chunks
from .neural_net import Neural_Net, MODEL_SAVER_ID


class Configuration(object):
    def __init__(
        self,
        n_input,
        encoder,
        decoder,
        encoder_args={},
        decoder_args={},
        training_epochs=200,
        batch_size=10,
        learning_rate=0.001,
        denoising=False,
        saver_step=None,
        train_dir=None,
        z_rotate=False,
        loss="chamfer",
        gauss_augment=None,
        saver_max_to_keep=None,
        loss_display_step=1,
        debug=False,
        n_z=None,
        n_output=None,
        latent_vs_recon=1.0,
        consistent_io=None,
    ):

        # Parameters for any AE
        self.n_input = n_input
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug

        # Used in VAE
        self.latent_vs_recon = np.array([latent_vs_recon], dtype=np.float32)[0]
        self.n_z = n_z

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

        self.consistent_io = consistent_io

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = list(self.__dict__.keys())
        vals = list(self.__dict__.values())
        index = np.argsort(keys)
        res = ""
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += "%30s: %s\n" % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + ".pickle", self)
        with open(file_name + ".txt", "w") as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return next(unpickle_data(file_name + ".pickle"))


class AutoEncoder(Neural_Net):
    """Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.
    """

    def __init__(self, name, graph, configuration):
        Neural_Net.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        in_shape = [None] + self.n_input
        out_shape = [None] + self.n_output

        samp_out_shape = [None] + configuration.n_samp_out

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, out_shape)
            else:
                self.gt = self.x

            # for calculating chamfer distance between a sample out the output and ground truth data
            self.s_out = tf.placeholder(tf.float32, samp_out_shape)

    def partial_fit(self, X, GT=None):
        """Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        """
        is_training(True, session=self.sess)
        try:
            if GT is not None:
                _, loss, recon = self.sess.run(
                    (self.train_step, self.loss, self.x_reconstr),
                    feed_dict={self.x: X, self.gt: GT},
                )
            else:
                _, loss, recon = self.sess.run(
                    (self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X}
                )

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss

    def reconstruct(self, X, GT=None, sort_reconstr=False, compute_loss=True):
        """Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)"""
        if compute_loss:
            loss = self.loss
        else:
            loss = self.no_op

        if sort_reconstr:
            x_reconstr = self.x_reconstr_sorted
        else:
            x_reconstr = self.x_reconstr

        if GT is None:
            return self.sess.run((x_reconstr, loss), feed_dict={self.x: X})
        else:
            return self.sess.run((x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

    def get_nn_distance(self, GT, S_OUT):
        return self.sess.run(
            self.nn_distance, feed_dict={self.s_out: S_OUT, self.gt: GT}
        )

    def get_loss(self, X, GT=None):
        if GT is None:
            feed_dict = {self.x: X}
        else:
            feed_dict = {self.x: X, self.gt: GT}

        return self.sess.run(self.loss, feed_dict=feed_dict)

    def get_loss_per_pc(self, feed_data, orig_data=None):
        feed_data_shape = feed_data.shape
        assert len(feed_data_shape) == 3, "The feed data should have 3 dimensions"

        if orig_data is not None:
            assert (
                feed_data_shape == orig_data.shape
            ), "The feed data and original data should have the same size"
        else:
            orig_data = feed_data

        n_examples = feed_data_shape[0]
        ae_loss = np.zeros(n_examples)
        for i in range(0, n_examples, 1):
            ae_loss[i] = self.get_loss(feed_data[i : i + 1], orig_data[i : i + 1])

        return ae_loss

    def get_sample(self, X):
        """Sample fps points from input data."""
        assert (
            self.configuration.exists_and_is_not_none("use_fps")
            and self.configuration.use_fps
        ), "For getting FPS sampled point clouds, use_fps flag should be on"

        fps_points, fps_idx = self.sess.run((self.s, self.idx), feed_dict={self.x: X})

        return fps_points, fps_idx

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        return self.sess.run(self.z, feed_dict={self.x: X})

    def interpolate(self, x, y, steps):
        """ Interpolate between and x and y input vectors in latent space.
        x, y np.arrays of size (n_points, dim_embedding).
        """
        in_feed = np.vstack((x, y))
        z1, z2 = self.transform(in_feed.reshape([2] + self.n_input))
        all_z = np.zeros((steps + 2, len(z1)))

        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_z[i, :] = (alpha * z2) + ((1.0 - alpha) * z1)

        return self.sess.run((self.x_reconstr), {self.z: all_z})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def train(self, train_data, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in range(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print(
                    (
                        "Epoch:",
                        "%04d" % (epoch),
                        "training time (minutes)=",
                        "{:.4f}".format(duration / 60.0),
                        "loss=",
                        "{:.9f}".format(loss),
                    )
                )
                if log_file is not None:
                    log_file.write(
                        "%04d\t%.9f\t%.4f\n" % (epoch, loss, duration / 60.0)
                    )

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (
                epoch % c.saver_step == 0 or epoch - 1 == 0
            ):
                checkpoint_path = osp.join(c.train_dir, MODEL_SAVER_ID)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none("summary_step") and (
                epoch % c.summary_step == 0 or epoch - 1 == 0
            ):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if (
                held_out_data is not None
                and c.exists_and_is_not_none("held_out_step")
                and (epoch % c.held_out_step == 0)
            ):
                loss, duration = self._single_epoch_train(
                    held_out_data, c, only_fw=True
                )
                print(
                    (
                        "Held Out Data :",
                        "forward time (minutes)=",
                        "{:.4f}".format(duration / 60.0),
                        "loss=",
                        "{:.9f}".format(loss),
                    )
                )
                if log_file is not None:
                    log_file.write(
                        "On Held_Out: %04d\t%.9f\t%.4f\n"
                        % (epoch, loss, duration / 60.0)
                    )
        return stats

    def get_reconstructions(self, pclouds, batch_size=50):
        """ Convenience wrapper of self.reconstruct to get the FPS sampled points for a set of input point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        """

        reconstructions = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            rcon, _ = self.reconstruct(pclouds[b], compute_loss=False)
            reconstructions.append(rcon)
        return np.vstack(reconstructions)

    def get_reconstructions_from_sampled(self, pclouds, batch_size=50):
        """ Get the reconstructions for a set of sampled point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
            batch_size size of point clouds batch
        """
        reconstructions = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            feed_dict = {self.s: pclouds[b]}
            rcon = self.sess.run(self.x_reconstr, feed_dict=feed_dict)
            reconstructions.append(rcon)
        return np.vstack(reconstructions)

    def get_samples(self, pclouds, batch_size=50):
        """ Convenience wrapper of self.get_sample to get the FPS sampled points for a set of input point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        """

        fps_points = []
        fps_idx = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            curr_fps_points, curr_fps_idx = self.get_sample(pclouds[b])
            fps_points.append(curr_fps_points)
            fps_idx.append(curr_fps_idx)
        return np.vstack(fps_points), np.vstack(fps_idx)

    def get_latent_codes(self, pclouds, batch_size=100):
        """ Convenience wrapper of self.transform to get the latent (bottle-neck) codes for a set of input point 
        clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        """
        latent_codes = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            latent_codes.append(self.transform(pclouds[b]))
        return np.vstack(latent_codes)

    def get_nn_distances(self, pclouds, samples, batch_size=100):
        """ Convenience wrapper of self.get_nn_distance to get the chamfer distance between point clouds and samples
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
            samples (N, Ks, 3) numpy array of N point clouds with Ks points each, Ks <= K.
        """
        nn_distances = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            nn_distances.append(self.get_nn_distance(pclouds[b], samples[b]))
        return np.vstack(nn_distances)
