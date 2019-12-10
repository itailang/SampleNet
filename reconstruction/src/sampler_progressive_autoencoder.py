"""
Created on September 7th, 2018

@author: itailang
"""
from __future__ import print_function

from builtins import str
from builtins import range
import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from tflearn import is_training

from .in_out import create_dir, pickle_data, unpickle_data
from .general_utils import apply_augmentations, iterate_in_chunks
from .neural_net import Neural_Net, MODEL_SAVER_ID


class SamplerProgressiveAutoEncoder(Neural_Net):
    """Basis class for a Neural Network that implements a Sampler with Auto-Encoder in TensorFlow."""

    def __init__(self, name, graph, configuration):
        Neural_Net.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        if configuration.exists_and_is_not_none("use_batch_size_for_place_holder"):
            if configuration.use_batch_size_for_place_holder:
                bs = configuration.batch_size
            else:
                bs = None
        else:
            bs = None

        in_shape = [bs] + self.n_input
        out_shape = [bs] + self.n_output

        samp_shape = [bs] + configuration.n_samp

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, out_shape)
            else:
                self.gt = self.x

            # for samples interpolation
            self.s1 = tf.placeholder(tf.float32, samp_shape)
            self.s2 = tf.placeholder(tf.float32, samp_shape)

            if self.is_denoising:
                self.pc1 = tf.placeholder(tf.float32, out_shape)
                self.pc2 = tf.placeholder(tf.float32, out_shape)

    def restore_ae_model(self, ae_model_path, ae_name, epoch, verbose=False):
        """Restore all the variables of a saved ae model.
        """
        global_vars = tf.global_variables()
        ae_params = [v for v in global_vars if v.name.startswith(ae_name)]

        saver_ae = tf.train.Saver(var_list=ae_params)
        saver_ae.restore(
            self.sess, osp.join(ae_model_path, MODEL_SAVER_ID + "-" + str(int(epoch)))
        )

        if verbose:
            print("AE Model restored from %s, in epoch %d" % (ae_model_path, epoch))

    def sort(self, X):
        """Sort points by farthest point sampling indices."""
        return self.sess.run(self.x_sorted, feed_dict={self.x: X})

    def sample(self, X):
        """Sample points from input data."""
        generated_points, idx = self.sess.run((self.s, self.idx), feed_dict={self.x: X})
        (
            sampled_points,
            sampled_points_idx,
            n_unique_points,
        ) = self.simple_projection_and_continued_fps(X, generated_points, idx)

        return generated_points, sampled_points, sampled_points_idx, n_unique_points

    def match_samples(self, S1, S2):
        return self.sess.run(self.s1_matched, feed_dict={self.s1: S1, self.s2: S2})

    def get_match_cost(self, PC1, PC2):
        return self.sess.run(self.match_cost, feed_dict={self.pc1: PC1, self.pc2: PC2})

    def get_nn_distance(self, X, S):
        return self.sess.run(self.nn_distance, feed_dict={self.x: X, self.s: S})

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

    def interpolate_samples(self, s1, s2, steps):
        """ Interpolate between and s1 and s1 samples.
        s1, s2 are np.arrays of size (n_samp_points, 3).
        """
        s1_matched = self.match_samples(
            np.expand_dims(s1, axis=0), np.expand_dims(s2, axis=0)
        )
        s1_matched = np.squeeze(s1_matched, axis=0)

        all_s = np.zeros([steps + 2] + [len(s1)] + [3])
        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_s[i, :, :] = (alpha * s2) + ((1.0 - alpha) * s1_matched)

        return all_s, self.sess.run(self.x_reconstr, {self.s: all_s})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def get_loss_ae(self, X, GT=None, S=None):
        if S is not None:
            feed_dict = {self.s: S}
        else:
            feed_dict = {self.x: X}

        if GT is not None:
            feed_dict[self.gt] = GT

        return self.sess.run(self.loss_ae, feed_dict=feed_dict)

    def get_loss_ae_per_pc(self, feed_data, samp_data, orig_data=None):
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
            ae_loss[i] = self.get_loss_ae(
                feed_data[i : i + 1], orig_data[i : i + 1], samp_data[i : i + 1]
            )

        return ae_loss

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

    def get_sorted_data(self, in_data, batch_size):
        n_examples = np.size(in_data, 0)
        b = batch_size

        # sort by FPS indices
        sorted_data = np.zeros(in_data.shape)
        for i in range(0, n_examples, b):
            sorted_data[i : i + b] = self.sort(in_data[i : i + b])

        return sorted_data

    def get_matches_cost(self, pclouds, pclouds_noisy):
        num_example = pclouds.shape[0]
        num_example_noisy = pclouds_noisy.shape[0]
        num_points = pclouds.shape[1]
        matches_cost = np.zeros([num_example, num_example_noisy])
        for i in range(num_example):
            for j in range(num_example_noisy):
                print(
                    "matching example %d/%d to noisy example %d/%d"
                    % (i + 1, num_example, j + 1, num_example_noisy)
                )
                matches_cost[i, j] = (
                    np.squeeze(
                        self.get_match_cost(
                            pclouds[i : i + 1], pclouds_noisy[j : j + 1]
                        ),
                        axis=0,
                    )
                    / num_points
                )

        return matches_cost
