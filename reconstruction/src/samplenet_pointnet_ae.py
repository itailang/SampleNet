"""
Created on September 24th, 2019

@author: itailang
"""
from __future__ import print_function

from builtins import range
import time
import tensorflow as tf
import numpy as np
import os.path as osp

from tflearn import is_training

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from .in_out import create_dir
from .sampler_autoencoder import SamplerAutoEncoder
from .general_utils import apply_augmentations, iterate_in_chunks
from .soft_projection import SoftProjection
from .neural_net import MODEL_SAVER_ID

try:
    from ..external.sampling.tf_sampling import farthest_point_sample, gather_point
except:
    print("Farthest Point Sample cannot be loaded. Please install it first.")

try:
    from ..external.structural_losses.tf_nndistance import nn_distance
    from ..external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print("External Losses (Chamfer-EMD) cannot be loaded. Please install them first.")


class SampleNetPointNetAE(SamplerAutoEncoder):
    """A sampler with soft projection for point-clouds."""

    def __init__(self, sampler_name, configuration, graph=None):
        c = configuration
        self.configuration = c

        SamplerAutoEncoder.__init__(self, sampler_name, graph, configuration)

        with tf.variable_scope(sampler_name):
            n_pc_point = self.x.get_shape().as_list()[1]  # number of input points

            idx_fps = farthest_point_sample(
                n_pc_point, self.x
            )  # (batch_size, n_pc_point)
            self.x_sorted = gather_point(self.x, idx_fps)  # (batch_size, n_pc_point, 3)

            if self.is_denoising:
                self._create_match_cost()

            self.q = c.sampler(self.x, c.n_samp)

            projector = SoftProjection(c.projection_group_size)
            self.projection_sigma = projector.sigma
            self.s, weights, dist = projector(self.x, self.q, c.hard_projection)

        with tf.variable_scope(c.ae_name):
            self.z = c.encoder(self.s, **c.encoder_args)

            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)

            if c.exists_and_is_not_none("close_with_tanh"):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(
                layer, [-1, self.n_output[0], self.n_output[1]]
            )

        with tf.variable_scope(sampler_name):
            self.saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=c.saver_max_to_keep
            )

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, "allow_gpu_growth"):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(
                osp.join(configuration.train_dir, "summaries"), self.graph
            )

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

            if c.restore_ae:
                self.restore_ae_model(
                    c.ae_dir, c.ae_name, c.ae_restore_epoch, verbose=True
                )

    def _create_match_samples(self):
        match = approx_match(self.s1, self.s2)
        s1_match_idx = tf.cast(tf.argmax(match, axis=2), dtype=tf.int32)
        self.s1_matched = gather_point(
            self.s1, s1_match_idx
        )  # self.s1_matched has the shape of self.s2

    def _create_match_cost(self):
        match = approx_match(self.pc1, self.pc2)
        self.match_cost = tf.reduce_mean(match_cost(self.pc1, self.pc2, match))

    def _create_loss(self):
        c = self.configuration

        # reconstruction loss
        if c.loss == "chamfer":
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss_ae = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == "emd":
            match = approx_match(self.x_reconstr, self.gt)
            self.loss_ae = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        # simplification loss
        (
            self.loss_simplification,
            self.dist,
            self.idx,
            self.dist2,
        ) = self._get_simplification_loss(self.x, self.q, c.n_samp[0])

        # temperature loss
        self.loss_projection = self.projection_sigma

        if c.alpha > 0.0:
            self.loss = self.loss_ae + c.alpha * self.loss_simplification
        else:
            self.loss = self.loss_ae

        self.loss += c.lmbda * self.loss_projection

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none("w_reg_alpha"):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += w_reg_alpha * rl

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("loss_ae", self.loss_ae)
        tf.summary.scalar("loss_simplification", self.loss_simplification)
        tf.summary.scalar("loss_projection", self.loss_projection)

    def _get_simplification_loss(self, ref_pc, samp_pc, pc_size):
        cost_p1_p2, idx, cost_p2_p1, _ = nn_distance(samp_pc, ref_pc)
        dist = cost_p1_p2
        dist2 = cost_p2_p1
        max_cost = tf.reduce_max(cost_p1_p2, axis=1)
        max_cost = tf.reduce_mean(max_cost)

        self.nn_distance = tf.reduce_mean(
            cost_p1_p2, axis=1, keep_dims=True
        ) + tf.reduce_mean(cost_p2_p1, axis=1, keep_dims=True)

        cost_p1_p2 = tf.reduce_mean(cost_p1_p2)
        cost_p2_p1 = tf.reduce_mean(cost_p2_p1)

        w = pc_size / 64.0
        if self.is_denoising:
            loss = cost_p1_p2 + max_cost + 2 * w * cost_p2_p1
        else:
            loss = cost_p1_p2 + max_cost + w * cost_p2_p1

        tf.summary.scalar("cost_p1_p2", cost_p1_p2)
        tf.summary.scalar("cost_p2_p1", cost_p2_p1)
        tf.summary.scalar("max_cost", max_cost)

        return loss, dist, idx, dist2

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, "exponential_decay"):
            self.lr = tf.train.exponential_decay(
                c.learning_rate,
                self.epoch,
                c.decay_steps,
                decay_rate=0.5,
                staircase=True,
                name="learning_rate_decay",
            )
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar("learning_rate", self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        train_vars = tf.trainable_variables()
        sampler_vars = [v for v in train_vars if v.name.startswith(c.experiment_name)]

        if c.fixed_ae:
            self.train_step = self.optimizer.minimize(self.loss, var_list=sampler_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def partial_fit(self, X, GT=None, S=None, compute_recon=False):
        """Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        """
        if compute_recon:
            x_reconstr = self.x_reconstr
        else:
            x_reconstr = self.no_op

        is_training(True, session=self.sess)
        try:
            if GT is None:
                feed_dict = {self.x: X}
            else:
                feed_dict = {self.x: X, self.gt: GT}

            if S is not None:
                feed_dict[self.s] = S

            (
                _,
                loss,
                loss_ae,
                loss_simplification,
                loss_projection,
                recon,
            ) = self.sess.run(
                (
                    self.train_step,
                    self.loss,
                    self.loss_ae,
                    self.loss_simplification,
                    self.loss_projection,
                    x_reconstr,
                ),
                feed_dict=feed_dict,
            )

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss, loss_ae, loss_simplification, loss_projection

    def reconstruct(self, X, GT=None, S=None, compute_loss=True):
        """Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)"""
        if compute_loss:
            loss = self.loss
            loss_ae = self.loss_ae
            loss_simplification = self.loss_simplification
            loss_projection = self.loss_projection
        else:
            loss = self.no_op
            loss_ae = self.no_op
            loss_simplification = self.no_op
            loss_projection = self.no_op

        feed_dict = {self.x: X}
        if GT is not None:
            feed_dict[self.gt] = GT

        if S is not None:
            feed_dict[self.s] = S

        return self.sess.run(
            (self.x_reconstr, loss, loss_ae, loss_simplification, loss_projection),
            feed_dict=feed_dict,
        )

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.0
        epoch_loss_ae = 0.0
        epoch_loss_simplification = 0.0
        epoch_loss_projection = 0.0
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in range(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if (
                    batch_i is None
                ):  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                original_data = None
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(
                batch_i, configuration
            )  # This is a new copy of the batch.

            _, loss, loss_ae, loss_simplification, loss_projection = fit(
                batch_i, original_data
            )

            # Compute average loss
            epoch_loss += loss
            epoch_loss_ae += loss_ae
            epoch_loss_simplification += loss_simplification
            epoch_loss_projection += loss_projection
        epoch_loss /= n_batches
        epoch_loss_ae /= n_batches
        epoch_loss_simplification /= n_batches
        epoch_loss_projection /= n_batches
        duration = time.time() - start_time

        if configuration.loss == "emd":
            epoch_loss_ae /= len(train_data.point_clouds[0])

        epoch_loss = (
            epoch_loss_ae
            + configuration.alpha * epoch_loss_simplification
            + configuration.lmbda * epoch_loss_projection
        )

        return (
            epoch_loss,
            epoch_loss_ae,
            epoch_loss_simplification,
            epoch_loss_projection,
            duration,
        )

    def train(self, train_data, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in range(c.training_epochs):
            (
                loss,
                loss_ae,
                loss_simplification,
                loss_projection,
                duration,
            ) = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append(
                (epoch, loss, loss_ae, loss_simplification, loss_projection, duration)
            )

            if epoch % c.loss_display_step == 0:
                print(
                    (
                        "Epoch:",
                        "%04d" % (epoch),
                        "training time (minutes)=",
                        "{:.4f}".format(duration / 60.0),
                        "loss=",
                        "{:.9f}".format(loss),
                        "loss_ae=",
                        "{:.9f}".format(loss_ae),
                        "loss_simplification=",
                        "{:.9f}".format(loss_simplification),
                        "loss_projection=",
                        "{:.9f}".format(loss_projection),
                    )
                )
                if log_file is not None:
                    log_file.write(
                        "%04d\t%.9f\t%.9f\t%.9f\t%.9f\t%.4f\n"
                        % (
                            epoch,
                            loss,
                            loss_ae,
                            loss_simplification,
                            loss_projection,
                            duration / 60.0,
                        )
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
                (
                    loss,
                    loss_ae,
                    loss_simplification,
                    loss_projection,
                    duration,
                ) = self._single_epoch_train(held_out_data, c, only_fw=True)
                print(
                    (
                        "Held Out Data :",
                        "forward time (minutes)=",
                        "{:.4f}".format(duration / 60.0),
                        "loss=",
                        "{:.9f}".format(loss),
                        "loss_ae=",
                        "{:.9f}".format(loss_ae),
                        "loss_simplification=",
                        "{:.9f}".format(loss_simplification),
                        "loss_projection=",
                        "{:.9f}".format(loss_projection),
                    )
                )
                if log_file is not None:
                    log_file.write(
                        "On Held_Out: %04d\t%.9f\t%.9f\t%.9f\t%.9f\t%.4f\n"
                        % (
                            epoch,
                            loss,
                            loss_ae,
                            loss_simplification,
                            loss_projection,
                            duration / 60.0,
                        )
                    )
        return stats

    def get_sample(self, X, complete_fps=True):
        """Sample points from input data."""
        projected_points, nn_idx = self.sess.run(
            (self.s, self.idx), feed_dict={self.x: X}
        )

        # note: for hard projection, np.array_equal(projected_points[i], X[i][nn_idx[i]]) is True

        if complete_fps:
            (
                sampled_points,
                sampled_points_idx,
                n_unique_points,
            ) = self.simple_projection_and_continued_fps(X, projected_points, nn_idx)
        else:
            sampled_points = projected_points
            sampled_points_idx = nn_idx

        return sampled_points, sampled_points_idx

    def get_samples(self, pclouds, batch_size=50):
        """ Convenience wrapper of self.get_sample to get the samples for a set of input point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
            batch_size size of point clouds batch
        """
        projected_samples = []
        projected_idx = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            prj, prj_idx = self.get_sample(pclouds[b])
            projected_samples.append(prj)
            projected_idx.append(prj_idx)
        return np.vstack(projected_samples), np.vstack(projected_idx)

    def unique(self, arr):
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def fps_from_given_pc(self, pts, k, given_pc):
        farthest_pts = np.zeros((k, 3))
        t = np.size(given_pc) // 3
        farthest_pts[0:t] = given_pc

        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, t):
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))

        for i in range(t, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

    def fps_from_given_indices(self, pts, k, given_idx):
        farthest_pts = np.zeros((k, 3))
        idx = np.zeros(k, dtype=int)
        t = np.size(given_idx)
        farthest_pts[0:t] = pts[given_idx]
        if t > 1:
            idx[0:t] = given_idx[0:t]
        else:
            idx[0] = given_idx

        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, t):
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))

        for i in range(t, k):
            idx[i] = np.argmax(distances)
            farthest_pts[i] = pts[idx[i]]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts, idx

    def simple_projection_and_continued_fps(self, full_pc, gen_pc, idx):
        batch_size = np.size(full_pc, 0)
        k = np.size(gen_pc, 1)
        out_pc = np.zeros_like(gen_pc)
        out_pc_idx = np.zeros([batch_size, k], dtype=int)
        n_unique_points = np.zeros([batch_size, 1])
        for ii in range(0, batch_size):
            best_idx = idx[ii]
            best_idx = self.unique(best_idx)
            n_unique_points[ii] = np.size(best_idx, 0)
            out_pc[ii], out_pc_idx[ii] = self.fps_from_given_indices(
                full_pc[ii], k, best_idx
            )

        return out_pc, out_pc_idx, n_unique_points
