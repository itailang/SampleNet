"""
Created on September 7th, 2018

@author: itailang
"""

import tensorflow as tf
import numpy as np

from .encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only


def sampler_with_convs_and_symmetry_and_fc(
    in_signal, pc_dim_out, non_linearity=tf.nn.relu
):
    """A sampling network that generate sample points from an input point cloud."""

    n_points, dummy = pc_dim_out
    if dummy != 3:
        raise ValueError()

    encoder_args = {
        "n_filters": [64, 128, 128, 256, 128],
        "filter_sizes": [1],
        "strides": [1],
        "non_linearity": non_linearity,
        "b_norm": True,
        "verbose": True,
    }
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    decoder_args = {
        "layer_sizes": [256, 256, np.prod([n_points, 3])],
        "b_norm": False,
        "b_norm_finish": False,
        "verbose": True,
    }
    out_signal = decoder_with_fc_only(layer, **decoder_args)

    out_signal = tf.reshape(out_signal, [-1, n_points, 3])
    return out_signal
