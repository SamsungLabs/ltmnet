"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils import tf_utils


def get_image_stats(image_shape, grid_size):
    """
    Information about the cropped image.
    :return: grid size, tile size, sizes of the 4 margins, meshgrids.
    """

    grid_rows = grid_size[0]
    grid_cols = grid_size[1]

    residual_height = image_shape[0] % grid_rows
    residual_width = image_shape[1] % grid_cols

    tile_height = image_shape[0] // grid_rows
    tile_width = image_shape[1] // grid_cols

    margin_top = tile_height // 2
    margin_left = tile_width // 2

    margin_bot = tile_height + residual_height - margin_top
    margin_right = tile_width + residual_width - margin_left

    return tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right


def get_weight_map(image_shape, grid_size, corner_weight, border_weight, center_weight):
    image_height = image_shape[0]
    image_width = image_shape[1]
    _, _, margin_top, margin_left, margin_bot, margin_right = get_image_stats(image_shape, grid_size)

    # corners
    corner_top_left = tf.constant(corner_weight, shape=(margin_top, margin_left))
    corner_top_right = tf.constant(corner_weight, shape=(margin_top, margin_right))
    corner_bot_left = tf.constant(corner_weight, shape=(margin_bot, margin_left))
    corner_bot_right = tf.constant(corner_weight, shape=(margin_bot, margin_right))

    # borders
    border_horizontal_width = image_width - margin_left - margin_right
    border_vertical_height = image_height - margin_top - margin_bot
    border_top = tf.constant(border_weight, shape=(margin_top, border_horizontal_width))
    border_bot = tf.constant(border_weight, shape=(margin_bot, border_horizontal_width))
    border_left = tf.constant(border_weight, shape=(border_vertical_height, margin_left))
    border_right = tf.constant(border_weight, shape=(border_vertical_height, margin_right))

    # center
    center = tf.constant(center_weight, shape=(border_vertical_height, border_horizontal_width))

    # concat
    # row 1 & 3
    row_top = tf.concat([corner_top_left, border_top, corner_top_right], axis=1)
    row_center = tf.concat([border_left, center, border_right], axis=1)
    row_bot = tf.concat([corner_bot_left, border_bot, corner_bot_right], axis=1)
    pixel_weights = tf.concat([row_top, row_center, row_bot], axis=0)

    return pixel_weights


def get_one_tone_curve_fig(tone_curves):
    # tone_curves: (bs, num_curves, control_points)
    control_points = tone_curves.numpy().shape[-1]
    num_curves = tone_curves.numpy().shape[-2]
    x = tf.linspace(0, 1, control_points)
    tone_curves = tf.squeeze(tone_curves)

    fig = plt.figure()
    if num_curves == 3:
        colors = ['r', 'g', 'b']
        for idx, t in enumerate(tone_curves):
            plt.scatter(x, t, c=colors[idx], alpha=0.2)
    else:
        plt.scatter(x, tone_curves)
    plt.scatter(x, x, s=5)
    return fig


def get_tone_curve_fig(tone_curves, grid_size=(8, 8)):
    # tone_curves: (grid_size, num_curves, control_points)
    control_points = tone_curves.numpy().shape[-1]
    grids = tone_curves.numpy().shape[0]
    assert grids == grid_size[0] * grid_size[1]
    fig = plt.figure(figsize=(50, 50))
    colors = ['r', 'g', 'b']
    x = tf.linspace(0, 1, control_points)
    grid_height = grid_size[0]
    grid_width = grid_size[1]
    for i in range(grid_height):
        for j in range(grid_width):
            ind = i * grid_width + j
            plt.subplot(grid_width, grid_height, ind+1)
            for idx, t in enumerate(tone_curves[ind]):
                plt.scatter(x, t, c=colors[idx], alpha=0.2)
            plt.scatter(x, x, s=5)

    return fig


class CustomTerminateOnNaN(tf.keras.callbacks.Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self):
        super(CustomTerminateOnNaN, self).__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('total_loss')
        if loss is not None:
            loss = tf_utils.sync_to_numpy_or_python_type(loss)
        if np.isnan(loss) or np.isinf(loss):
            print('Batch %d: Invalid loss, terminating training' % (batch))
            self.model.stop_training = True
