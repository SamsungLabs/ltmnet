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


def ltm_loss(interp_im, gt_im, weight_map, residual=None, residual_weight=0.0, vgg=None):
    """
    :param residual_weight: weight of (LTMed image + residual) in L1 loss, only used when residual is not None
    :param residual: optional residual map to interpolated image
    :param interp_im: float32 (bs, h, w, c)
        if residual is not None, interp_im = pred_im + residual
        else, interp_im = pred_im
    :param gt_im: float32  (bs, h, w, c)
    :param weight_map: (h, w, 1)
    :return:
    """
    vgg_loss = 0.0
    if vgg is not None:
        x = tf.keras.applications.vgg19.preprocess_input(interp_im * 255)
        y = tf.keras.applications.vgg19.preprocess_input(gt_im * 255)
        x_feats = vgg(x)
        y_feats = vgg(y)
        num_layers = len(x_feats)
        vgg_loss = tf.add_n([tf.reduce_mean((x_feats[i] - y_feats[i]) ** 2) for i in range(num_layers)])
        vgg_loss /= num_layers

    if residual is not None:
        without_res_im = interp_im - residual  # recover the image without residual
        l1_without_res = tf.abs(gt_im - without_res_im)
        l1_with_res = tf.abs(gt_im - interp_im)
        l1 = (1 - residual_weight) * l1_without_res + residual_weight * l1_with_res
        l1_loss = tf.reduce_mean(weight_map * l1)
    else:
        l1_loss = tf.reduce_mean(weight_map * tf.abs(gt_im - interp_im))

    return l1_loss, vgg_loss

