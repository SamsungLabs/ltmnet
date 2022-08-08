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
import keras.metrics


class PSNRMean(keras.metrics.Mean):
    def __init__(self, name="psnr", **kwargs):
        super(PSNRMean, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.image.psnr(y_pred, y_true, max_val=1.0)
        psnr = tf.reduce_mean(psnr)
        return super(PSNRMean, self).update_state(
            psnr, sample_weight=sample_weight)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


class SSIMMean(keras.metrics.Mean):
    def __init__(self, name="ssim", **kwargs):
        super(SSIMMean, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ssim = tf.image.ssim(y_pred, y_true, max_val=1.0)
        ssim = tf.reduce_mean(ssim)
        return super(SSIMMean, self).update_state(
            ssim, sample_weight=sample_weight)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

