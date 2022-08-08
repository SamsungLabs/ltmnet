"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Image augmentation utility functions.

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import adjust_contrast, adjust_brightness


def flip_left_right(images):
    return [tf.image.flip_left_right(image) for image in images]


def flip_up_down(images):
    return [tf.image.flip_up_down(image) for image in images]


def random_flip_left_right(images):
    do_flip = tf.random.uniform([]) > 0.5
    if do_flip:
        images = flip_left_right(images)
    return images


def random_flip_up_down(images):
    do_flip = tf.random.uniform([]) > 0.5
    if do_flip:
        images = flip_up_down(images)
    return images


def random_contrast(image):
    lower, upper = 0.8, 1.0
    contrast_factor = tf.random.uniform([], lower, upper)
    return adjust_contrast(image, contrast_factor)


def random_brightness(image):
    lower, upper = -0.2, 0.1
    delta = tf.random.uniform([], lower, upper)
    return adjust_brightness(image, delta)
