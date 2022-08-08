"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import numpy as np
import os
import cv2
from glob import glob


def find_path_to_gt(in_file):
    """
    Custom function for finding ground truth image. Need to be changed if folder structure changes.
    :param in_file:
    :return:
    """
    gt_file = in_file.replace('/input/', '/gt/')
    gt_dir = os.path.dirname(gt_file)
    in_basename = os.path.basename(in_file)
    # check files with same ID (first 5 characters)
    gt_fns = glob(os.path.join(gt_dir, in_basename[:5] + '*'))
    if len(gt_fns) != 1:
        raise Exception('Zero or multiple GT files exist: {}\nin_basename = {}'.format(gt_fns, in_basename))
    gt_file = gt_fns[0]

    return gt_file


def resize(inp, gt, target_size):
    height = target_size[0]
    width = target_size[1]
    inp = cv2.resize(inp, (width, height), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_AREA)
    return inp, gt


def load_data_helper(input_dir, gt_dir, im_list_fn, resize_to=(512, 512), bit_depth=(8, 8), normalize=True,
                     ret_filenames=False, max_files=None):
    """
    Loads MIT-Adobe FiveK dataset according to filenames specified in im_list_fn.
    :param input_dir: Directory to input images
    :param gt_dir: Directory to ground truth images
    :param im_list_fn: Path to a file containing a list of image filenames to load
    :param resize_to:
    :param bit_depth:
    :param normalize: Whether to normalize images
    :param ret_filenames: Whether to return filenames
    :param max_files:
    :return: An array of variable size images, or an np.array of uniform resized images
    """
    inputs = []
    gts = []
    max_val_inp = 2 ** bit_depth[0] - 1
    max_val_gt = 2 ** bit_depth[1] - 1
    allow_types = ['jpg', 'png', 'tif']
    in_fns = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1][-3:] in allow_types]
    gt_fns = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1][-3:] in allow_types]
    # Build a dictionary of {file_prefix: file_name}
    in_fn_dict = {os.path.splitext(x)[0].split('-')[0]: x for x in in_fns}
    gt_fn_dict = {os.path.splitext(x)[0].split('-')[0]: x for x in gt_fns}

    im_list = open(im_list_fn).readlines()
    im_list = im_list if max_files is None else im_list[:max_files]
    im_list = [x.rstrip() for x in im_list]

    for im_prefix in im_list:
        if im_prefix not in in_fn_dict or im_prefix not in gt_fn_dict:
            raise ValueError('Cannot find image {} in input or ground truth dataset directory.'.format(im_prefix))

        in_fn = in_fn_dict[im_prefix]
        gt_fn = gt_fn_dict[im_prefix]
        in_fp = os.path.join(input_dir, in_fn)
        gt_fp = os.path.join(gt_dir, gt_fn)
        inp = np.array(cv2.cvtColor(cv2.imread(in_fp, -1), cv2.COLOR_BGR2RGB))
        gt = np.array(cv2.cvtColor(cv2.imread(gt_fp, -1), cv2.COLOR_BGR2RGB))

        if resize_to:
            inp, gt = resize(inp, gt, resize_to)

        if normalize:
            inp = (inp / max_val_inp).astype(np.float32)
            gt = (gt / max_val_gt).astype(np.float32)

        inputs.append(inp)
        gts.append(gt)

    if resize_to:
        inputs = np.stack(inputs)
        gts = np.stack(gts)

    if ret_filenames:
        return inputs, gts, im_list
    else:
        return inputs, gts


def load_data(input_dir, gt_dir, train_list_fn, val_list_fn, test_list_fn, resize_to=(512, 512),
              bit_depth=(8, 8)):
    inp_train, gt_train = load_data_helper(input_dir, gt_dir, train_list_fn, resize_to, bit_depth=bit_depth)
    inp_val, gt_val = load_data_helper(input_dir, gt_dir, val_list_fn, resize_to, bit_depth=bit_depth)
    inp_test, gt_test = load_data_helper(input_dir, gt_dir, test_list_fn, resize_to, bit_depth=bit_depth)
    return inp_train, gt_train, inp_val, gt_val, inp_test, gt_test
