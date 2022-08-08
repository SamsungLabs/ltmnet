"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Prepare LTM dataset for LTMNet training.

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import os

import cv2
import numpy as np
import dataloader
from argparse import ArgumentParser
from prepare.nima.model import Nima


def resize_image(image, max_dim=800):
    h, w = image.shape[:2]
    if h > w:
        new_h = int(max_dim + .5)
        new_w = int(w / h * max_dim + .5)
    else:
        new_h = int(h / w * max_dim + .5)
        new_w = int(max_dim + .5)
    if len(image.shape) == 3:
        image_resized = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        raise Exception('Invalid image depth = {}'.format(len(image.shape)))
    return image_resized


def apply_clahe(x, clip_limit, grid_size):
    """
    x - RGB image, [0-255] uint8 
    """
    channel_order = 'RGB'
    to_ycrcb_flag = eval('cv2.COLOR_{}2YCR_CB'.format(channel_order))
    from_ycrcb_flag = eval('cv2.COLOR_YCR_CB2{}'.format(channel_order))

    y_cr_cb = cv2.cvtColor(x, to_ycrcb_flag)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    x = np.dstack((clahe.apply(y_cr_cb[:, :, 0]), y_cr_cb[:, :, 1], y_cr_cb[:, :, 2]))
    x = cv2.cvtColor(x, from_ycrcb_flag)

    return x


def save_image(image, filename, out_dir, clip_limit, grid_size, max_dim=1024):
    out_name = os.path.basename(filename)
    out_name = '{}-cl{:.1f}-gs{}x{}.png'.format(out_name, clip_limit, grid_size[0], grid_size[1])
    out_name = os.path.join(out_dir, out_name)
    image = image[:, :, ::-1]
    image = resize_image(image, max_dim=max_dim)
    cv2.imwrite(out_name, image)
    print('Image saved to: {}'.format(out_name))


def save_intermediates(image_cl_gs_nima, filename, intermediate_out_dir, max_dim=256):
    basename = os.path.basename(filename)
    for tup in image_cl_gs_nima:
        image, clip_limit, grid_size, nima = tup
        rs_image = resize_image(image, max_dim=max_dim)[:, :, ::-1]
        out_name = '{}-cl{:.1f}-gs{}x{}-nima_{:6f}.png'.format(basename, clip_limit, grid_size[0], grid_size[1], nima)
        out_name = os.path.join(intermediate_out_dir, out_name)
        cv2.imwrite(out_name, rs_image)


def calc_mean_score(scores):
    scores = np.array(scores)
    scores_norm = scores / scores.sum() 
    return (scores_norm * np.arange(1, 11)).sum()


def find_best_by_nima(images, nima):
    images_resized = [np.array(cv2.resize(x, (224, 224), interpolation=cv2.INTER_NEAREST)) for x in images]
    images_resized = np.stack(images_resized, axis=0)  # (bs, 224, 224, 3)
    images_resized = nima.preprocessing_function()(images_resized)
    predictions = nima.nima_model.predict(images_resized, workers=8, use_multiprocessing=True, verbose=1)
    mean_preds = [calc_mean_score(p) for p in predictions]
    print('mean preds: ', mean_preds)
    best_score = max(mean_preds)
    best_idx = mean_preds.index(best_score)
    best_image = images[best_idx]
    return best_image, best_idx, mean_preds


def select_clahe_by_nima(args):
    # Pass dataset_dir as a placeholder for path to target images, because no need for target images here.
    inps_fullsize, _, filenames = dataloader.load_data_helper(args.dataset_dir, args.dataset_dir,
                                                              args.filelist_fn, resize_to=None,
                                                              bit_depth=(8, 8), normalize=False,
                                                              ret_filenames=True,
                                                              max_files=None)
    # build model and load weights
    nima = Nima('MobileNet', weights=None)
    nima.build()
    nima.nima_model.load_weights(args.nima_model_fn)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    if args.intermediate_out_dir:
        os.makedirs(args.intermediate_out_dir, exist_ok=True)

    for i in range(len(inps_fullsize)):
        inp = inps_fullsize[i]
        filename = filenames[i]
        clip_limits = np.arange(0.5, 3.0, 0.5)  # [0.5, 1.0, 1.5, 2.0. 2.5]
        grid_sizes = [(2, 2), (4, 4), (8, 8)]

        outputs = [(apply_clahe(inp, cl, gs), cl, gs) for cl in clip_limits for gs in grid_sizes]
        clahe_images = [tup[0] for tup in outputs]
        best_output, best_idx, nima_scores = find_best_by_nima(clahe_images, nima)
        best_cl = outputs[best_idx][1]
        best_gs = outputs[best_idx][2]

        # append nima score for each image to outputs
        for i, score in enumerate(nima_scores):
            tup = outputs[i]
            outputs[i] = (tup[0], tup[1], tup[2], score)

        # save the best output
        if args.out_dir:
            save_image(best_output, filename, args.out_dir, best_cl, best_gs, max_dim=args.max_dimension)
        if args.intermediate_out_dir:
            save_intermediates(outputs, filename, args.intermediate_out_dir, max_dim=args.intermediate_dimension)


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--out_dir', type=str, help='Output directory for saving selected CLAHE-ed image.')
    arg_parser.add_argument('--intermediate_out_dir', type=str,
                            help='Output directory for saving all CLAHE-ed candidate images.')
    arg_parser.add_argument('--max_dimension', type=int, default=1024,
                            help='Maximum dimension for saving all best candidate images.')
    arg_parser.add_argument('--intermediate_dimension', type=int, default=256,
                            help='Maximum dimension for saving all CLAHE-ed candidate images.')
    arg_parser.add_argument('--dataset_dir', type=str, help='Dataset directory.')
    arg_parser.add_argument('--filelist_fn', type=str, help='Path to list of filenames.')
    arg_parser.add_argument('--nima_model_fn', type=str, help='Path NIMA pretrained model.',
                            default='./nima/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5')

    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    select_clahe_by_nima(args)
