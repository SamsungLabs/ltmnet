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

import os.path

import utils
from models.ltmnet_helpers import predict_one
import tensorflow as tf
import matplotlib.pyplot as plt
import dataloader
import numpy as np
import time


def plot_one_tone_curve(tone_curves, out=None, show=False):
    fig = utils.get_one_tone_curve_fig(tone_curves)
    plt.figure(fig.number)  # set fig as the current figure

    if out:
        dot = out.rfind('.')
        suffix = out[(dot + 1):]
        prefix = out[:dot]
        name = prefix + '_tc.' + suffix
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        print('Figure saved to: {}'.format(name))
    if show:
        plt.show()
    plt.close()


def plot_all_tone_curves(tone_curves, output_path=None, show=False, grid_size=(8, 8)):
    fig = utils.get_tone_curve_fig(tone_curves, grid_size)
    plt.figure(fig.number)  # set fig as the current figure

    if output_path:
        basename, ext = os.path.splitext(os.path.basename(output_path))
        out_name = basename + '_all_tcs' + ext
        out_fn = os.path.join(os.path.dirname(output_path), out_name)
        plt.savefig(out_fn, bbox_inches='tight', pad_inches=0)
        print('Figure saved to: {}'.format(out_fn))
    if show:
        plt.show()
    plt.close()


def get_pixelwise_l2(image):
    return np.maximum(np.sqrt((image ** 2).sum(axis=-1, keepdims=True)), 1e-8)


def get_pixelwise_avg(image):
    return np.maximum(image.mean(axis=-1, keepdims=True), 1e-8)


def plot_images(in_im, gt_im, interp, figsize=None, output_path=None, show=False, residual=None):
    ## Only transfer magnitude of predicted image
    # interp = interp.numpy().astype(np.float32)
    # mag_in = get_pixelwise_l2(in_im)
    # mag_out = get_pixelwise_l2(interp)
    # interp2 = np.clip((mag_out / mag_in) * in_im, 0, 1)

    # mag_in = get_pixelwise_avg(in_im)
    # mag_out = get_pixelwise_avg(interp)
    # interp3 = np.clip((mag_out / mag_in) * in_im, 0, 1)

    # display_list = [in_im, gt_im, interp, interp2, interp3]
    # title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Input with Pred Mag', 'Input with Pred Avg']
    #############################################

    display_list = [in_im, gt_im, interp]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    if residual is not None:
        residual = residual * 0.5 + 0.5
        display_list.append(residual)
        title.append('Predicted Residual')
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.clf()
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')

    if output_path:
        basename, ext = os.path.splitext(os.path.basename(output_path))
        out_name = basename + '_pred' + ext
        out_fn = os.path.join(os.path.dirname(output_path), out_name)
        plt.savefig(out_fn, bbox_inches='tight', pad_inches=0)
        print('Figure saved to: {}'.format(out_fn))

    if show:
        plt.show()
    plt.close()


def generate_results(in_im, gt_im, interp, tone_curves, generate_images=True, figsize=None, generate_curves='all',
                     output_path=None, show=False, grid_size=(8, 8), residual=None):
    if generate_images:
        plot_images(in_im, gt_im, interp, figsize=figsize, output_path=output_path, show=show, residual=residual)
    if generate_curves == 'all':
        # tone_curves: (grid_size, control_points)
        plot_all_tone_curves(tone_curves, output_path=output_path, show=show, grid_size=grid_size)


def generate_all_results(model, result_type, args, save_dir=None, show_ims=False, max_predictions=None,
                         max_figures=None):
    """

    :param model:
    :param result_type: train, validation, or test
    :param args:
    :param save_dir:
    :param show_ims: whether to display the output figures
    :param max_predictions: maximum number of predictions to measure metrics for
    :param max_figures: maximum number of output figures to save, None for max_figures == max_predictions
    :return:
    """

    if max_predictions == 0:
        return

    bit_depth = (8, 8)
    
    if result_type == 'train':
        display_list_fn = args.train_list_fn
    elif result_type == 'validation':
        display_list_fn = args.val_list_fn
    else:
        display_list_fn = args.test_list_fn

    inps_resized, gts_resized, filenames = dataloader.load_data_helper(args.ds_input_dir, args.ds_gt_dir,
                                                                       display_list_fn, resize_to=args.resize_to,
                                                                       bit_depth=bit_depth, ret_filenames=True,
                                                                       max_files=max_predictions)

    inps_fullsize, gts_fullsize = dataloader.load_data_helper(args.ds_input_dir, args.ds_gt_dir,
                                                              display_list_fn, resize_to=None,
                                                              bit_depth=bit_depth, normalize=False,
                                                              max_files=max_predictions)

    max_val_inp = 2 ** bit_depth[0] - 1
    max_val_gt = 2 ** bit_depth[1] - 1

    num_predictions = len(filenames)

    if max_figures is None:
        max_figures = num_predictions
    else:
        max_figures = min(max_figures, num_predictions)

    psnrs = np.zeros(num_predictions)
    ssims = np.zeros(num_predictions)
    times = np.zeros(num_predictions)

    for i in range(num_predictions):
        inp = inps_resized[i]
        inp_fullres = (inps_fullsize[i] / max_val_inp).astype(np.float64)
        inp_fullres_int = inps_fullsize[i]
        gt_full = (gts_fullsize[i] / max_val_gt).astype(np.float64)

        t0 = time.time()
        results = predict_one(model, inp, inp_fullres_int, args)
        t_full = time.time() - t0
        interp_full = tf.clip_by_value(results[0], 0.0, 1.0)
        tone_curves = results[1]

        r = None

        if 'res' in args.model_architecture:
            r = results[2]

        psnrs[i] = tf.image.psnr(interp_full, gt_full, max_val=1.0).numpy()
        ssims[i] = tf.image.ssim(interp_full, gt_full, max_val=1.0).numpy()
        times[i] = t_full
        
        if i < max_figures:
            metrics_str = '-psnr_{:.4f}_ssim_{:.4f}'.format(psnrs[i], ssims[i])
            output_path = os.path.join(save_dir, filenames[i] + metrics_str + '.png') if save_dir else None
                
            # Only generate up to max_figures result images to save time and storage
            generate_results(inp_fullres, gt_full, interp_full, tone_curves, generate_images=True, figsize=(30, 50),
                             generate_curves='all', output_path=output_path, show=show_ims, grid_size=args.grid_size,
                             residual=r)

    psnr_str = '{}: Average PSNR for full-size results: {}'.format(result_type, psnrs.mean())
    ssim_str = '{}: Average SSIM for full-size results: {}'.format(result_type, ssims.mean())
    time_f_str = '{}: Average inference time for full-res inputs: {} sec.'.format(result_type, times.mean())

    # Save timings
    f = open(args.time_pred_fn, 'a')
    f.write(time_f_str + '\n')
    f.close()

    # Save evaluation results
    f = open(args.eval_fn, 'a')
    f.write('\n'.join([psnr_str, ssim_str]) + '\n')
    f.close()
