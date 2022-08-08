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

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import linear_model, metrics
import os
import glob
import argparse


def load_image(path):
    max_val = 2 ** 8 - 1
    return np.array(cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)).astype(np.float32) / max_val


def polyfit_channel(x, y, order):
    """
    Fit a polynomial using polynomial regression to a set of (x, y) pairs.
    (x1, y1): x1 - pixel 1 of input image, y1 - pixel 1 of ground truth image
    :param x: input pixels
    :param y: ground truth pixels
    :param order: order of polynomial
    :return: RMSE, query points and predicted points for figure plotting
    """
    x = x.flatten()
    y = y.flatten()
    X = np.array([x ** i for i in range(order + 1)]).T
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    y_pred = regr.predict(X)
    rmse = np.sqrt(metrics.mean_squared_error(y_pred, y))

    x_test_plot = np.linspace(x.min(), x.max(), 200)
    X_test_plot = np.array([x_test_plot ** i for i in range(order + 1)]).T
    y_pred_plot = regr.predict(X_test_plot)
    return rmse, x_test_plot, y_pred_plot


def plot_transfer_functions(input, gt, x_test_plots, y_pred_plots, order, rmse, out_path):
    titles = ['input', 'transfer function R', 'transfer function G', 'transfer function B', 'ground truth']
    fig, axes = plt.subplots(1, len(titles), figsize=(15, 4))
    fig.suptitle('Polynomial Order: {}, RMSE: {:.6f}'.format(order, rmse))

    axes[0].set_title(titles[0])
    axes[0].imshow(input)
    axes[0].axis('off')

    for i in range(3):
        axes[i + 1].set_title(titles[i + 1])
        axes[i + 1].scatter(input[..., i], gt[..., i], s=4)
        axes[i + 1].plot(x_test_plots[i], y_pred_plots[i], color='r')
        asp = np.diff(axes[i + 1].get_xlim())[0] / np.diff(axes[i + 1].get_ylim())[0]
        axes[i + 1].set_aspect(asp)

    axes[-1].set_title(titles[-1])
    axes[-1].imshow(gt)
    axes[-1].axis('off')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def polyfit_image(input, gt, order=4, savefig=False, out_path=None):
    """
    Fit a polynomial to transfer functions of each channel.
    Compute an average of the RMSE of all 3 channels.
    :param input:
    :param gt:
    :param order: order of polynomial
    :param savefig:
    :param out_path:
    :return:
    """
    rmse_total = 0
    x_test_plots = []
    y_pred_plots = []
    for c in range(3):
        rmse, x_test_plot, y_pred_plot = polyfit_channel(input[..., c], gt[..., c], order=order)
        rmse_total += rmse
        x_test_plots.append(x_test_plot)
        y_pred_plots.append(y_pred_plot)

    rmse_avg = rmse_total / 3
    if savefig:
        plot_transfer_functions(input, gt, x_test_plots, y_pred_plots, order, rmse_avg, out_path=out_path)
    return rmse_avg


def compute_rmse(input_dir, gt_dir, output_dir, order=6, savefig=False):
    """
    Save transfer function plots and RMSE by image name to output_dir
    """
    in_fns = glob.glob(os.path.join(input_dir, '*.png'))
    fns = [os.path.basename(f) for f in in_fns]
    fns = sorted(fns)

    rmse_by_imname_fn = os.path.join(output_dir, 'rmse_by_imname.txt')
    rmse_f = open(rmse_by_imname_fn, 'w')

    for f in fns:
        in_path = os.path.join(input_dir, f)
        gt_path = os.path.join(gt_dir, f)
        in_im = load_image(in_path)
        gt_im = load_image(gt_path)
        out_name = os.path.splitext(f)[0] + '.jpg'
        transfer_func_out_path = os.path.join(output_dir, out_name)
        rmse = polyfit_image(in_im, gt_im, order=order, savefig=savefig, out_path=transfer_func_out_path)
        rmse_f.write('{}: {}\n'.format(f, rmse))
    rmse_f.close()


def sort_image_name_by_rmse_and_compute_avg(rmse_fn):
    """
    Content format of rmse_fn
    <filenmame>: <rmse>
    """
    MAX_LINES = 5000
    rmse_f = open(rmse_fn, 'r')
    rmse_by_fn = rmse_f.readlines()
    rmse_tuples = [(s.split(':')[0], float(s.split(':')[1])) for s in rmse_by_fn]

    rmse_avg = np.array([t[1] for t in rmse_tuples]).mean()
    rmse_avg_fn = os.path.join(os.path.dirname(rmse_fn), 'rmse_avg.txt')
    rmse_avg_f = open(rmse_avg_fn, 'w')
    rmse_avg_f.write(str(rmse_avg))
    rmse_avg_f.close()

    rmse_tuples_sorted = sorted(rmse_tuples, key=lambda t: t[1], reverse=True)

    fns_sorted_fn = os.path.join(os.path.dirname(rmse_fn), 'sorted_imname_by_rmse.txt')
    prefixes = [rmse_tuple[0].split('-')[0] for rmse_tuple in rmse_tuples_sorted][:MAX_LINES]
    fns_sorted = open(fns_sorted_fn, 'w')
    for p in prefixes:
        fns_sorted.write(p + '\n')
    fns_sorted.close()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_dir', type=str, help='Input images.', required=True)
    arg_parser.add_argument('--gt_dir', type=str, help='Ground truth images.', required=True)
    arg_parser.add_argument('--out_dir', type=str, help='Directory to save transfer function figures and text files.',
                            required=True)
    arg_parser.add_argument('--poly_order', type=int, help='Degree of polynomial.', default=6)
    arg_parser.add_argument('--savefig', action='store_true',
                            help='Whether to save transfer functions and fitted polynomial as figures. Figures will '
                                 'be saved to out_dir.')

    args = arg_parser.parse_args()

    compute_rmse(args.input_dir, args.gt_dir, args.out_dir, order=args.poly_order, savefig=args.savefig)
    rmse_fn = os.path.join(args.out_dir, 'rmse_by_imname.txt')
    sort_image_name_by_rmse_and_compute_avg(rmse_fn)


if __name__ == '__main__':
    main()





