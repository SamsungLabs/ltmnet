"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import argparse
from backports.zoneinfo import ZoneInfo
import datetime
import os


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--eval', action='store_true',
                            help='Whether to load an existing model and evaluate it on validation images. When set to '
                                 'true, exp_dir and exp_id should point to an existing directory.')
    arg_parser.add_argument('--pretrained_model_dir', type=str, help='Path to a pretrained model')
    arg_parser.add_argument('--dataset_dir', '-ds', type=str,
                            help='Dataset directory. Assumed directory structure: ./input, ./gt', required=True)
    arg_parser.add_argument('--train_list_fn', type=str, help='Text file containing filenames for training images.',
                            required=True)
    arg_parser.add_argument('--val_list_fn', type=str, help='Text file containing filenames for validation images.',
                            required=True)
    arg_parser.add_argument('--test_list_fn', type=str, help='Text file containing filenames for test images.',
                            required=True)
    arg_parser.add_argument('--exp_dir', type=str, help='Directory for saving logs and checkpoints. When eval=False, '
                                                        'experiment outputs will be stored under exp_dir/<exp_id>. '
                                                        'When eval=True, experiment outputs will be stored under '
                                                        'exp_dir/<pretrained_model_name>_results.')
    arg_parser.add_argument('--exp_id', type=str, help='Experiment ID.', default='')
    arg_parser.add_argument('--batch_size', '-bs', type=int, help='Batch size.')
    arg_parser.add_argument('--epochs', type=int, help='Number of epochs to train.', default=300)
    arg_parser.add_argument('--control_points', '-cp', type=int, help='Number of control points per curve.',
                            default=256)
    arg_parser.add_argument('--curves', type=int, help='Number of curves per image. Options: "3" -> 1 curve for each '
                                                       'R,G,B channel, "1" -> 1 curve for all RGB channels ',
                            default=3)
    arg_parser.add_argument('--grid_size', '-gs', nargs='+', type=int,
                            help='Size of the grid for dividing an image; (height, width)',
                            default=[8, 8])
    arg_parser.add_argument('--resize_to', nargs='+', type=int,
                            help='Input size to resize to before feeding into neural net; (height, width)',
                            default=[512, 512])
    arg_parser.add_argument('--ckpt_monitor', type=str, help='Which metric the checkpoint saver monitors.',
                            default='val_psnr')
    arg_parser.add_argument('--ckpt_monitor_mode', type=str, help='Whether to minimize or maximize the metric '
                                                                  'monitored by the checkpoint.',
                            default='max')
    arg_parser.add_argument('--max_val_predictions', type=int, help='Maximum number of validation images to generate '
                                                                    'predictions for. None for no upper limit', 
                            default=None)
    arg_parser.add_argument('--max_train_predictions', type=int, help='Maximum number of training images to generate '
                                                                      'predictions for.', default=0)
    arg_parser.add_argument('--max_test_predictions', type=int, help='Maximum number of testing images to generate '
                                                                     'predictions for.', default=0)
    arg_parser.add_argument('--vgg_weight', type=float, help='Weight of vgg loss.', default=0)
    arg_parser.add_argument('--l1_weight', type=float, help='Weight of L1 loss.', default=1.0)
    arg_parser.add_argument('--pix_weight', nargs='+', type=float, help='Pixel map weights. Corner, border, center.',
                            default=[1., 1., 1.])
    arg_parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=1e-3)
    arg_parser.add_argument('--learning_rate_schedule', action='store_true', help='Use pre-defined learning rate '
                                                                                  'scheduler.')
    arg_parser.add_argument('--aug_type', type=str,
                            help='Image augmentation type: none, flip, contrast, flip_contrast, etc.', default='none')
    arg_parser.add_argument('--model_architecture', type=str, help='Model architecture.', default='conv')
    arg_parser.add_argument('--residual_layers', type=int, help='Number of layers in the residual network.',
                            default=5)
    arg_parser.add_argument('--residual_filters', type=int, help='Number of filters in the residual network.',
                            default=32)
    arg_parser.add_argument('--residual_weight', type=float, help='Weight of (LTMed image + residual) in L1 loss.',
                            default=1.0)

    args = arg_parser.parse_args()
    return args


def prepare_experiment(args):
    args.ds_input_dir = os.path.join(args.dataset_dir, 'input')
    args.ds_gt_dir = os.path.join(args.dataset_dir, 'gt')

    if args.eval:
        ckpt_name = '{}_results'.format(os.path.basename(args.pretrained_model_dir))
        out_dir = os.path.join(args.exp_dir, ckpt_name)
        args.ckpt_dir = args.pretrained_model_dir
        exp_name = ckpt_name
    else:
        # Create experiment directory for new experiment
        timestamp = datetime.datetime.now(tz=ZoneInfo('US/Eastern')).strftime("%y%m%d_%H%M")
        if args.exp_id:
            exp_name = '{}_{}'.format(args.exp_id, timestamp)
        else:
            exp_name = 'exp_{}'.format(timestamp)
        out_dir = os.path.join(args.exp_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        # Directories/Files for checkpoint, training logs, model summary
        args.ckpt_dir = os.path.join(out_dir, 'ckpt', 'ckpt_best')
        args.tb_log_dir = os.path.join(out_dir, 'tensorboard_logs')
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.tb_log_dir, exist_ok=True)

        args.train_hist_fn = os.path.join(out_dir, 'training_history.pickle')
        args.model_summary_fn = os.path.join(out_dir, 'model_summary.txt')
        args.time_train_fn = os.path.join(out_dir, 'time_train.txt')

    # Directories/Files for prediction outputs
    args.results_dir = os.path.join(out_dir, 'results')
    args.results_dir_val = os.path.join(args.results_dir, 'validation')
    args.results_dir_train = os.path.join(args.results_dir, 'train')
    args.results_dir_test = os.path.join(args.results_dir, 'test')
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.results_dir_val, exist_ok=True)
    os.makedirs(args.results_dir_train, exist_ok=True)
    os.makedirs(args.results_dir_test, exist_ok=True)
    args.time_pred_fn = os.path.join(args.results_dir, 'time_pred.txt')
    args.eval_fn = os.path.join(args.results_dir, 'eval.txt')

    # Save experiment's arguments/parameters
    if not args.eval:
        args_file = os.path.join(out_dir, 'args.txt')
        args_str = ''
        for key in vars(args):
            args_str += '{}: {}\n'.format(key, vars(args)[key])
        f = open(args_file, 'w')
        f.write(args_str)
        f.close()

    return args


def prepare_args():
    args = parse_args()
    args = prepare_experiment(args)
    return args
