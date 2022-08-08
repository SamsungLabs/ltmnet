"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)

Generate a list of HDR+ file prefixes for each training, validation, and test dataset.

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""


import os
from argparse import ArgumentParser
import glob
import random


def gen_files(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    train_list_fn = os.path.join(args.out_dir, 'images_train.txt')
    val_list_fn = os.path.join(args.out_dir, 'images_val.txt')
    test_list_fn = os.path.join(args.out_dir, 'images_test.txt')
    in_fns = glob.glob(os.path.join(args.in_dir, '*.jpg'))
    in_fns = sorted([os.path.splitext(os.path.basename(fn))[0] for fn in in_fns])
    random.Random(0).shuffle(in_fns)
    train_list = sorted(in_fns[args.train_range[0]: args.train_range[1] + 1])
    val_list = sorted(in_fns[args.val_range[0]: args.val_range[1] + 1])
    test_list = sorted(in_fns[args.test_range[0]: args.test_range[1] + 1])

    all_fn = [train_list_fn, val_list_fn, test_list_fn]
    all_list = [train_list, val_list, test_list]
    for i in range(3):
        with open(all_fn[i], 'w') as f:
            lines = '\n'.join(all_list[i])
            f.write(lines)


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--out_dir', type=str, help='Output directory.')
    arg_parser.add_argument('--in_dir', type=str, help='Input directory.')
    arg_parser.add_argument('--train_range', nargs='+', type=int, help='Index range for training data. 1-based. '
                                                                       'start, end both inclusive.', default=[101,
                                                                                                              600])
    arg_parser.add_argument('--val_range', nargs='+', type=int, help='Index range for validation data. 1-based. '
                                                                     'start, end both inclusive.', default=[1, 100])
    arg_parser.add_argument('--test_range', nargs='+', type=int, help='Index range for training data. 1-based. start, '
                                                                      'end both inclusive.', default=[601, 741])
    args_ = arg_parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()
    gen_files(args)

    # example run
    # python -m prepare.gen_file_lists \
    # --out_dir "./mock_data" \
    # --train_range 101 600 \
    # --val_range 1 100 \
    # --test_range 4501 5000

