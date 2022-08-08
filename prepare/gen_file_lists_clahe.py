"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com; lucyzhao.zlx@gmail.com)

Generate a list of prefixes for the LTM dataset.

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


def gen_files(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    ds_list_fn = os.path.join(args.out_dir, 'image_list.txt')
    train_list = ['a{:04d}'.format(x) for x in range(args.image_range[0], args.image_range[1] + 1)]

    if args.images_rm_fn is not None:
        prefixes_rm = []
        with open(args.images_rm_fn, 'r') as f:
            prefixes_rm = [p.strip() for p in f.readlines()]
        train_list = sorted(list(set(train_list) - set(prefixes_rm)))

    with open(ds_list_fn, 'w') as f:
        lines = '\n'.join(train_list)
        f.write(lines)


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--out_dir', type=str, help='Output directory.')
    arg_parser.add_argument('--image_range', nargs='+', type=int, help='Index range for data. 1-based. '
                                                                       'start, end both inclusive.', default=[1,
                                                                                                              2500])
    arg_parser.add_argument('--images_rm_fn', type=str, help='Path to a file containing prefixes of files to remove.', default=None)
    args_ = arg_parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()
    gen_files(args)


