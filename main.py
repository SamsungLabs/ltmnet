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

import tensorflow as tf
import time
import dataloader
from augment import random_flip_left_right, random_flip_up_down, random_contrast, random_brightness
from train import train
from arg_parser import prepare_args
from result_gen import generate_all_results

'''
Load Data
'''


def load_image(in_im, gt_im, bit_depth, int_type):
    in_im_int = tf.cast(in_im * (2 ** bit_depth - 1), int_type)
    return in_im, gt_im, in_im_int


def augment(dataset, aug_type='none'):
    """
    Perform random augmentation (flip, contrast).
    Args:
        dataset: TF dataset.
        aug_type: A string indicating a combination of augmentation options
            (e.g., 'none', 'flip', 'contrast', 'flip_contrast', ...).

    Returns:
        Augmented dataset.
    """
    if 'flip' in aug_type:
        dataset = dataset.map(lambda x, y: random_flip_left_right([x, y]))
        dataset = dataset.map(lambda x, y: random_flip_up_down([x, y]))
    if 'contrast' in aug_type:
        dataset = dataset.map(lambda x, y: (random_contrast(x), y))
    if 'brightness' in aug_type:
        dataset = dataset.map(lambda x, y: (random_brightness(x), y))
    return dataset


def pad_gt(in_im, gt_im, paddings):
    pt, pb, pl, pr = paddings
    paddings = tf.constant([[pt, pb], [pl, pr], [0, 0]])  # [[top, bot], [left, right]]
    gt_im = tf.pad(gt_im, paddings, 'SYMMETRIC')
    return in_im, gt_im


def load_dataset(args):
    int_type = tf.uint8
    bit_depth = (8, 8)

    autotune = tf.data.experimental.AUTOTUNE
    inp_train, gt_train, inp_val, gt_val, _, _ = dataloader.load_data(args.ds_input_dir, args.ds_gt_dir,
                                                                      args.train_list_fn, args.val_list_fn,
                                                                      args.test_list_fn,
                                                                      resize_to=args.resize_to,
                                                                      bit_depth=bit_depth)
    assert len(gt_val) % args.batch_size == 0, 'Validation data must be divisible by batch size for metrics to work ' \
                                               'correctly. '
    train_dataset = tf.data.Dataset.from_tensor_slices((inp_train, gt_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((inp_val, gt_val))

    train_dataset = augment(train_dataset, args.aug_type)
    train_dataset = train_dataset.map(lambda x, y: load_image(x, y, bit_depth[0], int_type),
                                      num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(buffer_size=100)
    train_dataset = train_dataset.batch(batch_size=args.batch_size, num_parallel_calls=autotune)

    val_dataset = val_dataset.map(lambda x, y: load_image(x, y, bit_depth[0], int_type),
                                  num_parallel_calls=autotune)
    val_dataset = val_dataset.batch(batch_size=args.batch_size, num_parallel_calls=autotune)
    return train_dataset, val_dataset


def main(args):
    if not args.eval:

        t0 = time.time()
        train_dataset, test_dataset = load_dataset(args)
        t_ds = time.time() - t0

        t0 = time.time()
        if args.learning_rate_schedule:
            num_steps = 1400 // args.batch_size
            learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [25 * num_steps, 50 * num_steps, 100 * num_steps],
                [2e-3, 1e-3, 1e-4, 1e-5]
            )
            lr = learning_rate_fn
        else:
            lr = args.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        train(optimizer, train_dataset, test_dataset, args)
        t_train = time.time() - t0

        t_ds_str = "Total time preparing dataset: {:.2f} sec, {:.4f} min".format(t_ds, t_ds / 60.0)
        t_train_str = "Total time training: {:.2f} sec, {:.4f} hrs".format(t_train, t_train / 3600.0)
        t_str = '{}\n{}'.format(t_ds_str, t_train_str)

        # Save training times
        f = open(args.time_train_fn, 'w')
        f.write(t_str)
        f.close()

    # Generate results
    print('Evaluating...')
    model = tf.keras.models.load_model('{}'.format(args.ckpt_dir))
    generate_all_results(model, 'validation', args, save_dir=args.results_dir_val, show_ims=False,
                         max_predictions=args.max_val_predictions)
    generate_all_results(model, 'train', args, save_dir=args.results_dir_train, show_ims=False,
                         max_predictions=args.max_train_predictions)
    generate_all_results(model, 'test', args, save_dir=args.results_dir_test, show_ims=False,
                         max_predictions=args.max_test_predictions, max_figures=3)


if __name__ == '__main__':
    args = prepare_args()
    main(args)
