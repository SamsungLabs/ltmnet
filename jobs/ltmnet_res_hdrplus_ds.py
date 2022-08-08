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

import os


def prep_environment():
    os.system('sudo apt-get update')
    os.system('sudo apt-get install unzip')
    os.system('pip install -r requirements.txt')


def prep_file_lists(dataset_dir):
    args_str = '--out_dir "{ds}" ' \
               '--in_dir "{ds}/input" ' \
               '--train_range 101 1500 ' \
               '--val_range 1 100 ' \
               '--test_range 1501 2000 ' \
        .format(ds=dataset_dir)

    print('Preparing file lists...')
    print('args_str = {}'.format(args_str))
    os.system('python3 -m prepare.gen_file_lists_hdrplus {} '.format(args_str))


def train(dataset_dir):
    args_str = '--exp_id "ltmnet_res_hdrplus_ds" ' \
               '--dataset_dir "{ds}" ' \
               '--train_list_fn "{ds}/images_train.txt" ' \
               '--val_list_fn "{ds}/images_val.txt" ' \
               '--test_list_fn "{ds}/images_test.txt" ' \
               '--exp_dir "./outputs/" ' \
               '--batch_size 20 ' \
               '--epochs 500 ' \
               '--control_points 256 ' \
               '--curves 3 ' \
               '--grid_size 8 8 ' \
               '--resize_to 512 512 ' \
               '--vgg_weight 1e-4 ' \
               '--l1_weight 10.0 ' \
               '--learning_rate 0.001 ' \
               '--max_train_predictions 0 ' \
               '--max_test_predictions 500 ' \
               '--max_val_predictions 0 ' \
               '--aug_type flip ' \
               '--pix_weight 1.0 0.5 0.25 ' \
               '--model_architecture res_conv ' \
               '--residual_layers 4 ' \
               '--residual_filters 8 ' \
               '--residual_weight 0.5 ' \
               '--ckpt_monitor val_ssim ' \
        .format(ds=dataset_dir)
    print('Start training...')
    print('args_str = {}'.format(args_str))
    os.system('python3 -m main {} '.format(args_str))


if __name__ == '__main__':
    prep_environment()
    os.system('pwd')
    # Replace this with your dataset directory, expected directory structure:
    # dataset_dir
    #   dataset_dir/input
    #   dataset_dir/gt
    dataset_dir = '/home/user/Data'
    prep_file_lists(dataset_dir)
    train(dataset_dir)
