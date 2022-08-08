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


def movedata(dataset_dir):
    print('Moving data...')

    input_dir = '{}/input'.format(dataset_dir)
    os.makedirs(input_dir, exist_ok=True)
    input_ds_name = 'mit-adobe-input-expertcwb-srgb-8bit-1024-png-all'

    # Replace ./dataset/{}.zip with your path to 
    # MIT-Adobe FiveK data exported from LightRoom.
    os.system(
        'cp "./dataset/{}.zip" "{}"'.format(input_ds_name, dataset_dir))
    os.system('unzip -j -q {}/{}.zip -d {}'.format(dataset_dir, input_ds_name, input_dir))


def prep_file_lists(dataset_dir):
    args_str = '--out_dir "{}" ' \
               '--image_range 1 2500 ' \
               '--images_rm_fn "prepare/data/mit-adobe-clahe-15v_images_rm.txt" ' \
        .format(dataset_dir)

    print('Preparing file lists...')
    print('args_str = {}'.format(args_str))
    os.system('python3 -m prepare.gen_file_lists_clahe {} '.format(args_str))


def gen_mit_adobe_clahe(dataset_dir):
    args_str = '--dataset_dir "{ds}/input" ' \
               '--filelist_fn "{ds}/image_list.txt" ' \
               '--out_dir "{ds}/mit-adobe-clahe-15v/long-edge-1024" ' \
               '--intermediate_out_dir "{ds}/mit-adobe-clahe-15v/long-edge-256" ' \
               '--nima_model_fn prepare/nima/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 ' \
        .format(ds=dataset_dir)
    print('args_str = {}'.format(args_str))
    os.system('python3 -m prepare.prep_mit_adobe_clahe {} '.format(args_str))


if __name__ == '__main__':
    prep_environment()
    os.system('pwd')
    # Replace this with your dataset directory, expected directory structure:
    # dataset_dir
    #   dataset_dir/input
    #   dataset_dir/gt
    dataset_dir = '/home/user/Data'
    movedata(dataset_dir)
    prep_file_lists(dataset_dir)
    gen_mit_adobe_clahe(dataset_dir)
