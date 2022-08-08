"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Prepare HDR+ dataset for LTM training.

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import os
from fractions import Fraction

import tifffile as tiff
import cv2
from argparse import ArgumentParser
from glob import glob
from .pipeline.pipeline import run_pipeline
from .pipeline.pipeline_utils import get_metadata, default_cropping, active_area_cropping, \
    get_raw_image, fix_orientation


def resize(image):
    h, w = image.shape[:2]
    if h > w:
        new_h = 1024
        new_w = int((new_h / h) * w)
    else:
        new_w = 1024
        new_h = int((new_w / w) * h)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def process_hdrplus_frame(dng_fn, metadata, lsc_fn=None):
    stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'lens_shading_correction', 'white_balance',
              'demosaic', 'xyz', 'srgb', 'gamma']

    params = {
        'input_stage': 'raw',
        'output_stage': 'gamma',
        'save_as': 'jpg',  # options: 'jpg', 'png', 'tif', etc.
        'white_balancer': 'default',  # options: default, or self-defined module
        'demosaicer': 'EA', 
    }

    image = get_raw_image(dng_fn)

    # load lens shading correction map from a separate file
    if lsc_fn is not None:
        lsc_map = tiff.imread(lsc_fn)
        metadata['lsc_map'] = lsc_map

    image = run_pipeline(image, metadata=metadata, params=params, stages=stages)
    return image, metadata


def prep_hdrplus(hdrplus_dir, out_dir, im_list_fn, use_merged=True):
    bursts_dir = os.path.join(hdrplus_dir, 'bursts')
    results_dir = os.path.join(hdrplus_dir, 'results_20171023')

    input_dir = os.path.join(out_dir, 'input-srgb-gamma')
    gt_dir = os.path.join(out_dir, 'gt-final')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    stats_fn = os.path.join(out_dir, 'stats.csv')
    stats_file = open(stats_fn, 'w')
    stats_file.write('burst_name,make,model,srgb_size,final_size,size_mismatch,dy,dx\n')

    im_list = open(im_list_fn).readlines()
    im_list = [x.rstrip() for x in im_list]

    if use_merged:
        burst_fns = sorted(glob(os.path.join(results_dir, '*')))
    else:
        burst_fns = sorted(glob(os.path.join(bursts_dir, '*')))

    burst_names = [os.path.basename(bfn) for bfn in burst_fns]

    for k, burst_name in enumerate(burst_names):
        print('processing {} / {}: {}'.format(k + 1, len(burst_names), burst_name))
        if burst_name not in im_list:
            print('{} not in curated dataset, skipping'.format(burst_name))
            continue 

        process_ok = True

        try:
            # process DNG reference frame (or merged frame) into sRGB-gamma JPG image
            # and save in `input-srgb-gamma` directory

            if use_merged:
                # merged frame
                inp_frame_fn = os.path.join(results_dir, burst_name, 'merged.dng')
                lsc_fn = None
            else:
                # reference frame
                ref_frame_idx_fn = os.path.join(results_dir, burst_name, 'reference_frame.txt')
                with open(ref_frame_idx_fn, 'r')as f:
                    ref_frame_idx = (int(f.read()))
                inp_frame_fn = os.path.join(bursts_dir, burst_name, 'payload_N{:03d}.dng'.format(ref_frame_idx))
                lsc_fn = os.path.join(bursts_dir, burst_name, 'lens_shading_map_N{:03d}.tiff'.format(ref_frame_idx))

            # copy final JPG image into `gt` directory
            final_jpg_fn = os.path.join(results_dir, burst_name, 'final.jpg')
            gt_image = cv2.imread(final_jpg_fn, cv2.IMREAD_UNCHANGED)
            gt_fn = os.path.join(gt_dir, burst_name + '.jpg')

            # final image size
            final_size = gt_image.shape[:2]
            # discard orientation; make sure height < width (most sensors)
            if final_size[0] > final_size[1]:
                final_size = [final_size[1], final_size[0]]

            metadata = get_metadata(inp_frame_fn)

            save_fn = os.path.join(input_dir, burst_name + '.jpg')
            input_image, metadata = process_hdrplus_frame(dng_fn=inp_frame_fn, metadata=metadata, lsc_fn=lsc_fn)

            # check default cropping
            input_image = check_default_cropping(input_image, final_size, metadata)

            # fix orientation
            input_image = fix_orientation(input_image, metadata['orientation'])

            input_image = (input_image * (2 ** 8 - 1)).astype('uint8')[..., ::-1]

            # check sizes
            if gt_image.shape != input_image.shape:
                process_ok = False
        except:
            process_ok = False
            input_image = gt_image = metadata = save_fn = gt_fn = None
            print('Error! Skipping!')

        if process_ok:
            # collect some info for inspection
            make = metadata['make']
            model = metadata['model']
            srgb_size = str(input_image.shape).replace(',', 'x')
            final_size = str(gt_image.shape).replace(',', 'x')
            size_mismatch = input_image.shape != gt_image.shape
            dy = input_image.shape[0] - gt_image.shape[0]
            dx = input_image.shape[1] - gt_image.shape[1]
            stats_file.write('{},{},{},{},{},{},{},{}\n'.format(
                burst_name, make, model, srgb_size, final_size, size_mismatch, dy, dx))

            input_image = resize(input_image)
            gt_image = resize(gt_image)

            cv2.imwrite(save_fn, input_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(gt_fn, gt_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    stats_file.close()


def check_active_area(inp_image, final_size, metadata):
    active_area = metadata['active_area']  # top, right, height, width

    if (active_area[2] - active_area[0]) >= final_size[0] and (active_area[3] - active_area[1]) >= final_size[1]:
        inp_image = active_area_cropping(inp_image, active_area)
    return inp_image


def check_default_cropping(inp_image, final_size, metadata):
    default_crop_origin = metadata['default_crop_origin']
    default_crop_size = metadata['default_crop_size']

    if type(default_crop_origin[0]) is Fraction:
        default_crop_origin = [float(x.numerator) / float(x.denominator) for x in default_crop_origin]
    if type(default_crop_size[0]) is Fraction:
        default_crop_size = [float(x.numerator) / float(x.denominator) for x in default_crop_size]

    if default_crop_size[1] == final_size[0] and default_crop_size[0] == final_size[1]:
        inp_image = default_cropping(inp_image, default_crop_origin, default_crop_size)

    return inp_image


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--hdrplus_dir', type=str, help='HDR+ directory containing bursts and results.'
                                                            '(e.g., `./hdrplus/20171106_subset`')
    arg_parser.add_argument('--out_dir', type=str, help='Output directory to save inputs and GTs.')
    arg_parser.add_argument('--im_list_fn', type=str, help='File containing the names of images included in the dataset.',
                            default='./prepare/data/hdrp_burst_names_no_crop_mismatch.txt')
    args_ = arg_parser.parse_args()
    return args_


if __name__ == '__main__':
    args1 = parse_args()
    prep_hdrplus(hdrplus_dir=args1.hdrplus_dir, out_dir=args1.out_dir, im_list_fn=args1.im_list_fn)

    """
    Example command:
    python3 -m prepare.prep_hdrplus --hdrplus_dir /home/user/Data/20171106_subset --out_dir /home/user/Data/hdrplus
    """
