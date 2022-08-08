"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (abdoukamel@gmail.com)

Camera pipeline utilities.

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
import exifread
import rawpy
from fractions import Fraction
from exifread.utils import Ratio
from .exif_utils import parse_exif, get_tag_values_from_ifds
from .dng_opcode import parse_opcode_lists


def get_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image.copy()
    return raw_image

    
def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    return raw_image


def save_image_stage(image, out_path, stage, save_as, bit_depth=8, stage_id=''):
    if bit_depth != 8 and save_as == 'jpg':
        save_as = 'png'

    if os.path.isdir(out_path):
        output_image_path = os.path.join(out_path, 'merged_{}_{}.{}'.format(stage_id, stage, save_as))
    else:
        out_path_no_ext, _ = os.path.splitext(out_path)
        output_image_path = '{}_{}_{}.{}'.format(out_path_no_ext, stage_id, stage, save_as)

    output_image = (image * (2 ** bit_depth - 1)).astype('uint' + str(bit_depth))

    if len(output_image.shape) > 2:
        output_image = output_image[:, :, ::-1]
    if save_as == 'jpg':
        cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_image_path, output_image)


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['active_area'] = get_active_area(tags, ifds)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    metadata['iso'] = get_iso(ifds)
    metadata['exposure_time'] = get_exposure_time(ifds)
    metadata['default_crop_origin'] = get_default_crop_origin(ifds)
    metadata['default_crop_size'] = get_default_crop_size(ifds)
    metadata['black_level_delta_h'] = get_black_level_delta_h(ifds)
    metadata['black_level_delta_v'] = get_black_level_delta_v(ifds)
    metadata['make'] = get_make(ifds)
    metadata['model'] = get_model(ifds)
    metadata['digital_zoom_ratio'] = get_digital_zoom_ratio(ifds)
    metadata['default_user_crop'] = get_default_user_crop(ifds)
    # ...

    # opcode lists
    metadata['opcode_lists'] = parse_opcode_lists(ifds)

    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_active_area(tags, ifds):
    possible_keys = ['Image Tag 0xC68D', 'Image Tag 50829', 'ActiveArea', 'Image ActiveArea']
    return get_values(tags, possible_keys)


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712', 'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714', 'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717', 'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728', 'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_color_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721', 'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722', 'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    return color_matrix_1, color_matrix_2


def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041', 'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_iso(ifds):
    # 0x8827	34855
    return get_tag_values_from_ifds(34855, ifds)


def get_exposure_time(ifds):
    # 0x829a	33434
    exposure_time = get_tag_values_from_ifds(33434, ifds)[0]
    return float(exposure_time.numerator) / float(exposure_time.denominator)


def get_default_crop_origin(ifds):
    return get_tag_values_from_ifds(50719, ifds)


def get_default_crop_size(ifds):
    return get_tag_values_from_ifds(50720, ifds)


def get_black_level_delta_h(ifds):
    bldh = get_tag_values_from_ifds(50715, ifds)
    if bldh and type(bldh[0]) is Fraction:
        bldh = [float(bldh[i].numerator) / bldh[i].denominator for i in range(len(bldh))]
    return bldh


def get_black_level_delta_v(ifds):
    bldv = get_tag_values_from_ifds(50716, ifds)
    if bldv and type(bldv[0]) is Fraction:
        bldv = [float(bldv[i].numerator) / bldv[i].denominator for i in range(len(bldv))]
    return bldv


def get_make(ifds):
    make = get_tag_values_from_ifds(271, ifds)
    if make:
        make = b''.join(make[:-1]).decode()
    return make


def get_model(ifds):
    model = get_tag_values_from_ifds(272, ifds)
    if model:
        model = b''.join(model[:-1]).decode()
    return model


def get_digital_zoom_ratio(ifds):
    digital_zoom_ratio = get_tag_values_from_ifds(41988, ifds)
    return digital_zoom_ratio


def get_default_user_crop(ifds):
    default_user_crop = get_tag_values_from_ifds(51125, ifds)
    return default_user_crop


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def active_area_cropping(image, active_area):
    if active_area is not None and active_area != [0, 0, image.shape[0], image.shape[1]]:
        image = image[active_area[0]: active_area[2], active_area[1]:active_area[3]]

    return image


def default_cropping(image, default_crop_origin, default_crop_size):
    if default_crop_origin is not None and default_crop_size is not None:
        if type(default_crop_origin[0]) is Fraction:
            default_crop_origin = [float(x.numerator) / float(x.denominator) for x in default_crop_origin]
        if type(default_crop_size[0]) is Fraction:
            default_crop_size = [float(x.numerator) / float(x.denominator) for x in default_crop_size]
        if np.any([x != int(x) for x in default_crop_size]):
            raise ValueError('Default crop size is not integer, default_crop_size = {}'.format(default_crop_size))

        # when default_crop_origin and default_crop_size come in (H,W) rather than (W,H), flip the order
        if (default_crop_size[0] < default_crop_size[1] and image.shape[0] < image.shape[1]) or \
                (default_crop_size[0] > default_crop_size[1] and image.shape[0] > image.shape[1]):
            default_crop_size.reverse()
            default_crop_origin.reverse()

        # check if any elements in default crop origin or default crop size is not an integer
        if np.any([x != int(x) for x in default_crop_origin]):
            xs, ys = np.meshgrid(np.arange(default_crop_size[0]) + default_crop_origin[0],
                                 np.arange(default_crop_size[1]) + default_crop_origin[1])
            xs = xs.astype(np.float32)
            ys = ys.astype(np.float32)
            image = cv2.remap(image, xs, ys, cv2.INTER_LINEAR)
        else:
            image = image[int(default_crop_origin[1]):int(default_crop_origin[1] + default_crop_size[1]),
                    int(default_crop_origin[0]):int(default_crop_origin[0] + default_crop_size[0]), :]
    return image


def resize(image, target_size):
    """
    target_size: (width, height)
    """
    return cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_LINEAR)


def normalize(raw_image, black_level, white_level, black_level_delta_h=None, black_level_delta_v=None, clip=True):
    h, w = raw_image.shape[:2]

    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

    # add black level delta h, v
    if black_level_delta_h:
        black_level_mask += np.tile(np.array(black_level_delta_h)[np.newaxis, :], [h, 1])
    if black_level_delta_v:
        black_level_mask += np.tile(np.array(black_level_delta_v)[:, np.newaxis], [1, w])

    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    if clip:
        normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def white_balance(normalized_image, as_shot_neutral, cfa_pattern, clip=True):
    if len(normalized_image.shape) == 3:
        return white_balance_rgb(normalized_image, as_shot_neutral, clip)

    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
        idx_y = idx[0]
        idx_x = idx[1]
        white_balanced_image[idx_y::step2, idx_x::step2] = \
            normalized_image[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def white_balance_rgb(image_rgb, illuminant, clip=True):
    illuminant_ = illuminant.copy()
    if type(illuminant_[0]) is Ratio:
        illuminant_ = ratios2floats(illuminant_)
    illuminant_ = np.reshape(illuminant_, (1, 1, 3))
    white_balanced_image = image_rgb / illuminant_
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def lens_shading_correction(raw_image, gain_map_opcode, bayer_pattern, gain_map=None, clip=True):
    """
    Apply lens shading correction map.
    :param raw_image: Input normalized (in [0, 1]) raw image.
    :param gain_map_opcode: Gain map opcode.
    :param bayer_pattern: Bayer pattern (RGGB, GRBG, ...).
    :param gain_map: Optional gain map to replace gain_map_opcode. 1 or 4 channels in order: R, Gr, Gb, and B.
    :param clip: Whether to clip result image to [0, 1].
    :return: Image with gain map applied; lens shading corrected.
    """

    if gain_map is None and gain_map_opcode:
        gain_map = gain_map_opcode.data['map_gain_2d']

    # resize gain map, make it 4 channels, if needed
    gain_map = cv2.resize(gain_map, dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                          interpolation=cv2.INTER_LINEAR)
    if len(gain_map.shape) == 2:
        gain_map = np.tile(gain_map[..., np.newaxis], [1, 1, 4])

    if gain_map_opcode:
        # TODO: consider other parameters

        top = gain_map_opcode.data['top']
        left = gain_map_opcode.data['left']
        bottom = gain_map_opcode.data['bottom']
        right = gain_map_opcode.data['right']
        rp = gain_map_opcode.data['row_pitch']
        cp = gain_map_opcode.data['col_pitch']

        gm_w = right - left
        gm_h = bottom - top

        # gain_map = cv2.resize(gain_map, dsize=(gm_w, gm_h), interpolation=cv2.INTER_LINEAR)

        # TODO
        # if top > 0:
        #     pass
        # elif left > 0:
        #     left_col = gain_map[:, 0:1]
        #     rep_left_col = np.tile(left_col, [1, left])
        #     gain_map = np.concatenate([rep_left_col, gain_map], axis=1)
        # elif bottom < raw_image.shape[0]:
        #     pass
        # elif right < raw_image.shape[1]:
        #     pass

    result_image = raw_image.copy()

    # one channel
    # result_image[::rp, ::cp] *= gain_map[::rp, ::cp]

    # per bayer channel
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    # blue channel index --> 3
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    # second green channel index --> 2
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        result_image[i0::2, j0::2] *= gain_map[:, :, bayer_pattern_idx[c]]

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)

    return result_image


def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag


def demosaic(bayer_image, cfa_pattern, output_channel_order='RGB', alg_type='EA', clip=True):
    """
    Demosaic a Bayer image.
    :param bayer_image: Image in Bayer format, single channel.
    :param cfa_pattern: Bayer/CFA pattern.
    :param output_channel_order: Either RGB or BGR.
    :param alg_type: algorithm type. options: '', 'EA' for edge-aware
    :param clip: Whether to clip values to [0, 1].
    :return: Demosaiced image.
    """
    max_val = 16383
    wb_image = (bayer_image * max_val).astype(dtype=np.uint16)

    if alg_type == 'EA':
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    else:
        raise ValueError('Unsupported demosaicing algorithm, alg_type = {}'.format(alg_type))

    demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

    if clip:
        demosaiced_image = np.clip(demosaiced_image, 0, 1)

    return demosaiced_image


def apply_color_space_transform(image, color_matrix_1, color_matrix_2):
    if type(color_matrix_1[0]) is Ratio:
        color_matrix_1 = ratios2floats(color_matrix_1)
    if type(color_matrix_2[0]) is Ratio:
        color_matrix_2 = ratios2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # inverse
    cam2xyz1 = np.linalg.inv(xyz2cam1)
    cam2xyz2 = np.linalg.inv(xyz2cam2)
    # for now, use one matrix  # TODO: interpolate btween both
    # simplified matrix multiplication
    xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_prophoto(xyz_image):
    xyz2pp = np.array([[1.3459433, -0.2556075, -0.0511118],
                       [-0.5445989, 1.5081673, 0.0205351],
                       [0.0000000, 0.0000000, 1.2118128]])
    xyz2pp = xyz2pp / np.sum(xyz2pp, axis=-1, keepdims=True)
    pp_image = np.matmul(xyz_image, xyz2pp.T)
    pp_image = np.clip(pp_image, 0.0, 1.0)
    return pp_image


def transform_prophoto_to_srgb(pp_image):
    # Solved from ppn2srgbn = xyz2srgbn @ xyz2ppn.inv
    pp2srgb = np.array([[1.84770997, -0.53597631, -0.31173367],
                        [-0.25511275, 1.24992018, 0.00519257],
                        [-0.0164456, -0.14912261, 1.1655682]])
    srgb_image = np.matmul(pp_image, pp2srgb.T)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])

    # xyz2srgb = np.linalg.inv(srgb2xyz)

    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    # normalize rows (needed?)
    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def apply_gamma(x):
    return x ** (1.0 / 2.2)


def process_to_save(image, out_dtype='uint8', channel_order='bgr'):
    """
    Process an RGB image to be saved with OpenCV.
    :param image: Input image.
    :param out_dtype: Target data type (e.g., 'uint8', 'uint16', ...).
    :param channel_order: Output channel order (e.g., 'bgr' for OpenCV, ...).
    :return: Processed image in the target data type and channel order.
    """
    in_dtype = str(image.dtype)

    if in_dtype != out_dtype:

        # normalize with source data type
        if in_dtype == 'uint8':
            image = image.astype('float32') / 255.0
        elif in_dtype == 'uint16':
            image = image.astype('float32') / 65535.0
        else:
            pass  # assuming float

        # quantize with target data type
        if out_dtype == 'uint8':
            max_val = 255
        elif out_dtype == 'uint16':
            max_val = 65535
        else:
            max_val = 255  # default
        image = (image * max_val).astype(out_dtype)

    # rearrange channel order, if needed
    if channel_order == 'bgr':
        image = image[:, :, ::-1]

    return image


def fix_missing_params(params):
    """
    Fix params dictionary by filling missing parameters with default values.
    :param params: Input params dictionary.
    :return: Fixed params dictionary.
    """
    params_fixed = params.copy()
    default_params = {
        'input_stage': 'raw',
        'output_stage': 'gamma',
        'save_as': 'jpg',
        'white_balancer': 'default',
        'demosaicer': '',
        'denoiser': 'fgs',
    }
    for key in default_params.keys():
        if key not in params_fixed:
            params_fixed[key] = default_params[key]
    return params_fixed