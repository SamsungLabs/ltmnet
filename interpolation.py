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

import tensorflow.experimental.numpy as np
np.experimental_enable_numpy_behavior()
import utils


def apply_ltm(image, tone_curve, num_curves):
    """
    Apply tone curve to an image (patch).
    :param image: (h, w, 3) if num_curves == 3, else (h, w)
    :param tone_curve: (num_curves, control_points)
    :param num_curves: 3 for 1 curve per channel, 1 for 1 curve for all channels.
    :return: tone-mapped image.
    """
    if num_curves == 3:
        r = tone_curve[0][image[..., 0]]
        g = tone_curve[1][image[..., 1]]
        b = tone_curve[2][image[..., 2]]
        new_image = np.stack((r, g, b), axis=-1)
    else:
        new_image = tone_curve[0][image]
    return new_image


def apply_gtm(image, tone_curve, num_curves):
    """
    Apply a single tone curve to an image.
    :param image: (h, w, 3) if num_curves == 3, else (h, w)
    :param tone_curve: (1, num_curves, control_points)
    :param num_curves: 3 for 1 curve per channel, 1 for 1 curve for all channels.
    :return: tone-mapped image.
    """
    tone_curve = tone_curve[0]
    out = apply_ltm(image, tone_curve, num_curves)
    return out


def apply_ltm_center(image, tone_curves, stats, num_curves):
    """
    Apply tone curves to the center region of an image.
    :param image: the original image.
    :param tone_curves: a list of all tone curves in row scan order.
    :return: interpolated center region of an image.
    """
    grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, meshgrids = stats
    xs_tl, ys_tl, xs_br, ys_br = meshgrids['center']

    # Get neighbourhoods
    neighbourhoods = []
    for y in range(margin_top, image.shape[0]-margin_bot, tile_height):
        for x in range(margin_left, image.shape[1]-margin_right, tile_width):
            neighbourhoods.append(image[y:y + tile_height, x:x + tile_width, :])

    assert len(neighbourhoods) == (grid_rows-1) * (grid_cols-1)

    # Get indices for all 4-tile neighbourhoods
    tile_ids = []
    for i in range(grid_rows - 1):
        for j in range(grid_cols - 1):
            start = i * grid_cols + j
            tile_ids.append([start, start + 1, start + grid_cols, start + grid_cols + 1])

    # Apply LTM and interpolate
    new_ns = []
    for i, n in enumerate(neighbourhoods):
        n_tile_ids = tile_ids[i]  # ids of the 4 tone curves (tiles) of the neighbourhood
        n_4versions = [apply_ltm(n, tone_curves[j], num_curves) for j in n_tile_ids]  # tl, tr, bl, br
        out = ys_br * xs_br * n_4versions[0] + ys_br * xs_tl * n_4versions[1] + ys_tl * xs_br * n_4versions[2] + ys_tl * xs_tl * n_4versions[3]
        out /= (tile_height-1) * (tile_width-1)

        new_ns.append(out)

    # Stack the interpolated neighbourhoods together
    rows = []
    for i in range(grid_rows - 1):
        cols = [new_ns[i * (grid_cols - 1) + j] for j in range(grid_cols - 1)]
        row = np.hstack(cols)
        rows.append(row)
    out = np.vstack(rows)
    return out


def apply_ltm_border(image, tone_curves, stats, num_curves=3):
    """
    Apply tone curves to the border, not including corner areas.
    :param image: the original image.
    :param tone_curves: a list of all tone curves in row scan order.
    :return: interpolated border regions of the image. In order of top, bottom, left, right.
    """
    grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, meshgrids = stats
    (top_xs_l, top_xs_r), (bot_xs_l, bot_xs_r), (left_ys_t, left_ys_b), (right_ys_t, right_ys_b) = meshgrids['border']

    # top, bottom, left, right neighbourhoods to be interpolated
    ntop = []
    nbot = []
    nleft = []
    nright = []

    for x in range(margin_left, image.shape[1] - margin_right, tile_width):
        ntop.append(image[:margin_top, x:x + tile_width, :])
        nbot.append(image[-margin_bot:, x:x + tile_width, :])

    for y in range(margin_top, image.shape[0] - margin_bot, tile_height):
        nleft.append(image[y:y + tile_height, :margin_left, :])
        nright.append(image[y:y + tile_height, -margin_right:, :])

    def apply_ltm_two_tiles(tc1, tc2, meshgrid1, meshgrid2, nbhd, interp_length, num_curves):
        """
        Apply tone curve to, and interpolate a two-tile neighbourhood, either horizontal or vertical
        :param tc1: left / top tone curves
        :param tc2: right / bottom tone curves
        :param meshgrid1: left / top meshgrids (leftmost / topmost positions are 0)
        :param meshgrid2: right / bottom meshgrids (rightmost / bottommost positions are 0)
        :param nbhd: neighbourhood to interpolate
        :param interp_length: normalizing factor of the meshgrid.
               Example: if xs = np.meshgrid(np.arange(10)), then interp_length = 9
        :return: interpolated neighbourhood
        """

        new_nbhd1 = apply_ltm(nbhd, tc1, num_curves)
        new_nbhd2 = apply_ltm(nbhd, tc2, num_curves)

        out = meshgrid1 * new_nbhd2 + meshgrid2 * new_nbhd1
        out /= interp_length
        return out

    new_ntop = [apply_ltm_two_tiles(tone_curves[i],  # left tone curve
                                    tone_curves[i + 1],  # right tone curve
                                    top_xs_l, top_xs_r,
                                    n, tile_width - 1, num_curves) for i, n in enumerate(ntop)]

    new_nbot = [apply_ltm_two_tiles(tone_curves[(grid_rows - 1) * grid_cols + i],  # left tone curve
                                    tone_curves[(grid_rows - 1) * grid_cols + i + 1],  # right tone curve
                                    bot_xs_l, bot_xs_r,
                                    n, tile_width - 1, num_curves) for i, n in enumerate(nbot)]

    new_nleft = [apply_ltm_two_tiles(tone_curves[i * grid_cols],  # top tone curve
                                     tone_curves[(i + 1) * grid_cols],  # bottom tone curve
                                     left_ys_t, left_ys_b,
                                     n, tile_height - 1, num_curves) for i, n in enumerate(nleft)]

    new_nright = [apply_ltm_two_tiles(tone_curves[(i + 1) * grid_cols - 1],  # top tone curve
                                      tone_curves[(i + 2) * grid_cols - 1],  # bottom tone curve
                                      right_ys_t, right_ys_b,
                                      n, tile_height - 1, num_curves) for i, n in enumerate(nright)]

    new_ntop = np.hstack(new_ntop)
    new_nbot = np.hstack(new_nbot)
    new_nleft = np.vstack(new_nleft)
    new_nright = np.vstack(new_nright)
    return new_ntop, new_nbot, new_nleft, new_nright


def apply_ltm_corner(image, tone_curves, stats, num_curves=3):
    """
    tone_curves: a list of all tone curves in row scan order.
    return: interpolated corner tiles in the order of top left, top right, bot left, bot right
    """
    grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, _ = stats

    corner_ids = [0, grid_cols - 1, -grid_rows, -1]
    tl_tile = image[:margin_top, :margin_left]
    tr_tile = image[:margin_top, -margin_right:]
    bl_tile = image[-margin_bot:, :margin_left]
    br_tile = image[-margin_bot:, -margin_right:]

    corner_tiles = [tl_tile, tr_tile, bl_tile, br_tile]
    corner_tcs = [tone_curves[i] for i in corner_ids]  # tcs: (grid_size, num_curves, control_points)
    new_tiles = [apply_ltm(corner_tiles[i], corner_tcs[i], num_curves) for i in range(len(corner_tcs))]
    return new_tiles[0], new_tiles[1], new_tiles[2], new_tiles[3]


def get_meshgrids(height, width):
    """
    Get two meshgrids of size (height, width). One with top left corner being (0, 0),
    the other with bottom right corner being (0, 0).
    :return: top left xs, ys, bottom right xs, ys
    """
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    # mesh grid for top left corner
    xs_tl = np.tile(np.abs(xs)[..., np.newaxis], 3)  # [0, 1, 2, ..., tile_width-1]
    ys_tl = np.tile(np.abs(ys)[..., np.newaxis], 3)
    # mesh grid for bottom right corner
    xs_br = np.tile(np.abs(xs - width + 1)[..., np.newaxis], 3)  # [-(tile_width-1), ..., -2, -1, 0]
    ys_br = np.tile(np.abs(ys - height + 1)[..., np.newaxis], 3)
    return xs_tl, ys_tl, xs_br, ys_br


def get_meshgrid_center(tile_height, tile_width):
    return get_meshgrids(tile_height, tile_width)


def get_meshgrid_border(tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right):
    """
    :return: meshgrids for the 4 border regions, in the order of top, bottom, left, right
    """
    # top
    top_xs_l, _, top_xs_r, _ = get_meshgrids(margin_top, tile_width)

    # bottom
    bot_xs_l, _, bot_xs_r, _ = get_meshgrids(margin_bot, tile_width)

    # left
    _, left_ys_t, _, left_ys_b = get_meshgrids(tile_height, margin_left)

    # right
    _, right_ys_t, _, right_ys_b = get_meshgrids(tile_height, margin_right)

    return (top_xs_l, top_xs_r), (bot_xs_l, bot_xs_r), (left_ys_t, left_ys_b), (right_ys_t, right_ys_b)


def get_image_stats(image, grid_size):
    """
    Information about the cropped image.
    :param image: the original image
    :return: grid size, tile size, sizes of the 4 margins, meshgrids.
    """

    grid_rows = grid_size[0]
    grid_cols = grid_size[1]

    tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right = utils.get_image_stats(image.shape,
                                                                                                       grid_size)

    meshgrid_center = get_meshgrid_center(tile_height, tile_width)
    meshgrid_border = get_meshgrid_border(tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right)

    meshgrids = {
        'center': meshgrid_center,
        'border': meshgrid_border
    }

    return grid_rows, grid_cols, tile_height, tile_width, margin_top, margin_left, margin_bot, margin_right, meshgrids


def do_interpolation(image, tone_curves, grid_size, num_curves=3):
    """
    Perform tone mapping and interpolation on an image.
    Center region: bilinear interpolation.
    Border region: linear interpolation.
    Corner region: no interpolation.
    :param num_curves: 3 -> 1 curve for each R,G,B channel, 1 -> 1 curve for all channels
    :param image: input int8
    :param tone_curves: (grid_size, num_curves, control_points)
    :param grid_size: (ncols, nrows)
    :return: image: float32, between [0-1]
    """
    if grid_size[0] == 1 and grid_size[1] == 1:
        return apply_gtm(image, tone_curves, num_curves).astype(np.float64)

    # get image statistics
    stats = get_image_stats(image, grid_size)

    # Center area:
    center = apply_ltm_center(image, tone_curves, stats, num_curves)

    # Border area:
    b_top, b_bot, b_left, b_right = apply_ltm_border(image, tone_curves, stats, num_curves)

    # Corner area:
    tlc, trc, blc, brc = apply_ltm_corner(image, tone_curves, stats, num_curves)

    # stack the corners, borders, and center together
    row_t = np.hstack([tlc, b_top, trc])
    row_c = np.hstack([b_left, center, b_right])
    row_b = np.hstack([blc, b_bot, brc])
    out = np.vstack([row_t, row_c, row_b])

    assert out.shape == image.shape
    return out
