import tensorflow as tf
from interpolation import do_interpolation


def enforce_constraint(tone_curves):
    """
    Enforce monotonically increasing constraint and
    between 0-1 constraint
    :param tone_curves: (bs, num_tiles, control_points)
    :return:
    """
    # Reconstruct by integration and normalization
    g_sum = tf.math.reduce_sum(tone_curves, axis=-1, keepdims=True)
    tone_curves = tf.math.cumsum(tone_curves, axis=-1) / g_sum
    return tone_curves


def post_process(tone_curves, in_im_int, grid_size, num_curves, constraint_tc=True):
    """
    :param tone_curves: (bs, grid_size, num_curves, control_points)
    :param in_im_int: int8 (bs, resized_height, resized_width)
    :param grid_size:
    :param constraint_tc: whether to enforce non-decreasing constraint on tone curves
    :return: interpolated images, tone curves
    """
    if constraint_tc:
        tone_curves = enforce_constraint(tone_curves)

    inputs = (in_im_int, tone_curves)
    interp_images = tf.map_fn(lambda x: do_interpolation(x[0], x[1], grid_size, num_curves=num_curves), inputs,
                              fn_output_signature=tf.float64)
    return interp_images, tone_curves


def post_process_residual(in_im, in_im_int, grid_size, num_curves, tc_net, residual_net, training=False, clip=True,
                          constraint_tc=True):
    """
    :param tone_curves: (bs, grid_size, num_curves, control_points)
    :param in_im_int: int8 (bs, resized_height, resized_width)
    :param gt_im: float32
    :return: interpolated images, non-decreasing tone curves
    """
    tone_curves = tc_net(in_im, training=training)
    if constraint_tc:
        tone_curves = enforce_constraint(tone_curves)

    inputs = (in_im_int, tone_curves)
    interp_images = tf.map_fn(lambda x: do_interpolation(x[0], x[1], grid_size, num_curves=num_curves), inputs,
                              fn_output_signature=tf.float64)
    r = residual_net(interp_images, training=training)
    interp_images = interp_images + r
    if clip:
        interp_images = tf.clip_by_value(interp_images, 0.0, 1.0)
    return interp_images, tone_curves, r



def predict_one(model, in_im, in_im_int, args):
    """
    :return:
        default: final predicted image, local tone curves,
        residual models: final predicted image, local tone curves, (residual)
        gtm_ltm models: final predicted image, local tone curves, (global tone curves, globally tone mapped image)
    """
    in_im = in_im[tf.newaxis, ...]
    in_im_int = in_im_int[tf.newaxis, ...]
    if 'res' in args.model_architecture:
        results = post_process_residual(in_im, in_im_int, args.grid_size, args.curves, model, model.residual_net,
                                        training=False, clip=True)
    else:
        tone_curves = model(in_im, training=False)
        results = post_process(tone_curves, in_im_int, args.grid_size, args.curves)
    results = [x[0] for x in results]  # drop the batch dimension
    return results
