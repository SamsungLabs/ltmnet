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
import pickle
from metrics import PSNRMean, SSIMMean
from losses import ltm_loss
import utils
from models.tone_curve_net import ToneCurveNetConv
from models.residual_net import LTMNetResConv
import os
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def get_model(args):
    model_args = {
        'grid_size': args.grid_size,
        'control_points': args.control_points,
        'input_size': args.resize_to,
        'curves': args.curves,
        'pix_weight': args.pix_weight,
    }

    if 'res' in args.model_architecture:
        model_args['residual_layers'] = args.residual_layers
        model_args['residual_filters'] = args.residual_filters
        model_args['residual_weight'] = args.residual_weight

    feat_layer = ['block1_conv1', 'block2_conv1']
    vgg = vgg_layers(feat_layer)
    model_args['vgg'] = vgg if args.vgg_weight != 0.0 else None
    model_args['vgg_weight'] = args.vgg_weight
    model_args['l1_weight'] = args.l1_weight

    if args.model_architecture == 'conv':
        model = ToneCurveNetConv(**model_args)
    elif args.model_architecture == 'res_conv':
        model = LTMNetResConv(**model_args)
    else:
        raise Exception('Model architecture not supported.')
    return model


def train(optimizer, train_ds, test_ds, args):
    # Checkpoint
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_dir,
        save_freq='epoch',
        monitor=args.ckpt_monitor,
        mode=args.ckpt_monitor_mode,
        save_best_only=True)

    # Tensorboard
    tb_dir = args.tb_log_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_dir,
        update_freq='batch',
        profile_batch=0)

    terminates_on_nan_callback = utils.CustomTerminateOnNaN()

    # Metrics
    metrics = [PSNRMean(), SSIMMean()]

    model = get_model(args)
    loss_fn = ltm_loss
    model.build(input_shape=(None, args.resize_to[0], args.resize_to[1], 3))
    model.compile(optimizer, loss_fn, metrics)

    # Save model architecture
    f = open(args.model_summary_fn, 'w')
    model.model().summary(print_fn=lambda arg: f.write(arg + '\n'))
    f.close()

    print('Starting model.fit()...')
    model.fit(train_ds,
              validation_data=test_ds,
              epochs=args.epochs,
              verbose=1,
              workers=2,
              use_multiprocessing=True,
              callbacks=[ckpt_callback, tensorboard_callback, terminates_on_nan_callback])

    # Save model history
    pickle.dump(model.history.history, open(args.train_hist_fn, 'wb'))

    # Save residual model architecture
    if 'res' in args.model_architecture:
        f = open(args.model_summary_fn, 'a')
        model.residual_net.summary(print_fn=lambda arg: f.write(arg + '\n'))
        f.close()
