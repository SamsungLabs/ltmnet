from tensorflow.keras import layers
from tensorflow.keras.models import Model
from models.base_models import BaseModel
from models.tone_curve_net import get_layers
from models.ltmnet_helpers import post_process_residual
import tensorflow as tf


def ResidualNet(num_layers=5, num_filters=32):
    # Adapted from DnCNN

    image_channels = 3
    input = layers.Input(shape=(None, None, image_channels))
    x = layers.Conv2D(num_filters, 3, kernel_initializer='Orthogonal', padding='same', activation='relu')(input)

    for i in range(num_layers - 2):
        x = layers.Conv2D(num_filters, 3, kernel_initializer='Orthogonal', padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001)(x)  # normalize across channels
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(image_channels, 3, kernel_initializer='Orthogonal', padding='same', use_bias=False,
                      activation='tanh')(x)

    model = Model(inputs=input, outputs=x)
    return model


class ToneCurveNetResConv(tf.keras.Model):
    def __init__(self, grid_size, control_points, input_size, curves):
        super(ToneCurveNetResConv, self).__init__()

        self.curves = curves
        self.grid_size = grid_size
        self.input_size = input_size
        self.control_points = control_points
        self.num_tiles = grid_size[0] * grid_size[1]

        self.model_layers = get_layers(grid_size, control_points, curves)

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)

        tone_curves = tf.reshape(x, [tf.shape(x)[0], self.num_tiles, self.curves, self.control_points])
        return tone_curves


class LTMNetResBase(BaseModel):
    def __init__(self, grid_size, control_points, input_size, curves, l1_weight=1.0, pix_weight=None,
                 tone_curve_net=None, residual_net=None, residual_weight=0.0, vgg_weight=1e-5, vgg=None):
        super(LTMNetResBase, self).__init__(grid_size, control_points, input_size, curves, l1_weight, pix_weight,
                                            vgg_weight, vgg)
        self.tone_curve_net = tone_curve_net
        self.residual_net = residual_net
        self.residual_weight = residual_weight

    def call(self, inputs, training=None, mask=None):
        return self.tone_curve_net(inputs, training=training)

    def train_step(self, data):
        in_im, gt_im, in_im_int = data
        with tf.GradientTape() as tape:
            pred_im, tc_nondec, r = post_process_residual(in_im, in_im_int, self.grid_size, self.curves, self,
                                                          self.residual_net, training=True, clip=False)
            l1_loss, vgg_loss = self.loss_fn(pred_im, gt_im, self.weight_map, residual=r,
                                             residual_weight=self.residual_weight, vgg=self.vgg)
            total_loss = self.l1_weight * l1_loss + self.vgg_weight * vgg_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(
            gradients,
            self.trainable_variables
        ))

        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'vgg_loss': vgg_loss,
        }

    def test_step(self, data):
        in_im, gt_im, in_im_int = data

        # Compute predictions
        pred_im, tc_nondec, r = post_process_residual(in_im, in_im_int, self.grid_size, self.curves, self,
                                                      self.residual_net, training=False, clip=True)

        # Compute loss
        l1_loss, vgg_loss = self.loss_fn(pred_im, gt_im, self.weight_map, residual=r,
                                         residual_weight=self.residual_weight, vgg=self.vgg)
        total_loss = self.l1_weight * l1_loss + self.vgg_weight * vgg_loss

        # Update the metrics.
        for m in self.cust_metrics:
            m.update_state(gt_im, pred_im)

        ret = {m.name: m.result() for m in self.metrics}
        ret['total_loss'] = total_loss
        ret['l1_loss'] = l1_loss
        ret['vgg_loss'] = vgg_loss
        return ret


class LTMNetResConv(LTMNetResBase):
    def __init__(self, grid_size, control_points, input_size, curves, l1_weight=1.0, pix_weight=None,
                 residual_layers=5, residual_filters=32, residual_weight=1.0, vgg_weight=1e-5, vgg=None):
        tone_curve_net = ToneCurveNetResConv(grid_size, control_points, input_size, curves)
        residual_net = ResidualNet(residual_layers, residual_filters)

        super(LTMNetResConv, self).__init__(grid_size=grid_size, control_points=control_points, input_size=input_size,
                                            curves=curves, l1_weight=l1_weight, pix_weight=pix_weight,
                                            tone_curve_net=tone_curve_net, residual_net=residual_net,
                                            residual_weight=residual_weight, vgg_weight=vgg_weight, vgg=vgg)

