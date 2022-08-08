from abc import ABC
import tensorflow as tf
from tensorflow.keras import layers
import utils
from models.ltmnet_helpers import post_process


class BaseModel(tf.keras.Model, ABC):
    def __init__(self, grid_size, control_points, input_size, curves, l1_weight=1.0, pix_weight=None, vgg_weight=1e-5,
                 vgg=None):
        super(BaseModel, self).__init__()
        self.vgg = vgg
        if pix_weight is None:
            pix_weight = [1., 1., 1.]

        self.optimizer = None
        self.loss_fn = None
        self.cust_metrics = None
        self.grid_size = grid_size
        self.num_tiles = grid_size[0] * grid_size[1]
        self.control_points = control_points
        self.input_size = input_size
        self.curves = curves
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight
        weight_map = utils.get_weight_map(input_size, grid_size, pix_weight[0], pix_weight[1], pix_weight[2])
        self.weight_map = tf.expand_dims(weight_map, axis=-1)

    def model(self):
        x = layers.Input(shape=(self.input_size[0], self.input_size[1], 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def compile(self, optimizer, loss, cust_metrics):
        super(BaseModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss
        self.cust_metrics = cust_metrics

    @property
    def metrics(self):
        return self.cust_metrics


class LTMBaseModel(BaseModel):
    def __init__(self, grid_size, control_points, input_size, curves, l1_weight=1.0,
                 pix_weight=None, vgg_weight=1e-5, vgg=None):
        super(LTMBaseModel, self).__init__(grid_size, control_points, input_size, curves, l1_weight, pix_weight,
                                           vgg_weight, vgg)

    def train_step(self, data):
        in_im, gt_im, in_im_int = data
        with tf.GradientTape() as tape:
            tc = self(in_im, training=True)
            pred_im, tc_nondec = post_process(tc, in_im_int, self.grid_size, self.curves)
            l1_loss, vgg_loss = self.loss_fn(pred_im, gt_im, self.weight_map, vgg=self.vgg)
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
            'learning_rate': self.optimizer._decayed_lr('float32'),
        }

    def test_step(self, data):
        in_im, gt_im, in_im_int = data

        # Compute predictions
        tc = self(in_im, training=False)
        pred_im, tc_nondec = post_process(tc, in_im_int, self.grid_size, self.curves)

        # Compute loss
        l1_loss, vgg_loss = self.loss_fn(pred_im, gt_im, self.weight_map, vgg=self.vgg)
        total_loss = self.l1_weight * l1_loss + self.vgg_weight * vgg_loss

        # Update the metrics.
        for m in self.cust_metrics:
            m.update_state(gt_im, pred_im)

        ret = {m.name: m.result() for m in self.metrics}
        ret['total_loss'] = total_loss
        ret['l1_loss'] = l1_loss
        ret['vgg_loss'] = vgg_loss
        return ret
