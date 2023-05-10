import tensorflow as tf
from tensorflow.keras import layers
from models.base_models import LTMBaseModel


def get_layers(grid_size, control_points, curves):
    assert grid_size[0] == grid_size[1]
    num_layers = 6
    model_layers = []
    channels = 4
    activation = 'relu'
    for i in range(num_layers):
        model_layers.append(layers.Conv2D(channels, 3, padding='same', activation=activation))

        if i == num_layers - 2:
            if grid_size[0] == 8 or grid_size[0] == 1:
                pool_size = (2, 2)
            elif grid_size[0] == 4:
                pool_size = (4, 4)
            elif grid_size[0] == 2:
                pool_size = (8, 8)
            elif grid_size[0] == 16:
                pool_size = (1, 1)
            else:
                raise Exception('Invalid grid size: {}'.format(grid_size))
            model_layers.append(layers.MaxPooling2D(pool_size))
            channels = curves * control_points
            activation = 'sigmoid'
        else:
            model_layers.append(layers.MaxPooling2D((2, 2)))
            channels *= 2
    return model_layers


class ToneCurveNetConv(LTMBaseModel):
    def __init__(self, grid_size, control_points, input_size, curves, l1_weight=1.0, pix_weight=None, vgg_weight=1e-5,
                 vgg=None):
        super(ToneCurveNetConv, self).__init__(grid_size=grid_size,
                                               control_points=control_points,
                                               input_size=input_size,
                                               curves=curves,
                                               l1_weight=l1_weight,
                                               pix_weight=pix_weight,
                                               vgg_weight=vgg_weight,
                                               vgg=vgg)

        self.model_layers = get_layers(grid_size, control_points, curves)

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)

        tone_curves = tf.reshape(x, [tf.shape(x)[0], self.num_tiles, self.curves, self.control_points])
        return tone_curves


class ToneCurveNetConvGrid1x1(ToneCurveNetConv):
    def __init__(self, grid_size, control_points, input_size, curves, l1_weight=1.0, pix_weight=None,
                 vgg_weight=1e-5, vgg=None):
        super(ToneCurveNetConvGrid1x1, self).__init__(grid_size=grid_size,
                                                      control_points=control_points,
                                                      input_size=input_size,
                                                      curves=curves,
                                                      l1_weight=l1_weight,
                                                      pix_weight=pix_weight,
                                                      vgg_weight=vgg_weight,
                                                      vgg=vgg)

        model_layers = get_layers((8, 8), control_points, curves)
        model_layers.append(layers.GlobalAvgPool2D())
        self.model_layers = model_layers
