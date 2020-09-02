from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import numpy as np
from typing import Tuple, Any, Sequence
from .utils import call_cascade

EPSILON = 1e-5


def handle_block_names(stage, cols):
    conv_name = 'decoder_stage{}-{}_conv'.format(stage, cols)
    bn_name = 'decoder_stage{}-{}_bn'.format(stage, cols)
    relu_name = 'decoder_stage{}-{}_relu'.format(stage, cols)
    up_name = 'decoder_stage{}-{}_upsample'.format(stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    return conv_name, bn_name, relu_name, up_name, merge_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not (use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x

    return layer


def Upsample2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                x = Concatenate(name=merge_name)([x] + skip)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x

    return layer


def Transpose2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2),
                      transpose_kernel_size=(4, 4), use_batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not (use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name + '1')(x)
        x = Activation('relu', name=relu_name + '1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x

    return layer


def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def K_meshgrid(x, y):
    return tf.meshgrid(x, y)


def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):
        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        transformations = K.reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)
        return interpolated_image


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.
    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
    From keras core
    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.
    # Returns
        data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                         for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data


def get_weights_from_hdf5_group(f):
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)

    weight_value_tuples = []
    for k, name in enumerate(filtered_layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        weight_value_tuples.append({'weights': weight_values})

    return weight_value_tuples


class Threshold(Layer):
    """
  It follows:
  ```
    f(x) = 1 for x > theta
    f(x) = 0 otherwise`
  ```
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as the input.
  Arguments:
    theta: Float >= 0 Threshold
  """

    def __init__(self, theta=1.0, **kwargs):
        super(Threshold, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)

    def call(self, inputs):
        theta = tf.cast(self.theta, inputs.dtype)
        return tf.cast(tf.greater(inputs, theta), inputs.dtype)

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(Thresholded, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


# based on https://github.com/Guillem96/efficientdet-tf/tree/84f353933faab2e5e12fb4917f739f1e9496b124
class ConvBlock(tf.keras.Model):

    def __init__(self,
                 features: int = None,
                 separable: bool = False,
                 activation: str = None,
                 **kwargs):
        super(ConvBlock, self).__init__()

        if separable:
            self.conv = tf.keras.layers.SeparableConv2D(filters=features,
                                                        **kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(features, **kwargs)
        self.bn = tf.keras.layers.BatchNormalization()

        if activation == 'swish':
            self.activation = tf.keras.layers.Activation(tf.nn.swish)
        elif activation is not None:
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = tf.keras.layers.Activation('linear')

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.bn(self.conv(x), training=training)
        return self.activation(x)


class Resize(tf.keras.Model):

    def __init__(self, features: int):
        super(Resize, self).__init__()
        self.antialiasing_conv = ConvBlock(features,
                                           separable=True,
                                           kernel_size=3,
                                           padding='same')

    def call(self,
             images: tf.Tensor,
             target_dim: Tuple[int, int, int, int] = None,
             training: bool = True) -> tf.Tensor:
        dims = target_dim[1:3]
        x = tf.image.resize(images, dims, method='nearest')
        x = self.antialiasing_conv(x, training=training)
        return x

def pool_rois(x, crop_size=None):
    x = tf.expand_dims(x, axis=0)
    if crop_size == None:
        crop_size = x.shape[1:3]

    boxes = [tf.convert_to_tensor([[0, 0, 0.4, 0.5]]),
             tf.convert_to_tensor([[0, 0.5, 0.4, 1]]),
             tf.convert_to_tensor([[0.3, 0, 0.7, 0.5]]),
             tf.convert_to_tensor([[0.3, 0.5, 0.7, 1]]),
             tf.convert_to_tensor([[0.6, 0, 1, 0.5]]),
             tf.convert_to_tensor([[0.6, 0.5, 1, 1]])
             ]
    box_indices = [0]

    out = []
    for b in boxes:
        car = tf.image.crop_and_resize(
            x, b, box_indices, crop_size, method='bilinear', extrapolation_value=0,
            name=None
        )
        car = tf.reshape(car, (*crop_size, x.shape[-1]))
        out.append(car)
    return tf.stack(out)


class UpsampleLike(tf.keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = target.shape
        if tf.keras.backend.image_data_format() == 'channels_first':
            source = tf.keras.backend.transpose(source, (0, 2, 3, 1))
            output = tf.image.resize(source, (target_shape[2], target_shape[3]), method='bilinear')
            output = tf.keras.backend.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return tf.image.resize(source, (target_shape[1], target_shape[2]), method='bilinear')

    def compute_output_shape(self, input_shape):
        print(input_shape)
        if tf.keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


def create_pyramid_features(in_features, feature_size=32):
    """ Creates the FPN layers on top of the backbone features.
    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.
    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """

    C1, *C_mid, C5 = in_features
    l = len(C_mid)
    C_mid.append(C5)

    P1 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced')(C1)
    P1_upsampled = UpsampleLike(name='P1_upsampled')([P1, C_mid[0]])
    P1_upsampled = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1')(
        P1_upsampled)
    P1_upsampled = tf.keras.layers.Activation(tf.nn.swish, name='P1_swish')(P1_upsampled)

    Pi_upsampled = []
    _Pi_upsampled = P1_upsampled
    for i in range(l):
        _Pi = tf.keras.layers.Conv2D(feature_size * 2 ** (i), kernel_size=1, strides=1, padding='same',
                                     name=f'Cmid{i}_reduced')(C_mid[i])
        _Pi = tf.keras.layers.Add(name=f'Pmid{i}_merged')([_Pi_upsampled, _Pi])
        _Pi_upsampled = UpsampleLike(name=f'Pmid{i}_upsampled')([_Pi, C_mid[i + 1]])
        _Pi_upsampled = tf.keras.layers.Conv2D(feature_size * 2 ** (i + 1), kernel_size=3, strides=1, padding='same',
                                               name=f'Pmid{i}')(_Pi_upsampled)
        _Pi_upsampled = tf.keras.layers.Activation(tf.nn.swish, name=f'Pmid{i}_swish')(_Pi_upsampled)
        Pi_upsampled.append(_Pi_upsampled)

    P5 = tf.keras.layers.Conv2D(feature_size * 2 ** (i + 1), kernel_size=1, strides=1, padding='same',
                                name='C5_reduced')(C5)
    P5 = tf.keras.layers.Add(name='P5_merged')([_Pi_upsampled, P5])
    P5 = tf.keras.layers.Conv2D(feature_size * 2 ** (i + 2), kernel_size=3, strides=1, padding='same', name='P5')(P5)
    P5 = tf.keras.layers.Activation(tf.nn.swish, name='P5_swish')(P5)

    return [P1_upsampled, *Pi_upsampled, P5]


class RetinaNetClassifier(tf.keras.Model):

    def __init__(self,
                 width: int,
                 depth: int,
                 opt=1):
        super(RetinaNetClassifier, self).__init__()

        self.width = width
        self.depth = depth
        self.opt = opt

        self.feature_extractors = [ConvBlock(width,
                                             kernel_size=3,
                                             activation='swish',
                                             padding='same')
                                   for _ in range(depth)]
        self.score_regressor = tf.keras.layers.Conv2D(opt,
                                                      kernel_size=3,
                                                      padding='same')

    def call(self, features: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = call_cascade(
            self.feature_extractors, features, training=training)
        return self.score_regressor(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'width': self.width,
            'depth': self.depth,
            'opt': self.opt
        })
        return config
