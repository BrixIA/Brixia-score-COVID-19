from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
import tensorflow as tf
import h5py
from .blocks import BilinearInterpolation, pool_rois, create_pyramid_features, get_weights_from_hdf5_group, \
    get_initial_weights, RetinaNetClassifier, Transpose2D_block, Upsample2D_block
from .utils import get_layer_number, to_tuple, call_cascade


def build_xnet(backbone, classes, skip_connection_layers,
               decoder_filters=(256, 128, 64, 32, 16),
               upsample_rates=(2, 2, 2, 2, 2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):
    input = backbone.input
    # print(n_upsample_blocks)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    if len(skip_connection_layers) > n_upsample_blocks:
        downsampling_layers = skip_connection_layers[int(len(skip_connection_layers) / 2):]
        skip_connection_layers = skip_connection_layers[:int(len(skip_connection_layers) / 2)]
    else:
        downsampling_layers = skip_connection_layers

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]
    downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                         for l in downsampling_layers])
    downsampling_list = [backbone.layers[downsampling_idx[i]].output for i in range(len(downsampling_idx))]
    downterm = [None] * (n_upsample_blocks + 1)
    for i in range(len(downsampling_idx)):
        # print(downsampling_list[0])
        # print(backbone.output)
        # print("")
        if downsampling_list[0].shape == backbone.output.shape:
            # print("VGG16 should be!")
            downterm[n_upsample_blocks - i] = downsampling_list[i]
        else:
            downterm[n_upsample_blocks - i - 1] = downsampling_list[i]
    downterm[-1] = backbone.output
    # print("downterm = {}".format(downterm))

    interm = [None] * (n_upsample_blocks + 1) * (n_upsample_blocks + 1)
    for i in range(len(skip_connection_idx)):
        interm[-i * (n_upsample_blocks + 1) + (n_upsample_blocks + 1) * (n_upsample_blocks - 1)] = skip_layers_list[i]
    interm[(n_upsample_blocks + 1) * n_upsample_blocks] = backbone.output

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks - j):
            upsample_rate = to_tuple(upsample_rates[i])

            if i == 0 and j < n_upsample_blocks - 1 and len(skip_connection_layers) < n_upsample_blocks:
                interm[(n_upsample_blocks + 1) * i + j + 1] = None
            elif j == 0:
                if downterm[i + 1] is not None:
                    interm[(n_upsample_blocks + 1) * i + j + 1] = up_block(decoder_filters[n_upsample_blocks - i - 2],
                                                                           i + 1, j + 1, upsample_rate=upsample_rate,
                                                                           skip=interm[(n_upsample_blocks + 1) * i + j],
                                                                           use_batchnorm=use_batchnorm)(downterm[i + 1])
                else:
                    interm[(n_upsample_blocks + 1) * i + j + 1] = None
            else:
                interm[(n_upsample_blocks + 1) * i + j + 1] = up_block(decoder_filters[n_upsample_blocks - i - 2],
                                                                       i + 1, j + 1, upsample_rate=upsample_rate,
                                                                       skip=interm[(n_upsample_blocks + 1) * i: (
                                                                                                                        n_upsample_blocks + 1) * i + j + 1],
                                                                       use_batchnorm=use_batchnorm)(
                    interm[(n_upsample_blocks + 1) * (i + 1) + j])

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(interm[n_upsample_blocks])
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model


class STN(tf.keras.layers.Layer):
    def __init__(self, in_shape: tuple, mask_resize: int = 128,
                 network_structure=((20, 5), (20, 5)),
                 dense_neurons=50, load_align_model=True,
                 align_model_weights=None, freeze_align_model=False):
        super(STN, self).__init__()

        assert not in_shape[1] % mask_resize, "The STN size must be a multiple of mask size"
        trainable = not freeze_align_model

        self.in_shape = in_shape
        self.mask_resize = mask_resize
        self.network_structure = network_structure
        self.dense_neurons = dense_neurons
        self.load_align_model = load_align_model
        self.align_model_weights = align_model_weights
        self.freeze_align_model = freeze_align_model

        weights = [{}] * (2 + len(network_structure))
        if load_align_model:
            print("Loading alignment model")
            if align_model_weights is None:
                raise ImportError('`load_weights` requires h5py.')
            with h5py.File(align_model_weights, mode='r') as f:
                if 'layer_names' not in f.attrs and 'model_weights' in f:
                    f = f['model_weights']
                    weights = get_weights_from_hdf5_group(f)
                if hasattr(f, 'close'):
                    f.close()
                elif hasattr(f.file, 'close'):
                    f.file.close()
        else:
            weights[-1] = {'weights': get_initial_weights(dense_neurons)}

        assert len(weights) == 2 + len(network_structure), "The weights do not match with the architectue"

        self.blocks = [MaxPool2D(pool_size=(in_shape[1] // mask_resize, in_shape[2] // mask_resize),
                                 name='locnet_input_adaptation')]

        for i, (n, f) in enumerate(network_structure):
            self.blocks.append(MaxPool2D(pool_size=(2, 2), name='locnet_pooling' + str(i)))
            self.blocks.append(Conv2D(n, (f, f), trainable=trainable, **weights[i], name='locnet_conv' + str(i)))
        self.blocks.append(Flatten(name='locnet_flatten'))
        self.blocks.append(Dense(dense_neurons, trainable=trainable, **weights[-2], name='locnet_dense'))
        self.blocks.append(Activation('relu', name='locnet_relu'))
        self.blocks.append(Dense(6, trainable=trainable, **weights[-1], name='locnet_alignment'))

    def call(self, input_tensor: tf.Tensor, training: bool = True):
        locnet = call_cascade(self.blocks,
                              input_tensor,
                              training=training)

        return locnet

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_shape': self.in_shape,
            'mask_resize': self.mask_resize,
            'network_structure': self.network_structure,
            'dense_neurons': self.dense_neurons,
            'load_align_model': self.load_align_model,
            'align_model_weights': self.align_model_weights,
            'freeze_align_model': self.freeze_align_model,
        })
        return config


def build_BScore(backbone,
                 skip_connection_layers=('stage1_unit1_relu1', 'stage2_unit2_relu1',
                                         'stage3_unit2_relu1', 'stage4_unit2_relu1'),
                 explict_self_attention=True,
                 pyramid_feature_size=64,
                 classes=4,
                 class_width=8,
                 class_depth=3
                 ):
    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]

    locnet_alignment = backbone.layers[get_layer_number(backbone, 'stn')]

    aligned_features = []
    for c in skip_layers_list:
        x = BilinearInterpolation(c.shape[1:3])([c, locnet_alignment.output])
        if explict_self_attention:
            x = tf.multiply(x, tf.image.resize(backbone.output, x.shape[1:3]))
        d = tf.reshape(x, shape=[-1] + c.shape[1:].as_list())
        aligned_features.append(d)

    pext = []
    for af in aligned_features:
        p = tf.keras.layers.Lambda(lambda x: tf.map_fn(pool_rois, x))(af)
        p = tf.reshape(p, (-1, *p.shape[2:]))
        pext.append(p)

    fpn_feats = create_pyramid_features([pext[0], pext[1], pext[2], pext[3]], feature_size=pyramid_feature_size)

    rn_class = []
    b = tf.reshape(fpn_feats[-1], (-1, 6, *fpn_feats[-1].shape[1:]))  # use the higher level one only
    for i in range(6):
        rn = RetinaNetClassifier(class_width, class_depth, classes)(b[:, i, :, :, :])
        rn = tf.keras.layers.GlobalAveragePooling2D()(rn)
        rn = tf.keras.layers.Activation('softmax')(rn)
        rn_class.append(rn)

    sact = tf.reshape(tf.stack(rn_class, axis=1), (-1, 3, 2, 4))
    model = Model(backbone.input, sact)
    return model
