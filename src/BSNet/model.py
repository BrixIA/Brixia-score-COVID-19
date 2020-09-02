from .builder import build_xnet, STN, build_BScore
from .utils import freeze_model
from .blocks import BilinearInterpolation
from .backbones import get_backbone
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

DEFAULT_SKIP_CONNECTIONS = {
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2',
              'block5_pool', 'block4_pool', 'block3_pool', 'block2_pool', 'block1_pool',
              ),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2',
              'block5_pool', 'block4_pool', 'block3_pool', 'block2_pool', 'block1_pool',
              ),
    'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                 'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                 ),
    'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                 'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                 ),
    'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                 'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                 ),
    'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                  'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                  ),
    'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                  'relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                  ),
    'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                  'stage4_unit1_relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                  ),
    'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0',
                   'stage4_unit1_relu1', 'stage3_unit2_relu1', 'stage2_unit2_relu1', 'stage1_unit2_relu1',
                   ),
    'inceptionv3': (228, 86, 16, 9),
    'inceptionresnetv2': (594, 260, 16, 9),
    'densenet121': (311, 139, 51, 4),
    'densenet169': (367, 139, 51, 4),
    'densenet201': (479, 139, 51, 4),
}


def BSNet(backbone_name='resnet18',
          input_shape=(512, 512, 1),
          input_tensor=None,
          encoder_weights=None,
          freeze_encoder=True,
          skip_connections='default',
          decoder_block_type='transpose',
          decoder_filters=(256, 128, 64, 32, 16),
          decoder_use_batchnorm=True,
          n_upsample_blocks=5,
          upsample_rates=(2, 2, 2, 2, 2),
          classes=4,
          activation='sigmoid',
          load_seg_model=True,
          seg_model_weights='./weights/segmentation-model.h5',
          freeze_segmentation=True,
          load_align_model=True,
          align_model_weights='./weights/alignment-model.h5',
          freeze_align_model=True,
          pretrain_aligment_net=False,
          explict_self_attention=True,
          load_bscore_model=True,
          bscore_model_weights='./weights/bscore-model.h5'
          ):
    """

    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            'dof' (pre-training on DoF)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer
        load_seg_model: (bool) wheter to load the segmentation model weighes. A proper path in `seg_model_weights` must be setted
        seg_model_weights: (str) path to a proper model file
        freeze_segmentation: (bool) Set segmentation layers weights as non-trainable. Useful for fine-tuning
        load_align_model: (bool) wheter to load the alignment model weighes. A proper path in `align_model_weights` must be setted
        align_model_weights: (str) path to a proper model file
        freeze_align_model: (bool) Set alignment layers weights as non-trainable. Useful for fine-tuning
        pretrain_aligment_net: (bool) create a model with only the alignment branch active
        explict_self_attention: (bool) multiply the segmentation map
        load_bscore_model: (bool) wheter to load the BScore model weighes. A proper path in `bscore_model_weights` must be setted
        bscore_model_weights: (str) path to a proper model file

    Returns:
        keras.models.Model instance

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if skip_connections == 'default':
        skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    seg_model = build_xnet(backbone,
                           classes=1,
                           skip_connection_layers=skip_connections,
                           decoder_filters=decoder_filters,
                           block_type=decoder_block_type,
                           activation=activation,
                           n_upsample_blocks=n_upsample_blocks,
                           upsample_rates=upsample_rates,
                           use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    seg_model._name = 'x-{}'.format(backbone_name)

    if load_seg_model:
        print("Loading segmentation model")
        try:
            seg_model.load_weights(seg_model_weights)
        except ValueError as e:
            print(f"Loading a wrong weight checkpoint for segmentation model. {e}")

    # lock the segmentation network weights for fine-tuning
    if freeze_segmentation:
        freeze_model(seg_model)

    if pretrain_aligment_net:
        input_data = Input(shape=input_shape)
    else:
        input_data = seg_model.output

    locnet = STN(in_shape=input_data.shape,
                 mask_resize=128,
                 network_structure=((20, 5), (20, 5)),
                 dense_neurons=50,
                 load_align_model=load_align_model,
                 align_model_weights=align_model_weights,
                 freeze_align_model=freeze_align_model
                 )(input_data)

    x = BilinearInterpolation(input_shape[:2], name='alignmnet_segmentation')([seg_model.output, locnet])

    if pretrain_aligment_net:
        align_model = Model(input_data, x)
    else:
        align_model = Model(backbone.input, x)

    bscore_model = build_BScore(align_model,
                                skip_connection_layers=('stage1_unit1_relu1', 'stage2_unit2_relu1',
                                                        'stage3_unit2_relu1', 'stage4_unit2_relu1'),
                                explict_self_attention=explict_self_attention,
                                pyramid_feature_size=64,
                                classes=classes,
                                class_width=8,
                                class_depth=3
                                )

    if load_bscore_model:
        print("Loading BScore model")
        try:
            bscore_model.load_weights(bscore_model_weights)
        except ValueError as e:
            print(f"Loading a wrong weight checkpoint for BScore model. {e}")

    return seg_model, align_model, bscore_model
