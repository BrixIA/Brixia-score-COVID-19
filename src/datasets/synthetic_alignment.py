import warnings
import albumentations as alb
import cv2
from pathlib import Path
import numpy as np
from skimage.transform import rotate, resize
from skimage.measure import regionprops
from .ImageDataAugmentor.image_data_augmentator import ImageDataAugmentor
warnings.filterwarnings('ignore')

l_path = Path(__file__).parent
train_path = (l_path / '../../data/segmentation/train').resolve()
default_config = {
    'seed': 123,
    'target_size': 128,
    'batch_size': 64,
    'verbose': True,
    'split': 0.2,
    'train_path': train_path,

    # augmentation
    'augm_p_distortions': 0.3,
    'augm_p_elastic': 1,
    'augm_p_grid': 1,
    'augm_p_optical': 1,
    'augm_p_shiftscalrot': 0.5,
    'augm_shift_limit': 0.2,
    'augm_scale_limit': 0.1,
    'augm_rotate_limit': 30,
}


def get_data(config=default_config):
    """
    Get synthetic alignment image generators
    :param config: configuration parameters
    :return: (tuple) training generator, val generator
    """
    train_gen = train_generator(config=config)
    val_gen = val_generator(config=config)

    return train_gen, val_gen


mask_datagen_aug = None
mask_datagen = None


def train_generator(config):
    global mask_datagen_aug
    global mask_datagen

    def adjust_data(mask):
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.expand_dims(mask, axis=-1).astype('float')
        return mask

    def adjust_gt(mask):
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        regions = regionprops(mask)
        props = regions[0]
        orientation = props.orientation
        rimage = rotate(mask.astype('float'),
                        orientation * np.pi)
        bbox = regionprops(rimage.astype(np.int))[0].bbox
        add = 3
        y0 = np.clip(bbox[0] - add, 0, mask.shape[0])
        y1 = np.clip(bbox[2] + add, 0, mask.shape[0])
        x0 = np.clip(bbox[1] - add, 0, mask.shape[1])
        x1 = np.clip(bbox[3] + add, 0, mask.shape[1])

        outim = resize(rimage[y0:y1, x0:x1],
                       mask.shape)

        return np.expand_dims(outim, axis=-1).astype('float')

    augm = alb.Compose([
        alb.OneOf([
            alb.ElasticTransform(alpha=60,
                                 sigma=60 * 0.20,
                                 alpha_affine=60 * 0.03,
                                 p=config["augm_p_elastic"]),
            alb.GridDistortion(p=config["augm_p_grid"]),
            alb.OpticalDistortion(
                distort_limit=0.2, shift_limit=0.05, p=config["augm_p_optical"]),
        ],
            p=config["augm_p_distortions"]),
        alb.ShiftScaleRotate(shift_limit=config["augm_shift_limit"],
                             scale_limit=config["augm_scale_limit"],
                             rotate_limit=config["augm_rotate_limit"],
                             interpolation=cv2.INTER_LINEAR,
                             border_mode=cv2.BORDER_CONSTANT,
                             p=config["augm_p_shiftscalrot"]),
        alb.Resize(config["target_size"], config["target_size"]),
    ])

    augm_mask = alb.Resize(config["target_size"], config["target_size"])

    mask_prefix = 'mask'
    mask_datagen_aug = ImageDataAugmentor(augment=augm,
                                          preprocess_input=adjust_data,
                                          augment_seed=config["seed"],
                                          augment_mode=mask_prefix,
                                          validation_split=config["split"])

    mask_datagen = ImageDataAugmentor(augment=augm_mask,
                                      preprocess_input=adjust_gt,
                                      augment_seed=config["seed"],
                                      augment_mode=mask_prefix,
                                      validation_split=config["split"])

    image_generator = mask_datagen_aug.flow_from_directory(
        config["train_path"],
        classes=[mask_prefix],
        class_mode=None,
        color_mode='gray',
        target_size=(config["target_size"], config["target_size"]),
        batch_size=config["batch_size"],
        save_to_dir=False,
        seed=config["seed"],
        subset="training"
    )

    mask_generator = mask_datagen.flow_from_directory(
        config["train_path"],
        classes=[mask_prefix],
        class_mode=None,
        color_mode='gray',
        target_size=(config["target_size"], config["target_size"]),
        batch_size=config["batch_size"],
        save_to_dir=False,
        seed=config["seed"],
        subset="training"
    )

    return zip(image_generator, mask_generator)


def val_generator(config):
    global mask_datagen_aug
    global mask_datagen

    mask_prefix = 'mask'
    image_generator = mask_datagen_aug.flow_from_directory(
        config["train_path"],
        classes=[mask_prefix],
        class_mode=None,
        color_mode='gray',
        target_size=(config["target_size"], config["target_size"]),
        batch_size=config["batch_size"],
        save_to_dir=False,
        seed=config["seed"],
        subset="validation"
    )

    mask_generator = mask_datagen.flow_from_directory(
        config["train_path"],
        classes=[mask_prefix],
        class_mode=None,
        color_mode='gray',
        target_size=(config["target_size"], config["target_size"]),
        batch_size=config["batch_size"],
        save_to_dir=False,
        seed=config["seed"],
        subset="validation"
    )

    return zip(image_generator, mask_generator)
