"""
The segmentation dataset is created with these chest x-rays datasets:
    - Montgomery County and Shenzhen Hospital: The Montgomery County dataset includes manually segmented lung masks.
                        https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/
    - Shenzhen Hospital dataset: manually segmented by Stirenko et al. The lung segmentation masks were dilated to load
                        lung boundary information within the training net and the images were resized to 512x512 pixels.
                        https://arxiv.org/abs/1803.01199
    - JSRT database: 154 nodule and 93 non-nodule images
                        http://db.jsrt.or.jp/eng.php
"""
import warnings
import albumentations as alb
import io
import cv2
import numpy as np
import os
from skimage import io, filters, exposure
from glob import glob
from tqdm.auto import tqdm
import argparse
from .ImageDataAugmentor.image_data_augmentator import ImageDataAugmentor
from .utils import preprocess, load_image, add_suffix

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join("..", "data")
SEGMENTATION_SOURCE_DIR = os.path.join(OUTPUT_DIR, "sources")

default_config = {
    'image_save_prefix': 'image',
    'mask_save_prefix': 'mask',
    'output_folder': OUTPUT_DIR,
    'source_dir': SEGMENTATION_SOURCE_DIR,

    'target_size': 512,
    'seed': 23,
    'batch_size': 8,
    'enable_preprocessing': False,

    'augm_p_corrections': 0.3,
    'augm_p_contrast': 1,
    'augm_p_gamma': 1,
    'augm_p_brightness': 1,

    'augm_p_blurs': 0.5,
    'augm_p_blur': 1,
    'augm_p_motionblur': 0.5,
    'augm_p_medianblur': 0.5,

    'augm_p_distortions': 0.3,
    'augm_p_elastic': 0.1,
    'augm_p_grid': 0.1,
    'augm_p_optical': 0.1,

    'augm_p_shiftscalrot': 0.5,
    'augm_shift_limit': 0.2,
    'augm_scale_limit': 0.1,
    'augm_rotate_limit': 20,
}


def train_generator(
        train_path,
        config=default_config
):
    def adjust_data(mask):
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.expand_dims(mask, axis=-1).astype('float')
        return mask

    AUGM = alb.Compose([
        alb.HorizontalFlip(),
        alb.OneOf([
            alb.RandomContrast(p=config["augm_p_contrast"]),
            alb.RandomGamma(p=config["augm_p_gamma"]),
            alb.RandomBrightness(p=config["augm_p_brightness"]),
        ],
            p=config["augm_p_corrections"]),
        alb.OneOf([
            alb.Blur(blur_limit=4, p=config["augm_p_blur"]),
            alb.MotionBlur(blur_limit=4, p=config["augm_p_motionblur"]),
            alb.MedianBlur(blur_limit=4, p=config["augm_p_medianblur"])
        ],
            p=config["augm_p_blurs"]),
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

    def image_preprocess(img):
        if config["enable_preprocessing"]:
            img = preprocess(img)
        return np.expand_dims(img, axis=-1)

    image_datagen = ImageDataAugmentor(
        rescale=1. / 255,
        augment=AUGM,
        preprocess_input=image_preprocess,
        augment_seed=config["seed"])

    mask_datagen = ImageDataAugmentor(augment=AUGM,
                                      preprocess_input=adjust_data,
                                      augment_seed=config["seed"],
                                      augment_mode="mask")

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[config["image_save_prefix"]],
        class_mode=None,
        color_mode='gray',
        target_size=(config["target_size"], config["target_size"]),
        batch_size=config["batch_size"],
        save_to_dir=False,
        seed=config["seed"])

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[config["mask_save_prefix"]],
        class_mode=None,
        color_mode='gray',
        target_size=(config["target_size"], config["target_size"]),
        batch_size=config["batch_size"],
        save_to_dir=False,
        seed=config["seed"])

    return zip(image_generator, mask_generator)


def val_generator(test_files, config=default_config):
    """
    generate the validation set
    :param test_files:
    :param config:
    :return:
    """
    X = []
    y = []
    for test_file in test_files:
        img = load_image(test_file, config["target_size"])[:, :, 0]
        if config["enable_preprocessing"]:
            img = preprocess(img)
        X.append(img)
        y.append(
            load_image(add_suffix(test_file, "mask"),
                       target_size=config["target_size"]))
    return np.array(X).reshape((-1, config["target_size"], config["target_size"], 1)), \
           np.array(y).reshape((-1, config["target_size"], config["target_size"], 1))


def get_data(config=default_config):
    SEGMENTATION_DIR = os.path.join(config["output_folder"], "segmentation")
    SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
    SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
    SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
    SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")

    preapre_dataset(input_folder=config["source_dir"],
                    output_folder=config["output_folder"],
                    image_size=config["target_size"],
                    force_reset=False)

    si = glob(os.path.join(SEGMENTATION_IMAGE_DIR, '*'))
    sm = glob(os.path.join(SEGMENTATION_MASK_DIR, '*'))
    if len(si) == len(sm) and len(si) > 0:
        tr = train_generator(SEGMENTATION_TRAIN_DIR, config=config)
        val = val_generator(SEGMENTATION_TEST_DIR, config=config)
        return tr, val
    else:
        print("Something goes wrong. I cannot find the dataset.")
        raise FileNotFoundError


def prepare_jsrt(j_image_dir, j_train_left_mask_dir, j_train_right_mask_dir,
                 j_test_left_mask_dir, j_test_right_mask_dir, segmentation_image_dir,
                 segmentation_mask_dir, segmentation_test_dir, image_size=512):
    jsrt_train = glob(os.path.join(j_train_left_mask_dir, '*.gif'))
    jsrt_test = glob(os.path.join(j_test_left_mask_dir, '*.gif'))

    for left_image_file in tqdm(jsrt_train):
        base_file = os.path.basename(left_image_file)
        image_file = os.path.join(j_image_dir, base_file[:-3] + 'IMG')
        right_image_file = os.path.join(j_train_right_mask_dir, base_file)

        image = 1.0 - np.fromfile(image_file, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        image = exposure.equalize_hist(image)
        image = (255 * image).astype(np.uint8)
        left_mask = io.imread(left_image_file)
        right_mask = io.imread(right_image_file)

        image = cv2.resize(image, (image_size, image_size))
        left_mask = cv2.resize(left_mask, (image_size, image_size))
        right_mask = cv2.resize(right_mask, (image_size, image_size))

        mask = np.maximum(left_mask, right_mask)

        cv2.imwrite(os.path.join(segmentation_image_dir, base_file[:-3] + 'png'), \
                    image)
        cv2.imwrite(os.path.join(segmentation_mask_dir, base_file[:-3] + 'png'), \
                    mask)

    for left_image_file in tqdm(jsrt_test):
        base_file = os.path.basename(left_image_file)
        image_file = os.path.join(j_image_dir, base_file[:-3] + 'IMG')
        right_image_file = os.path.join(j_test_right_mask_dir, base_file)

        image = 1.0 - np.fromfile(image_file, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        image = exposure.equalize_hist(image)
        image = (255 * image).astype(np.uint8)
        left_mask = io.imread(left_image_file)
        right_mask = io.imread(right_image_file)

        image = cv2.resize(image, (image_size, image_size))
        left_mask = cv2.resize(left_mask, (image_size, image_size))
        right_mask = cv2.resize(right_mask, (image_size, image_size))

        mask = np.maximum(left_mask, right_mask)

        filename, fileext = os.path.splitext(base_file[:-3] + 'png')
        cv2.imwrite(os.path.join(segmentation_test_dir, base_file[:-3] + 'png'), \
                    image)
        cv2.imwrite(os.path.join(segmentation_test_dir, \
                                 "%s_mask%s" % (filename, fileext)), mask)


def prepare_shenzhen(s_image_dir, s_mask_dir, segmentation_image_dir, segmentation_mask_dir,
                     segmentation_test_dir, image_size=512):
    shenzhen_mask_dir = glob(os.path.join(s_mask_dir, '*.png'))
    shenzhen_test = shenzhen_mask_dir[0:50]
    shenzhen_train = shenzhen_mask_dir[50:]

    for mask_file in tqdm(shenzhen_mask_dir):
        base_file = os.path.basename(mask_file).replace("_mask", "")
        image_file = os.path.join(s_image_dir, base_file)

        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))

        if (mask_file in shenzhen_train):
            cv2.imwrite(os.path.join(segmentation_image_dir, base_file), \
                        image)
            cv2.imwrite(os.path.join(segmentation_mask_dir, base_file), \
                        mask)
        else:
            filename, fileext = os.path.splitext(base_file)

            cv2.imwrite(os.path.join(segmentation_test_dir, base_file), \
                        image)
            cv2.imwrite(os.path.join(segmentation_test_dir, \
                                     "%s_mask%s" % (filename, fileext)), mask)


def prepare_montgomery(m_image_dir, m_left_mask_dir, m_right_mask_dir,
                       segmentation_image_dir, segmentation_mask_dir,
                       segmentation_test_dir, image_size=512):
    montgomery_left_mask_dir = glob(os.path.join(m_left_mask_dir, '*.png'))
    montgomery_test = montgomery_left_mask_dir[0:50]
    montgomery_train = montgomery_left_mask_dir[50:]

    for left_image_file in tqdm(montgomery_left_mask_dir):
        base_file = os.path.basename(left_image_file)
        image_file = os.path.join(m_image_dir, base_file)
        right_image_file = os.path.join(m_right_mask_dir, base_file)

        image = cv2.imread(image_file)
        left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
        right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (image_size, image_size))
        left_mask = cv2.resize(left_mask, (image_size, image_size))
        right_mask = cv2.resize(right_mask, (image_size, image_size))

        mask = np.maximum(left_mask, right_mask)

        if left_image_file in montgomery_train:
            cv2.imwrite(os.path.join(segmentation_image_dir, base_file), \
                        image)
            cv2.imwrite(os.path.join(segmentation_mask_dir, base_file), \
                        mask)
        else:
            filename, fileext = os.path.splitext(base_file)
            cv2.imwrite(os.path.join(segmentation_test_dir, base_file), \
                        image)
            cv2.imwrite(os.path.join(segmentation_test_dir, \
                                     "%s_mask%s" % (filename, fileext)), mask)


def preapre_dataset(input_folder=SEGMENTATION_SOURCE_DIR, output_folder=OUTPUT_DIR, image_size=None,
                    force_reset=False):
    SEGMENTATION_DIR = os.path.join(output_folder, "segmentation")
    SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
    SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
    SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
    SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")

    if force_reset:
        print("Cleaning dataset...")
        os.rmdir(SEGMENTATION_IMAGE_DIR)
        os.rmdir(SEGMENTATION_MASK_DIR)
        os.rmdir(SEGMENTATION_TEST_DIR)

    # Create folder structure
    os.makedirs(SEGMENTATION_IMAGE_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_MASK_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_TEST_DIR, exist_ok=True)

    si = glob(os.path.join(SEGMENTATION_IMAGE_DIR, '*'))
    sm = glob(os.path.join(SEGMENTATION_MASK_DIR, '*'))
    if len(si) == len(sm) and len(si) > 0:
        print(f"Segmentation images and mask already extracted ({len(si)} items)")
        return

    MONTGOMERY_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "MontgomerySet")
    MONTGOMERY_IMAGE_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "CXR_png")
    MONTGOMERY_LEFT_MASK_DIR = os.path.join(
        MONTGOMERY_TRAIN_DIR, "ManualMask", "leftMask")
    MONTGOMERY_RIGHT_MASK_DIR = os.path.join(
        MONTGOMERY_TRAIN_DIR, "ManualMask", "rightMask")
    if len(glob(os.path.join(MONTGOMERY_IMAGE_DIR, '*'))) > 0:
        prepare_montgomery(m_image_dir=MONTGOMERY_IMAGE_DIR,
                           m_left_mask_dir=MONTGOMERY_LEFT_MASK_DIR,
                           m_right_mask_dir=MONTGOMERY_RIGHT_MASK_DIR,
                           segmentation_image_dir=SEGMENTATION_IMAGE_DIR,
                           segmentation_mask_dir=SEGMENTATION_MASK_DIR,
                           segmentation_test_dir=SEGMENTATION_TEST_DIR,
                           image_size=image_size)
    else:
        print(
            f"The Montgomery dataset is not present, or in a different position. {MONTGOMERY_IMAGE_DIR} is empty or not exist.")

    SHENZHEN_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "ChinaSet_AllFiles")
    SHENZHEN_IMAGE_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "CXR_png")
    SHENZHEN_MASK_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "mask")
    if len(glob(os.path.join(SHENZHEN_IMAGE_DIR, '*'))) > 0:
        prepare_shenzhen(SHENZHEN_IMAGE_DIR,
                         SHENZHEN_MASK_DIR,
                         segmentation_image_dir=SEGMENTATION_IMAGE_DIR,
                         segmentation_mask_dir=SEGMENTATION_MASK_DIR,
                         segmentation_test_dir=SEGMENTATION_TEST_DIR,
                         image_size=image_size)
    else:
        print(
            f"The Shenzhen dataset is not present, or in a different position. {SHENZHEN_IMAGE_DIR} is empty or not exist.")

    JSRT_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "JSRT")
    JSRT_IMAGE_DIR = os.path.join(JSRT_DIR, "images")
    JSRT_TRAIN_LEFT_MASK_DIR = os.path.join(
        JSRT_DIR, "annotations", "fold1", "masks", "left lung")
    JSRT_TRAIN_RIGHT_MASK_DIR = os.path.join(
        JSRT_DIR, "annotations", "fold1", "masks", "right lung")
    JSRT_TEST_LEFT_MASK_DIR = os.path.join(
        JSRT_DIR, "annotations", "fold2", "masks", "left lung")
    JSRT_TEST_RIGHT_MASK_DIR = os.path.join(
        JSRT_DIR, "annotations", "fold2", "masks", "right lung")
    if len(glob(os.path.join(JSRT_IMAGE_DIR, '*'))) > 0:
        prepare_jsrt(JSRT_IMAGE_DIR,
                     JSRT_TRAIN_LEFT_MASK_DIR,
                     JSRT_TRAIN_RIGHT_MASK_DIR,
                     JSRT_TEST_LEFT_MASK_DIR,
                     JSRT_TEST_RIGHT_MASK_DIR,
                     segmentation_image_dir=SEGMENTATION_IMAGE_DIR,
                     segmentation_mask_dir=SEGMENTATION_MASK_DIR,
                     segmentation_test_dir=SEGMENTATION_TEST_DIR,
                     image_size=image_size)
    else:
        print(f"The JSRT dataset is not present, or in a different position. {JSRT_IMAGE_DIR} is empty or not exist.")


def get_arguments():
    parser = argparse.ArgumentParser(description='Segmentation dataset')
    parser.add_argument('--input_folder', type=str, default=SEGMENTATION_SOURCE_DIR,
                        help='Folder where the raw datasets are unzipped.')
    parser.add_argument('--output_folder', type=str, default=OUTPUT_DIR,
                        help='Where to place the created segmentation dataset.')
    parser.add_argument('--image_size', type=int, help='Image and mask size.', default=512)
    parser.add_argument('--force_reset', help='Clear dataset folder and rebuild it.', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    preapre_dataset(args.input_folder, args.output_folder, args.image_size, args.force_reset)
    print("Done!")
