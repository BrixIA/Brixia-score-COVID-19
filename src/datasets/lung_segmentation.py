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
from pathlib import Path
from skimage import io, exposure
from tqdm.auto import tqdm
import argparse
from .ImageDataAugmentor.image_data_augmentator import ImageDataAugmentor
from .utils import preprocess, load_image, add_suffix

warnings.filterwarnings('ignore')

l_path = Path(__file__).parent
OUTPUT_DIR = (l_path / '../../data/').resolve()
SEGMENTATION_SOURCE_DIR = OUTPUT_DIR / 'sources'

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


def train_generator(train_path, config=default_config):
    """
    Create a generator with the training data. It includes an online-augmentation.
    :param train_path: segmentation dataset train path
    :param config: configuration dictionary.
    :return: keras image generator
    """

    def adjust_data(mask):
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.expand_dims(mask, axis=-1).astype('float')
        return mask

    augm = alb.Compose([
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
        augment=augm,
        preprocess_input=image_preprocess,
        augment_seed=config["seed"])

    mask_datagen = ImageDataAugmentor(augment=augm,
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
    :param test_files: segmentation dataset test path
    :param config: configuration parameters. Config parameters are (target_size, enable_preprocessing)
    :return: keras image generator
    """
    X = []
    y = []
    for test_file in test_files:
        img = load_image(test_file.as_posix(), config["target_size"])[:, :, 0]
        if config["enable_preprocessing"]:
            img = preprocess(img)
        X.append(img)
        y.append(
            load_image(add_suffix(test_file.as_posix(), "mask"),
                       target_size=config["target_size"]))
    return np.array(X).reshape((-1, config["target_size"], config["target_size"], 1)), \
           np.array(y).reshape((-1, config["target_size"], config["target_size"], 1))


def get_data(config=default_config):
    """
    Get segmentation image generators
    :param config: configuration parameters

    Default config:
    ===
    'image_save_prefix': 'image',
    'mask_save_prefix': 'mask',
    'output_folder': ../data,
    'source_dir': ../data/sources,

    'target_size': 512,             #output image and mask size
    'seed': 23,
    'batch_size': 8,
    'enable_preprocessing': False,  #whether to enable preprocessing

    # online augmentation parameters
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
    :return: (tuple) training generator; preloaded val data as tuple of np.ndarray (image, mask)
    """
    segmentation_test_dir = config["output_folder"] / "segmentation" / "test"
    segmentation_train_dir = config["output_folder"] / "segmentation" / "train"
    segmentation_image_dir = segmentation_train_dir / "image"
    segmentation_mask_dir = segmentation_train_dir / "mask"

    preapre_dataset(input_folder=config["source_dir"],
                    output_folder=config["output_folder"],
                    image_size=config["target_size"],
                    force_reset=False)

    si = len(list(segmentation_image_dir.glob('*')))
    sm = len(list(segmentation_mask_dir.glob('*')))
    if si == sm and si > 0:
        tr = train_generator(segmentation_train_dir, config=config)

        val_files = [test_file for test_file in segmentation_test_dir.glob("*.png")
                     if ("_mask" not in test_file.name and "_dilate" not in test_file.name and
                         "_predict" not in test_file.name)
                     ]
        val = val_generator(val_files, config=config)
        return tr, val
    else:
        print("Something goes wrong. I cannot find the dataset.")
        raise FileNotFoundError


def prepare_jsrt(j_image_dir, j_train_left_mask_dir, j_train_right_mask_dir,
                 j_test_left_mask_dir, j_test_right_mask_dir, segmentation_image_dir,
                 segmentation_mask_dir, segmentation_test_dir, image_size=512):
    jsrt_train = j_train_left_mask_dir.glob('*.gif')
    jsrt_test = j_test_left_mask_dir.glob('*.gif')

    for left_image_file in tqdm(jsrt_train):
        base_file = left_image_file.name
        image_file = j_image_dir / left_image_file.stem + '.IMG'
        right_image_file = j_train_right_mask_dir / base_file

        image = 1.0 - np.fromfile(image_file.as_posix(), dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        image = exposure.equalize_hist(image)
        image = (255 * image).astype(np.uint8)
        left_mask = io.imread(left_image_file.as_posix())
        right_mask = io.imread(right_image_file.as_posix())

        image = cv2.resize(image, (image_size, image_size))
        left_mask = cv2.resize(left_mask, (image_size, image_size))
        right_mask = cv2.resize(right_mask, (image_size, image_size))

        mask = np.maximum(left_mask, right_mask)

        cv2.imwrite((segmentation_image_dir / base_file.stem + 'png').as_posix(),
                    image)
        cv2.imwrite((segmentation_mask_dir / base_file.stem + 'png').as_posix(),
                    mask)

    for left_image_file in tqdm(jsrt_test):
        base_file = left_image_file.name
        image_file = j_image_dir / left_image_file.stem + '.IMG'
        right_image_file = j_test_right_mask_dir / base_file

        image = 1.0 - np.fromfile(image_file.as_posix(), dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        image = exposure.equalize_hist(image)
        image = (255 * image).astype(np.uint8)
        left_mask = io.imread(left_image_file.as_posix())
        right_mask = io.imread(right_image_file.as_posix())

        image = cv2.resize(image, (image_size, image_size))
        left_mask = cv2.resize(left_mask, (image_size, image_size))
        right_mask = cv2.resize(right_mask, (image_size, image_size))

        mask = np.maximum(left_mask, right_mask)

        cv2.imwrite((segmentation_test_dir / base_file.stem + 'png').as_posix(),
                    image)
        cv2.imwrite((segmentation_test_dir / f"{base_file.stem}_mask.png").as_posix(),
                    mask)


def prepare_shenzhen(s_image_dir, s_mask_dir, segmentation_image_dir, segmentation_mask_dir,
                     segmentation_test_dir, image_size=512):
    shenzhen_mask_dir = list(s_mask_dir.glob('*.png'))
    shenzhen_train = shenzhen_mask_dir[50:]

    for mask_file in tqdm(shenzhen_mask_dir):
        base_file = mask_file.name.replace("_mask", "")
        image_file = s_image_dir / base_file

        image = cv2.imread(image_file.as_posix())
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))

        if mask_file in shenzhen_train:
            cv2.imwrite((segmentation_image_dir / base_file).as_posix(),
                        image)
            cv2.imwrite((segmentation_mask_dir / base_file).as_posix(),
                        mask)
        else:
            cv2.imwrite((segmentation_test_dir / base_file).as_posix(),
                        image)
            cv2.imwrite((segmentation_test_dir / f"{base_file.stem}_mask{base_file.suffix}").as_posix(), mask)


def prepare_montgomery(m_image_dir, m_left_mask_dir, m_right_mask_dir,
                       segmentation_image_dir, segmentation_mask_dir,
                       segmentation_test_dir, image_size=512):
    montgomery_left_mask_dir = list(m_left_mask_dir.glob('*.png'))
    montgomery_train = montgomery_left_mask_dir[50:]

    for left_image_file in tqdm(montgomery_left_mask_dir):
        base_file = left_image_file.name
        image_file = m_image_dir / base_file
        right_image_file = m_right_mask_dir / base_file

        image = cv2.imread(image_file.as_posix())
        left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
        right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (image_size, image_size))
        left_mask = cv2.resize(left_mask, (image_size, image_size))
        right_mask = cv2.resize(right_mask, (image_size, image_size))

        mask = np.maximum(left_mask, right_mask)

        if left_image_file in montgomery_train:
            cv2.imwrite((segmentation_image_dir / base_file).as_posix(),
                        image)
            cv2.imwrite((segmentation_mask_dir / base_file).as_posix(),
                        mask)
        else:
            cv2.imwrite((segmentation_test_dir / base_file).as_posix(),
                        image)
            cv2.imwrite((segmentation_test_dir / f"{base_file.stem}_mask{base_file.stem}").as_posix(),
                        mask)


def preapre_dataset(input_folder=SEGMENTATION_SOURCE_DIR, output_folder=OUTPUT_DIR, image_size=None,
                    force_reset=False):
    segmentation_test_dir = output_folder / "segmentation" / "test"
    segmentation_train_dir = output_folder / "segmentation" / "train"
    segmentation_image_dir = segmentation_train_dir / "image"
    segmentation_mask_dir = segmentation_train_dir / "mask"

    if force_reset:
        print("Cleaning dataset...")
        os.rmdir(segmentation_image_dir)
        os.rmdir(segmentation_mask_dir)
        os.rmdir(segmentation_test_dir)

    # Create folder structure
    os.makedirs(segmentation_image_dir, exist_ok=True)
    os.makedirs(segmentation_mask_dir, exist_ok=True)
    os.makedirs(segmentation_test_dir, exist_ok=True)

    si = len(list(segmentation_image_dir.glob('*')))
    sm = len(list(segmentation_mask_dir.glob('*')))
    if si == sm and si > 0:
        print(f"Segmentation images and mask already extracted ({si} items)")
        return

    print("Extract Montgomery")
    montgomery_train_dir = input_folder / "MontgomerySet"
    montgomery_image_dir = montgomery_train_dir / "CXR_png"
    montgomery_left_mask_dir = montgomery_train_dir / "ManualMask" / "leftMask"
    montgomery_right_mask_dir = montgomery_train_dir / "ManualMask" / "rightMask"
    if len(list(montgomery_image_dir.glob('*'))) > 0:
        prepare_montgomery(m_image_dir=montgomery_image_dir,
                           m_left_mask_dir=montgomery_left_mask_dir,
                           m_right_mask_dir=montgomery_right_mask_dir,
                           segmentation_image_dir=segmentation_image_dir,
                           segmentation_mask_dir=segmentation_mask_dir,
                           segmentation_test_dir=segmentation_test_dir,
                           image_size=image_size)
    else:
        print(
            f"The Montgomery dataset is not present, or in a different position. {montgomery_image_dir} is empty or not exist.")

    print("Extract Shenzhen")
    shenzhen_train_dir = input_folder / "ChinaSet_AllFiles"
    shenzhen_image_dir = shenzhen_train_dir / "CXR_png"
    shenzhen_mask_dir = shenzhen_train_dir / "mask"
    if len(list(shenzhen_image_dir.glob('*'))) > 0:
        prepare_shenzhen(shenzhen_image_dir,
                         shenzhen_mask_dir,
                         segmentation_image_dir=segmentation_image_dir,
                         segmentation_mask_dir=segmentation_mask_dir,
                         segmentation_test_dir=segmentation_test_dir,
                         image_size=image_size)
    else:
        print(
            f"The Shenzhen dataset is not present, or in a different position. {shenzhen_image_dir} is empty or not exist.")

    print("Extract JSRT")
    jsrt_dir = input_folder / "JSRT"
    jsrt_image_dir = jsrt_dir / "images"
    jsrt_train_left_mask_dir = jsrt_dir / "annotations" / "fold1" / "masks" / "left lung"
    jsrt_train_right_mask_dir = jsrt_dir / "annotations" / "fold1" / "masks" / "right lung"
    jsrt_test_left_mask_dir = jsrt_dir / "annotations" / "fold2" / "masks" / "left lung"
    jsrt_test_right_mask_dir = jsrt_dir / "annotations" / "fold2" / "masks" / "right lung"
    if len(list(jsrt_image_dir.glob('*'))) > 0:
        prepare_jsrt(jsrt_image_dir,
                     jsrt_train_left_mask_dir,
                     jsrt_train_right_mask_dir,
                     jsrt_test_left_mask_dir,
                     jsrt_test_right_mask_dir,
                     segmentation_image_dir=segmentation_image_dir,
                     segmentation_mask_dir=segmentation_mask_dir,
                     segmentation_test_dir=segmentation_test_dir,
                     image_size=image_size)
    else:
        print(f"The JSRT dataset is not present, or in a different position. {jsrt_image_dir} is empty or not exist.")


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
