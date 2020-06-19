# Many of the functions below are adapted from tesorflow-addons
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from functools import partial
import cv2
import numpy as np
from skimage import exposure, filters


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.
    Args:
      image: 2/3/4D tensor.
    Returns:
      4D tensor with the same type.
    """
    with tf.control_dependencies(
            [
                tf.debugging.assert_rank_in(
                    image, [2, 3, 4], message="`image` must be 2/3/4D tensor"
                )
            ]
    ):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.
    Args:
      image: 4D tensor.
      ndims: The original rank of the image.
    Returns:
      `ndims`-D tensor with the same type.
    """
    with tf.control_dependencies(
            [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


@tf.function
def equalize_image_gray(image, data_format="channels_last"):
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(image, channel, nbins=256):
        """Scale the data in the channel to implement equalize."""
        image_dtype = image.dtype

        if data_format == "channels_last":
            image = tf.cast(image[:, :, channel], tf.int32)
        elif data_format == "channels_first":
            image = tf.cast(image[channel], tf.int32)
        else:
            raise ValueError(
                "data_format can either be channels_last or channels_first"
            )
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(image, [0, nbins], nbins=nbins)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // nbins

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, nbins)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.

        if step == 0:
            result = image
        else:
            result = tf.gather(build_lut(histo, step), image)

        return tf.cast(result, image_dtype)

    image = scale_channel(4095 * image, channel=0, nbins=4096)
    image = image / tf.math.reduce_max(image)
    image = tf.expand_dims(image, axis=-1)
    return image


@tf.function
def equalize(image, data_format="channels_last"):
    """Equalize image(s)
    Args:
      image: A tensor of shape
          (num_images, num_rows, num_columns, num_channels) (NHWC), or
          (num_images, num_channels, num_rows, num_columns) (NCHW), or
          (num_rows, num_columns, num_channels) (HWC), or
          (num_channels, num_rows, num_columns) (CHW), or
          (num_rows, num_columns) (HW). The rank must be statically known (the
          shape is not `TensorShape(None)`).
      data_format: Either 'channels_first' or 'channels_last'
    Returns:
      Image(s) with the same type and shape as `images`, equalized.
    """
    image_dims = tf.rank(image)
    image = to_4D_image(image)
    fn = partial(equalize_image_gray, data_format=data_format)
    image = tf.map_fn(fn, image)
    return from_4D_image(image, image_dims)


def load_image(filename, target_size=512):
    """
    Load and resize a grayscale image from file
    :param filename: filename with path
    :param target_size: output size
    :return: loaded image as (num_rows, num_columns, 1)
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"File error: {filename}")
    img = img / 255
    img = cv2.resize(img, (target_size, target_size))
    img = np.reshape(img, img.shape + (1,))
    return img


def image_preprocess(img, clip_limit=0.01, med_filt=3):
    """
    preprocess input image
    :param med_filt: median filter kernel size
    :param clip_limit: CLAHE clip limit
    :param img: (Rows, Cols, 1)
    :return: (Rows, Cols, 1)
    """
    img = img[:,:,0]
    img = preprocess(img, clip_limit=0.01, med_filt=3)
    return np.expand_dims(img, axis=-1)


def preprocess(img, clip_limit=0.01, med_filt=3):
    """
    Preprocess single CXR with clahe, median filtering and clipping
    :param img: input image (Rows, Cols)
    :param clip_limit: CLAHE clip limit
    :param med_filt: median filter kernel size
    :return: (Rows, Cols)
    """
    img = img.astype('float32')/img.max()

    img_eq = exposure.equalize_adapthist(
        img, clip_limit=clip_limit)
    img_eq_median = filters.median(img_eq, np.ones(
        (med_filt,med_filt))).astype(np.float32)

    lower, upper = np.percentile(img_eq_median.flatten(), [2, 98])
    img_clip = np.clip(img_eq_median, lower, upper)
    return (img_clip - lower)/(upper - lower)


def add_suffix(base_file, suffix):
    if isinstance(base_file, str):
        filename, fileext = os.path.splitext(base_file)
        return "%s_%s%s" % (filename, suffix, fileext)
    elif isinstance(base_file, list):
        out = []
        for bf in base_file:
            filename, fileext = os.path.splitext(bf)
            out.append("%s_%s%s" % (filename, suffix, fileext))
        return out
    else:
        raise ValueError
