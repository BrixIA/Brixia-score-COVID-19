"""
Code heavily adapted from:
*https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/
For similar projects, see also:
*https://github.com/davidfreire/Augmentation_project

Utilities for real-time data augmentation on image data.
"""
import warnings
from six.moves import range

import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import random

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
except ImportError:
    scipy = None
from .directory_iterator import DirectoryIterator


class ImageDataAugmentor(Sequence):
    """Generate batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).
    # Arguments
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        preprocess_input: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image, and should output a Numpy tensor with the same shape.
        preprocess_image_before_augmentation: function that will be implied on each input image (not mask).
            The function will run after the image is resized, before it is augmented.
            The function should take one argument:
            one image, and should output a Numpy tensor with the same shape.
        augment: augmentations passed as albumentations or imgaug transformation
            or sequence of transformations.
        augment_seed: makes augmentations from albumentations deterministic.
            Notice! imgaug uses input seeds so to make the transformations deterministic,
            use ia.seed(X) or call transformations with .to_deterministic().
        augment_mode: should be either 'image' or 'mask'. If latter, only the
            albumentation transformation relevant for segmentation mask augmentation will be use.
            Notice! In imgaug the 'mask'-mode cannot be selected, so you have to sure that your augmentations
            are fit for mask generation (e.g. no noise generation/color manipulation).
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation
            (strictly between 0 and 1).
        dtype: Dtype to use for the generated arrays.
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 augment=None,
                 augment_mode='image',
                 augment_seed=None,
                 rescale=None,
                 preprocess_input=None,
                 preprocess_image_before_augmentation=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 dtype='float32'):

        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.augment = augment
        self.augment_seed = augment_seed
        self.augment_mode = augment_mode
        if self.augment_mode == 'mask' and self.augment_seed == None:
            warnings.warn('This ImageDataAugmentor uses `augment_mode=mask` '
                          'but no `augment_seed` was given. Setting `augment_seed=0`.'
                          )
            self.augment_seed = 0

        self.rescale = rescale
        self.preprocess_input = preprocess_input
        self.preprocess_image_before_augmentation = preprocess_image_before_augmentation
        self.dtype = dtype
        self.total_transformations_done = 0

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataAugmentor specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataAugmentor specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataAugmentor specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataAugmentor specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation=cv2.INTER_NEAREST):
        """Takes the path to a directory & generates batches of augmented data.
        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "gray", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
                If set to False, sorts the data in alphanumeric order.
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataAugmentor`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation
        )

    def standardize(self, x):
        """Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standarize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:
        standardize(np.copy(x))
        # Arguments
            x: Batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocess_input:
            x = self.preprocess_input(x)
        if self.rescale:
            x = np.multiply(x, self.rescale)
        if self.samplewise_center:
            x = x - np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x = np.divide(x, (np.std(x, keepdims=True) + 1e-6))

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataAugmentor specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-6)
            else:
                warnings.warn('This ImageDataAugmentor specifies '
                              '`featurewise_std_normalization`, '
                              'but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataAugmentor specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def transform_image(self, image):
        """
        Add comments
        """

        if self.augment_mode == 'mask':
            if self.augment is not None:
                if 'albumentations' in str(type(self.augment)):
                    if self.augment_seed:
                        random.seed(self.augment_seed + self.total_transformations_done)
                    data = self.augment(image=np.zeros_like(image), mask=image)
                    image = data['mask']

                elif 'imgaug' in str(type(self.augment)):
                    warnings.warn('imgaug does not yet support mask generation: consider using albumentations instead.'
                                  'The masks were generated using the augmentations provided:'
                                  'make sure they are fit for mask generation.')

                    if self.augment_seed is not None:
                        warnings.warn('You are using `imgaug` for mask generation.'
                                      'Make sure to call imgaug augmentations with `.to_deterministic()` to ensure'
                                      'that images and masks are augmented correctly together.')
                    image = self.augment(image=image)

                image = self.standardize(image)
                self.total_transformations_done += 1

        else:
            if self.preprocess_image_before_augmentation:
                image = self.preprocess_image_before_augmentation(image)
            if self.augment is not None:
                if 'albumentations' in str(type(self.augment)):
                    if self.augment_seed is not None:
                        random.seed(self.augment_seed + self.total_transformations_done)

                    image = self.augment(image=image)['image']

                elif 'imgaug' in str(type(self.augment)):
                    image = self.augment(image=image)

            image = self.standardize(image)

            self.total_transformations_done += 1

        return image

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits the data generator to some sample data.
        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.
        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.
        # Arguments
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
       """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' +
                self.data_format + '" (channels on axis ' +
                str(self.channel_axis) + '), i.e. expected '
                                         'either 1, 3 or 4 channels on axis ' +
                str(self.channel_axis) + '. '
                                         'However, it was passed an array with shape ' +
                str(x.shape) + ' (' + str(x.shape[self.channel_axis]) +
                ' channels).')

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype)
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.augment(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x = x - self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x = np.divide(x, (self.std + 1e-6))

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            flat_x = np.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)
