import os
import numpy as np
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage.interpolation import affine_transform
from tensorflow.compat.v1.keras.utils import Sequence


def transform_matrix_offset_center(matrix, x, y):
    o_x = (x / 2) + 0.5
    o_y = (y / 2) + 0.5
    offset_matrix = np.array([[1, 0, o_x],
                              [0, 1, o_y],
                              [0, 0, 1]])

    reset_matrix = np.array([[1, 0, -o_x],
                             [0, 1, -o_y],
                             [0, 0, 1]])

    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [affine_transform(x_channel,
                                       final_affine_matrix,
                                       final_offset,
                                       order=0,
                                       mode=fill_mode,
                                       cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


class FileContentError(Exception):
    """
    Class for raising Exceptions when encountered files with invalid content
    """
    pass


class ImageDataGenerator(Sequence):
    """
    ImageDataGenerator for generating triplets
    importing from the base Class : tensorflow.keras.utils is necessary

    # Arguments
        data_path           : str --> path to folder where images exist
        triplets_csv        : str --> path to file where the triplets_csv exists
        batch_size          : int --> generates batch_size*3 images per batch
        target_size         : tuple --> input size of the image to be fed into the model
        num_batch_samples   : int --> number of batches to train on per each epoch
                                      (useful for faster model check pointing : default - None)
        rescale             : bool --> rescale images to by their max value (default - True)
        shuffle             : bool --> shuffles the triplet indices after every epoch (default - True)
        augment             : bool --> returns augmented images (default - True)

    # if augment is True, then list of available augmentations are (higher the values, more the augmentation effect)
        rotation_range       : int : range(0, 360) --> angle by which the image is to be rotated anti/clock-wise
        height_shift_range   : float : range(0, 1) --> factor by which the height of the image is to be shifted
        width_shift_range    : float : range(0, 1) --> factor by which the width of the image is to be shifted
        channel_shift_range  : float : range(0, 1) --> factor by which the channel/color of the image is to be shifted
        shear_range          : float : range(0, 1) --> factor by which the image is to be sheared / squished
        zoom_range           : float : range(0, 1) --> factor by which the image is to be zoomed
        horizontal_flip      : bool --> flips the image horizontally
        vertical_flip        : bool --> flips the image vertically
        fill_mode            : str --> Points outside the boundaries of the input
                                       are filled according to the given mode
                                       (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval                 : float --> Value used for points outside the boundaries
                                         of the input if `mode='constant'`.

    # Returns
        A tuple containing batch of images with their respective labels
    """

    def __init__(self,
                 data_path,
                 triplets_csv,
                 batch_size,
                 target_size=(224, 224),
                 embedding_size=2048,
                 num_batch_samples=None,
                 rescale=True,
                 shuffle=False,
                 verbose=True,
                 augment=False,
                 rotation_range=None,
                 height_shift_range=None,
                 width_shift_range=None,
                 channel_shift_range=None,
                 shear_range=None,
                 zoom_range=None,
                 horizontal_flip=False,
                 vertical_flip=False,
                 fill_mode='nearest',
                 cval=0.0):

        if os.path.exists(data_path):
            self.data_path = data_path
        else:
            raise FileNotFoundError("File `{}` doesn't exist".format(data_path))

        if os.path.exists(triplets_csv):
            self.triplets_csv = triplets_csv
        else:
            raise FileNotFoundError("File `{}` doesn't exist".format(triplets_csv))

        # file contents check
        with open(self.triplets_csv) as f:
            triplets = np.array([line.split(",") for line in f.read().split("\n") if len(line.split(",")) == 3])

            if len(set(triplets.ravel()).difference(set(os.listdir(data_path)))) != 0:
                raise FileContentError("`{}` contains images not in `{}`".format(triplets_csv, data_path))
            else:
                self.triplets = triplets

        self.num_triplets = len(self.triplets)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.num_triplets / self.batch_size))
        self.num_batch_samples = num_batch_samples
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.verbose = verbose
        self.embedding_size = embedding_size
        self.rescale_factor = 255 if rescale else 1
        self.on_epoch_end()

        if self.augment:
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
            self.rotation_range = rotation_range
            self.height_shift_range = height_shift_range
            self.width_shift_range = width_shift_range
            self.channel_shift_range = channel_shift_range
            self.shear_range = shear_range
            self.zoom_range = zoom_range
            self.horizontal_flip = horizontal_flip
            self.vertical_flip = vertical_flip
            self.fill_mode = fill_mode
            self.cval = cval

        if self.verbose:
            self.info()

    def info(self):

        """
        Returns the information of the files loaded
        """

        logging.info("Found {:,} images in {}".format(len(os.listdir(self.data_path)), self.data_path))
        logging.info("Found {:,} triplets(*3 images)".format(self.num_triplets))
        logging.info(
            "Found ({:,} query, {:,} positive, {:,} negative) unique images ".format(len(set(self.triplets[:, 0])),
                                                                                     len(set(self.triplets[:, 1])),
                                                                                     len(set(self.triplets[:, 2]))))
        logging.info("batch size of {:,} returns {:,} batches".format(self.batch_size, self.num_batches))

        if self.num_batch_samples:
            logging.info("Training on {:,} randomly sampled batches".format(self.num_batch_samples))

    def __len__(self):

        """
        Returns the number of  batches
        """

        if self.num_batch_samples:
            return self.num_batch_samples

        return self.num_batches

    def on_epoch_end(self):

        """
        Shuffles the indices of triplets.
        if shuffle is set to True
        """

        if self.shuffle:
            np.random.shuffle(self.triplets)

    def random_transform(self, x, seed=9):

        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range and np.random.random() < 0.5:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range and np.random.random() < 0.5:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range and np.random.random() < 0.5:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range and np.random.random() < 0.5:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range and np.random.random() < 0.5:
            zval = np.round(np.random.uniform(1 - self.zoom_range, 1, 1)[0], 2)
            zx, zy = zval, zval
        else:
            zx, zy = 1, 1

        transform_matrix = None

        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range and np.random.random() < 0.5:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

        if self.horizontal_flip and np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)
            return x

        if self.vertical_flip and np.random.random() < 0.5:
            x = flip_axis(x, img_row_axis)
            return x

        return x

    def get_image_array(self, file):

        """Loads image from path .
        # Arguments
            file : path to image.
            target_size : size to which the image is to be resized.
        # Returns
            A randomly transformed version of the image.
        """

        x = img_to_array(load_img(file, target_size=self.target_size)) / self.rescale_factor

        if self.augment:
            return self.random_transform(x)

        return x

    def __getitem__(self, batch_index):

        """Used to access the batches.
        # Arguments
            batch_index : index of the batch to be retrieved
        # Returns
            A tuple containing batch of images and targets.
        """

        # current batch range : tuple  containing the start and end indices
        curr_batch_idxs = np.arange(self.batch_size * batch_index,
                                    min(self.num_triplets, self.batch_size * (batch_index + 1)))

        if len(curr_batch_idxs) != self.batch_size:
            random_idxs = np.random.choice(self.num_triplets, size=self.batch_size - len(curr_batch_idxs),
                                           replace=False)
            curr_batch_idxs = np.concatenate((curr_batch_idxs, random_idxs))

        curr_batch_imgs = np.array(
            [self.get_image_array(os.path.join(self.data_path, imf)) for imf in self.triplets[curr_batch_idxs].ravel()])

        return curr_batch_imgs, np.zeros((self.batch_size * 3, self.embedding_size))
