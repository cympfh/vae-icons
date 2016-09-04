import os
import numpy
from PIL import Image
from random import randrange
from chainer.dataset import dataset_mixin


class ImageDataset(dataset_mixin.DatasetMixin):

    """
    GrayScale image -> 3channel
    horizontal flip
    """

    def __init__(self, paths, root='.', dtype=numpy.float32):
        self._paths = paths
        self._root = root
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        with Image.open(path) as f:
            image = numpy.asarray(f, dtype=self._dtype)

        if image.ndim == 2:
            image = image.reshape((image.shape[0], image.shape[1], 1))
            image = numpy.concatenate((image, image, image), axis=2)

        if randrange(0, 2) == 0:
            image = image[::-1, :, :]

        image = image.transpose(2, 0, 1)
        if image.shape != (3, 48, 48):
            print(path, image.shape)

        image = image / 256

        return image
