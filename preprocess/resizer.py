import cv2
import numpy as np
import tensorflow as tf


class ImageResizer:
    """Resize the input image."""

    def __init__(self, library='tensorflow'):
        """
        Initialize the ImageResizer with the specified library.

        Args:
            library (str): Library to use for resizing. Options: 'opencv', 'tensorflow'.

        Attributes:
            library (str): Library to use for resizing (in lower case). Options: 'opencv', 'tensorflow'.

        """
        self.library = library.lower()

    def resize(self, path: str, width: int, height: int) -> np.ndarray:
        """
        Resize the input image using the library specified.

        Args:
            height (int): Target height of the resized image.
            path (str): Path to an input image.
            width (int): Target width of the resized image.

        Returns:
            np.ndarray: Resized image.

        """
        if self.library == 'opencv':
            return self._resize_with_opencv(path, width, height)
        elif self.library == 'tensorflow':
            return self._resize_with_tensorflow(path, width, height)
        else:
            raise ValueError('Invalid library specified.')

    def _resize_with_opencv(self, path: str, width: int, height: int) -> np.ndarray:
        """
        Resize the input image using OpenCV.

        Args:
            height (int): Target height of the resized image.
            path (str): Path to an input image.
            width (int): Target width of the resized image.

        Returns:
            np.ndarray: Resized image.

        """
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _resize_with_tensorflow(self, path: str, width: int, height: int) -> np.ndarray:
        """
        Resize the input image using TensorFlow.

        Args:
            height (int): Target height of the resized image.
            path (str): Path to an input image.
            width (int): Target width of the resized image.
        
        Returns:
            np.ndarray: Resized image.

        """
        image = tf.keras.preprocessing.image.load_img(path, target_size=(height, width))
        image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
        return image
