from typing import Tuple

import cv2
import numpy as np


class ImageDenoiser:

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel: Tuple[int, int], deviation: int) -> np.ndarray:
        """
        Apply Gaussian blur to the input image.
        Larger kernel sizes and standard deviations result in stronger blurring.
        
        Args:
            image (np.ndarray): Input image as a numpy array.
            kernel (Tuple[int, int]): Kernel size.
            deviation (int): Standard deviation.
        
        Returns:
            np.ndarray: Blurred image.

        """
        image = cv2.GaussianBlur(image, kernel, deviation)
        return image

    @staticmethod
    def apply_median_filtering(image: np.ndarray, kernel: int) -> np.ndarray:
        """
        Apply median filtering to the input image.
        Larger kernel sizes result in stronger blurring.

        Args:
            image (np.ndarray): Input image as a numpy array.
            kernel (int): Kernel size.

        Returns:
            np.ndarray: Blurred image.

        """
        image = cv2.medianBlur(image, kernel)
        return image
