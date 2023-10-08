import cv2
import numpy as np


class ImageEnhancement:
    """Collection of image enhancement methods."""

    @staticmethod
    def binarize(image: np.ndarray, threshold: int) -> np.ndarray:
        """
        Apply binarization to the input image.

        Args:
            image (np.ndarray): Input image as a numpy array.
            threshold (int): Threshold value.
        
        Returns:
            np.ndarray: Binarized image.

        """
        image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
        return image

    @staticmethod
    def sharpen_edges(image: np.ndarray) -> np.ndarray:
        """
        Apply edge sharpening using the Laplacian operator.
        The Laplacian operator highlights areas of rapid intensity change.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Image with sharpened edges.

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Laplacian(image, cv2.CV_64F)
        image = np.uint8(np.abs(image))
        return image

    @staticmethod
    def appply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to the input image.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            gamma (float):
                Gamma value for the correction. 
                A higher gamma value (> 1.0) increases brightness, enhances contrast.
                A lower value (< 1.0) darkens the image, reduces contrast.

        Returns:
            np.ndarray: Image with gamma correction applied.

        Note:
            Gamma correction is a non-linear operation that adjusts the intensity values
            in the image to modify its brightness.
            The formula used is: output = (input / 255.0) ** (1.0 / gamma) * 255.

        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
        return image

    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance the contrast of the input image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Image with histogram equalization applied.

        Note:
            Histogram equalization redistributes the intensity values in the image.
            Effective for images with limited dynamic range.

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        return image
