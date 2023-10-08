import numpy as np


class ImageNormalizer:
    """Normalize the input image."""

    @staticmethod
    def z_score(image: np.ndarray) -> np.ndarray:
        """
        Apply Z-Score Normalization to the input image.
        Can be beneficial when the model assumes a Gaussian distribution of input values.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Normalized image.

        """
        return (image - np.mean(image)) / np.std(image)

    @staticmethod
    def custom_norm(image: np.ndarray, factor: int) -> np.ndarray:
        """
        Apply custom normalization to the input image.
        Useful when the model benefits from input values within a specific range, i.e. [0, 1], [0, 2].
        The case of factor=1 is equivalent to Min-Max scaling.

        Args:
            image (np.ndarray): Input image.
            factor (int): Normalization factor.

        Returns:
            np.ndarray: Normalized image.

        """
        return factor * (image / 255.0)
