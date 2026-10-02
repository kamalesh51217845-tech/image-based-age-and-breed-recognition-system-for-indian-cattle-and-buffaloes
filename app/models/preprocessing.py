"""
Image Preprocessing Module for Cattle Recognition System
Handles resizing, normalization, and noise reduction
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Union


class ImagePreprocessor:
    """
    Preprocesses images for the cattle breed and age classification model.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor.

        Args:
            target_size: Target dimensions for resizing (height, width)
        """
        self.target_size = target_size
        # ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def load_image(self, image_source: Union[str, bytes, Image.Image]) -> np.ndarray:
        """
        Load image from various sources.

        Args:
            image_source: File path, bytes, or PIL Image

        Returns:
            numpy array of the image in RGB format
        """
        if isinstance(image_source, str):
            # Load from file path
            image = Image.open(image_source)
        elif isinstance(image_source, bytes):
            # Load from bytes
            image = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError("Unsupported image source type")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return np.array(image)

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions while maintaining aspect ratio.

        Args:
            image: Input image as numpy array

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a blank canvas and center the image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using bilateral filtering.
        Preserves edges while smoothing noise.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        # Bilateral filter: good for preserving edges
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.

        Args:
            image: Input image in RGB format

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back to RGB
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.

        Args:
            image: Input image (0-255 range)

        Returns:
            Normalized image ready for model input
        """
        # Convert to float and scale to 0-1
        normalized = image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (normalized - self.mean) / self.std

        return normalized

    def preprocess(self, image_source: Union[str, bytes, Image.Image],
                   enhance: bool = True, denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.

        Args:
            image_source: Image file path, bytes, or PIL Image
            enhance: Whether to apply contrast enhancement
            denoise: Whether to apply noise reduction

        Returns:
            Tuple of (preprocessed image for model, display image)
        """
        # Load image
        image = self.load_image(image_source)

        # Apply optional preprocessing
        if denoise:
            image = self.reduce_noise(image)

        if enhance:
            image = self.enhance_contrast(image)

        # Resize
        resized = self.resize(image)

        # Keep a copy for display
        display_image = resized.copy()

        # Normalize for model
        normalized = self.normalize(resized)

        # Add batch dimension and convert to channels-first format for PyTorch
        # Shape: (1, 3, 224, 224)
        model_input = np.transpose(normalized, (2, 0, 1))
        model_input = np.expand_dims(model_input, axis=0)

        return model_input, display_image

    def preprocess_for_tensorflow(self, image_source: Union[str, bytes, Image.Image],
                                   enhance: bool = True, denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing pipeline for TensorFlow models (channels-last format).

        Args:
            image_source: Image file path, bytes, or PIL Image
            enhance: Whether to apply contrast enhancement
            denoise: Whether to apply noise reduction

        Returns:
            Tuple of (preprocessed image for model, display image)
        """
        # Load image
        image = self.load_image(image_source)

        # Apply optional preprocessing
        if denoise:
            image = self.reduce_noise(image)

        if enhance:
            image = self.enhance_contrast(image)

        # Resize
        resized = self.resize(image)

        # Keep a copy for display
        display_image = resized.copy()

        # Normalize for model
        normalized = self.normalize(resized)

        # Add batch dimension - TensorFlow uses channels-last
        # Shape: (1, 224, 224, 3)
        model_input = np.expand_dims(normalized, axis=0)

        return model_input, display_image
