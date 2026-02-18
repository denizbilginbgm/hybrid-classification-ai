from typing import List

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.core import config


class ImagePreprocessor:
    def process_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Process multiple PIL images through the preprocessing pipeline.

        :param images: List of PIL Image objects
        :return: List of processed PIL Image objects
        """
        processed_images = []

        for index, image in enumerate(images):
            processed_image = self.process_single_image(image)
            processed_images.append(processed_image)
        return processed_images

    def process_single_image(self, image: Image.Image) -> Image.Image:
        """
        Process a single PIL image through the preprocessing pipeline.

        :param image: PIL Image object
        :return: Processed PIL Image object
        """
        # Converting PIL Image to OpenCV format
        cv_image = self.__pil_to_cv2(image)

        # Converting to grayscale
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        # Rotation correction
        if config.ENABLE_ROTATION:
            gray = self.__correct_rotation(gray)

        # Deskew
        if config.ENABLE_DESKEW:
            gray = self.__deskew_image(gray)

        return self.__cv2_to_pil(gray)

    def __correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image rotation (0°, 90°, 180°, 270°).
        Uses Tesseract OSD or edge-based detection.

        :param image: Grayscale image as numpy array
        :return: Rotated image
        """
        try:
            # Get orientation info
            osd = pytesseract.image_to_osd(image)
            rotation = int([line for line in osd.split('\n') if 'Rotate' in line][0].split(':')[1].strip())

            # Correct rotation
            if rotation > 0:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            """
            elif rotation == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            """

        except Exception:
            # Fallback: Use projection profile method
            image = self.__detect_rotation_by_projection(image)

        return image

    def __detect_rotation_by_projection(self, image: np.ndarray) -> np.ndarray:
        """
        Detect rotation using projection profile variance method.
        Tests 0°, 90°, 180°, 270° and selects the one with highest variance.

        :param image: Grayscale image
        :return: Best rotated image
        """
        rotations = {
            0: image,
            90: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            180: cv2.rotate(image, cv2.ROTATE_180),
            270: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        }

        max_variance = 0
        best_rotation = 0

        for angle, rotated in rotations.items():
            # Calculate horizontal projection
            projection = np.sum(rotated, axis=1)
            variance = np.var(projection)

            if variance > max_variance:
                max_variance = variance
                best_rotation = angle

        return rotations[best_rotation]

    def __deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in the image using Hough Line Transform or projection profile.

        :param image: Grayscale image
        :return: Deskewed image
        """
        # Method 1: Try Hough Transform approach
        angle = self.__detect_skew_hough(image)

        # If angle is too small, skip deskewing
        if abs(angle) < 0.5:
            return image

        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def __detect_skew_hough(self, image: np.ndarray) -> float:
        """
        Detect skew angle using Hough Line Transform.

        :param image: Grayscale image
        :return: Skew angle in degrees
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return 0.0

        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            # Filter out vertical and horizontal lines
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle
        return float(np.median(angles))

    def __binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize the image using adaptive or Otsu's thresholding.

        :param image: Grayscale image
        :return: Binarized image
        """
        if config.BINARIZATION_METHOD == "adaptive":
            # Adaptive thresholding - better for varying lighting conditions
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
        else:
            # Otsu's thresholding - automatic global threshold
            _, binary = cv2.threshold(
                image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def __pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format.
        :param pil_image: PIL Image to format
        :return: Formatted OpenCV image
        """
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def __cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL Image.

        :param cv_image: Preprocessed image
        :return: Preprocessed PIL image
        """
        if len(cv_image.shape) == 2:  # Grayscale
            return Image.fromarray(cv_image)
        else:  # Color
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))