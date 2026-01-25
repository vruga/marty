"""Orange ball detection: HSV threshold, contours, circle fit, depth, 3D position."""

import cv2
import numpy as np
from config import (
    HSV_LOWER, HSV_UPPER, MIN_BALL_RADIUS_PX, MAX_BALL_RADIUS_PX,
    MIN_CIRCULARITY, DEPTH_SAMPLE_SIZE
)


class BallDetector:
    def __init__(self):
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, color_image, depth_image, camera):
        """Detect orange ball and return 3D position in camera frame.

        Args:
            color_image: BGR image (HxWx3 uint8)
            depth_image: Depth image (HxW uint16, millimeters)
            camera: RealSenseCamera instance for deprojection

        Returns:
            tuple: (position_3d, pixel_center, radius_px, mask)
                   position_3d is np.array([x,y,z]) in meters or None if not detected
                   pixel_center is (cx, cy) or None
                   radius_px is detected radius in pixels or None
                   mask is the binary detection mask
        """
        # Convert to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Threshold for orange
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Morphological cleanup
        mask = cv2.erode(mask, self.kernel_erode, iterations=1)
        mask = cv2.dilate(mask, self.kernel_dilate, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        best_radius = None
        best_position = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < np.pi * MIN_BALL_RADIUS_PX**2:
                continue

            # Fit minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            if radius < MIN_BALL_RADIUS_PX or radius > MAX_BALL_RADIUS_PX:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue

            # This contour is a valid ball candidate
            cx_int, cy_int = int(cx), int(cy)

            # Sample depth in a small region around center
            depth_m = self._sample_depth(depth_image, cx_int, cy_int)
            if depth_m is None or depth_m <= 0.1 or depth_m > 5.0:
                continue

            # Deproject to 3D
            position_3d = camera.deproject_pixel(cx, cy, depth_m)

            # Keep the largest valid detection
            if best_radius is None or radius > best_radius:
                best_center = (cx_int, cy_int)
                best_radius = radius
                best_position = position_3d

        return best_position, best_center, best_radius, mask

    def _sample_depth(self, depth_image, cx, cy):
        """Average depth over a small region around (cx, cy).

        Args:
            depth_image: HxW uint16 depth in millimeters
            cx, cy: center pixel coordinates

        Returns:
            float: depth in meters, or None if insufficient valid samples
        """
        h, w = depth_image.shape
        half = DEPTH_SAMPLE_SIZE // 2

        y_start = max(0, cy - half)
        y_end = min(h, cy + half + 1)
        x_start = max(0, cx - half)
        x_end = min(w, cx + half + 1)

        region = depth_image[y_start:y_end, x_start:x_end].astype(np.float32)
        valid = region[region > 0]

        if len(valid) < 3:
            return None

        return float(np.median(valid)) / 1000.0  # mm to meters
