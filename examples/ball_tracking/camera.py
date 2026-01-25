"""RealSense camera wrapper: init streams, get aligned RGB+depth, pixel-to-3D deprojection.

Camera coordinate frame (RealSense / rs2_deproject_pixel_to_point):
  X = right, Y = down, Z = forward (into the scene). Origin at color optical center. Meters.
Intrinsics: from RealSense color stream (not calibrated here). Used only for deprojection.
"""

import numpy as np
import pyrealsense2 as rs
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class RealSenseCamera:
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.intrinsics = None

    def start(self):
        """Initialize and start RealSense pipeline with aligned color+depth streams."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self.pipeline.start(config)

        # Create alignment object (align depth to color)
        self.align = rs.align(rs.stream.color)

        # Get intrinsics from the color stream
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        """Get aligned color and depth frames.

        Returns:
            tuple: (color_image, depth_image) as numpy arrays, or (None, None) on failure.
                   color_image is HxWx3 uint8 BGR, depth_image is HxW uint16 in mm.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def deproject_pixel(self, pixel_x, pixel_y, depth_m):
        """Convert pixel coordinates + depth to 3D point in camera frame.

        Uses RealSense color intrinsics. Frame: X right, Y down, Z forward (meters).

        Args:
            pixel_x: x coordinate in image (column)
            pixel_y: y coordinate in image (row)
            depth_m: depth in meters (RealSense depth is in mm; convert before calling)

        Returns:
            np.ndarray: [x, y, z] in camera frame (meters)
        """
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [pixel_x, pixel_y], depth_m
        )
        return np.array(point_3d)

    def stop(self):
        """Stop the RealSense pipeline."""
        if self.pipeline:
            self.pipeline.stop()


if __name__ == "__main__":
    import cv2

    cam = RealSenseCamera()
    cam.start()
    print(f"Camera started: {cam.width}x{cam.height} @ {cam.fps}fps")
    print(f"Intrinsics: fx={cam.intrinsics.fx:.1f}, fy={cam.intrinsics.fy:.1f}")

    try:
        while True:
            color, depth = cam.get_frames()
            if color is None:
                continue
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )
            combined = np.hstack((color, depth_colormap))
            cv2.imshow("RealSense", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
