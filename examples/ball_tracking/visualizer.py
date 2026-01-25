"""OpenCV debug window: detection overlay, trajectory visualization, state info."""

import cv2
import numpy as np
from config import (
    VIS_WINDOW_NAME, VIS_TRAJECTORY_COLOR, VIS_DETECTION_COLOR, VIS_INTERCEPT_COLOR
)


class Visualizer:
    """Debug visualization overlay for ball tracking."""

    def __init__(self):
        self.window_created = False

    def draw(self, color_image, detection_center, detection_radius, mask,
             trajectory_2d=None, intercept_2d=None, state="IDLE",
             ball_pos_3d=None, ball_vel_3d=None, ee_pos=None):
        """Draw debug overlay on the color image.

        Args:
            color_image: BGR image to draw on (modified in place)
            detection_center: (cx, cy) pixel center of detected ball, or None
            detection_radius: radius in pixels, or None
            mask: binary detection mask (shown in corner)
            trajectory_2d: list of (px, py) projected trajectory points
            intercept_2d: (px, py) projected intercept point
            state: current state machine state string
            ball_pos_3d: [x,y,z] ball position in robot frame
            ball_vel_3d: [vx,vy,vz] ball velocity
            ee_pos: [x,y,z] end-effector position

        Returns:
            np.ndarray: Display image with overlays
        """
        display = color_image.copy()
        h, w = display.shape[:2]

        # Draw ball detection
        if detection_center is not None and detection_radius is not None:
            cv2.circle(display, detection_center, int(detection_radius),
                       VIS_DETECTION_COLOR, 2)
            cv2.circle(display, detection_center, 3, VIS_DETECTION_COLOR, -1)

        # Draw predicted trajectory
        if trajectory_2d is not None and len(trajectory_2d) > 1:
            for i in range(len(trajectory_2d) - 1):
                pt1 = trajectory_2d[i]
                pt2 = trajectory_2d[i + 1]
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(display, pt1, pt2, VIS_TRAJECTORY_COLOR, 2)

        # Draw intercept point
        if intercept_2d is not None:
            cv2.drawMarker(display, intercept_2d, VIS_INTERCEPT_COLOR,
                           cv2.MARKER_CROSS, 20, 3)
            cv2.putText(display, "INTERCEPT", (intercept_2d[0] + 15, intercept_2d[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, VIS_INTERCEPT_COLOR, 1)

        # State info overlay (top-left)
        y_offset = 25
        state_color = (0, 255, 0) if state == "LOCKED" else (255, 255, 255)
        cv2.putText(display, f"State: {state}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        y_offset += 25

        if ball_pos_3d is not None:
            cv2.putText(display, f"Ball: [{ball_pos_3d[0]:.3f}, {ball_pos_3d[1]:.3f}, {ball_pos_3d[2]:.3f}]",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

        if ball_vel_3d is not None:
            speed = np.linalg.norm(ball_vel_3d)
            cv2.putText(display, f"Speed: {speed:.2f} m/s",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

        if ee_pos is not None:
            cv2.putText(display, f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw mask in corner (small)
        if mask is not None:
            mask_small = cv2.resize(mask, (w // 4, h // 4))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            display[0:h // 4, w - w // 4:w] = mask_color

        return display

    def show(self, display_image):
        """Show the display image in an OpenCV window.

        Returns:
            int: Key pressed (0xFF masked), or -1 if no key
        """
        if not self.window_created:
            cv2.namedWindow(VIS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self.window_created = True

        cv2.imshow(VIS_WINDOW_NAME, display_image)
        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        """Close the visualization window."""
        cv2.destroyAllWindows()
        self.window_created = False
