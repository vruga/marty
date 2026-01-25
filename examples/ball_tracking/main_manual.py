"""Manual ball tracking mode - robot only moves when you press SPACE.

Usage:
    python main_manual.py

Controls:
    SPACE - Lock current ball position, robot moves there slowly
    R     - Return to ready position
    Q     - Quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import cv2

from config import CALIBRATION_FILE_PATH
from camera import RealSenseCamera
from detector import BallDetector
from calibration import load_transform
from ball_to_robot import VisionToRobotPipeline
from controller import RobotController


def main():
    print("=== Manual Ball Tracking ===")
    print("Controls:")
    print("  SPACE - Lock ball position and move arm there")
    print("  R     - Return to ready position")
    print("  Q     - Quit")
    print()

    # Load calibration
    if not os.path.exists(CALIBRATION_FILE_PATH):
        print(f"ERROR: No calibration file at {CALIBRATION_FILE_PATH}")
        print("Run calibrate_camera.py first, or create a default transform.")
        # Create identity transform for testing
        print("Using identity transform for testing...")
        cam_to_robot_T = np.eye(4)
    else:
        cam_to_robot_T = load_transform(CALIBRATION_FILE_PATH)
        print("Loaded calibration.")

    # Initialize: pipeline = detect → transform to robot → target (we use .ball_pos_robot for SPACE)
    cam = RealSenseCamera()
    cam.start()
    print("Camera started.")
    detector = BallDetector()
    pipeline = VisionToRobotPipeline(detector, cam, cam_to_robot_T, predictor=None, mode="direct")
    controller = RobotController(use_hardware=True)

    # State
    locked_position = None
    state = "IDLE"  # IDLE, MOVING, RETURNING

    # Move to ready
    print("Moving to ready position...")
    for _ in range(100):
        controller.move_to_ready()
        time.sleep(0.02)
    print("Ready. Waiting for commands...")

    try:
        while True:
            color, depth = cam.get_frames()
            if color is None:
                continue

            # Pipeline: detect → transform to robot frame
            r = pipeline.process(color, depth)
            ball_pos_robot = r.ball_pos_robot
            center, radius, mask = r.center, r.radius, r.mask

            # Draw display
            display = color.copy()

            # Draw detection
            if center is not None:
                cv2.circle(display, center, int(radius), (0, 0, 255), 2)
                cv2.circle(display, center, 3, (0, 0, 255), -1)
                if ball_pos_robot is not None:
                    cv2.putText(display, f"Ball: [{ball_pos_robot[0]:.3f}, {ball_pos_robot[1]:.3f}, {ball_pos_robot[2]:.3f}]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw state
            state_color = (0, 255, 0) if state == "MOVING" else (255, 255, 255)
            cv2.putText(display, f"State: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

            # Draw locked position
            if locked_position is not None:
                cv2.putText(display, f"Target: [{locked_position[0]:.3f}, {locked_position[1]:.3f}, {locked_position[2]:.3f}]",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw EE position
            ee_pos = controller.get_ee_position()
            cv2.putText(display, f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Show mask in corner
            if mask is not None:
                h, w = display.shape[:2]
                mask_small = cv2.resize(mask, (w // 4, h // 4))
                mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                display[0:h // 4, w - w // 4:w] = mask_color

            # Instructions
            cv2.putText(display, "SPACE=lock, R=ready, Q=quit",
                        (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Manual Ball Tracking", display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):  # SPACE - lock position
                if ball_pos_robot is not None:
                    locked_position = ball_pos_robot.copy()
                    state = "MOVING"
                    print(f"[LOCKED] Target: {locked_position}")
                else:
                    print("[ERROR] No ball detected")

            elif key == ord('r'):  # R - return to ready
                locked_position = None
                state = "RETURNING"
                print("[RETURNING]")

            # Execute motion based on state
            if state == "MOVING" and locked_position is not None:
                # Create target pose
                target_pose = np.array([
                    locked_position[0],
                    locked_position[1],
                    locked_position[2],
                    0.0, 0.0, 0.0
                ])
                result, success = controller.move_to_target_pose(target_pose)
                if not success:
                    print("[IK FAILED]")

            elif state == "RETURNING":
                at_ready = controller.move_to_ready()
                if at_ready:
                    state = "IDLE"
                    print("[IDLE]")

            # Small delay for control rate
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.stop()
        controller.disconnect()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
