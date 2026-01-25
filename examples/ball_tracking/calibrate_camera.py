"""Standalone calibration tool: collect points, compute & save camera-to-robot transform.

Usage:
    python calibrate_camera.py

Flow:
  1. Robot moves to predefined poses. At each pose you confirm the orange marker
     on the end-effector is visible and press 'c' to capture (or 's' to skip).
  2. Camera 3D: detector finds orange blob, depth at center, deproject via
     RealSense color intrinsics -> camera frame (X right, Y down, Z forward).
  3. Robot 3D: FK from qpos[1:5] + base yaw from qpos[0] -> robot frame
     (X forward, Y left, Z up). Optional MARKER_OFFSET_IN_ROBOT_FRAME is added
     if the marker is not at the EE.
  4. Kabsch (SVD) computes rigid T: p_robot = T @ [p_cam; 1]. Saved to JSON.

Important: Put the orange marker as close as possible to the EE. Set ROBOT_TYPE
in config.py to match your arm (so100 / so101). Use 6–8 diverse poses. The marker
is detected with the same BallDetector as the ball; if it is much smaller or
larger, temporarily adjust MIN/MAX_BALL_RADIUS_PX in config for this run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import cv2

from config import (
    ROBOT_PORT, CALIBRATION_FILE, ROBOT_TYPE,
    CALIBRATION_FILE_PATH, NUM_CALIBRATION_POINTS,
    MARKER_OFFSET_IN_ROBOT_FRAME,
)
from camera import RealSenseCamera
from detector import BallDetector
from calibration import compute_rigid_transform, save_transform
from lerobot_kinematics import lerobot_FK, get_robot, feetech_arm

# Predefined calibration joint positions (joint 0 varies, joints 1-4 vary)
CALIBRATION_POSITIONS = [
    np.array([0.0, -2.5, 2.5, 0.0, -1.57, -0.157]),
    np.array([0.3, -2.5, 2.5, 0.0, -1.57, -0.157]),
    np.array([-0.3, -2.5, 2.5, 0.0, -1.57, -0.157]),
    np.array([0.0, -2.0, 2.0, 0.0, -1.57, -0.157]),
    np.array([0.3, -2.0, 2.0, 0.3, -1.57, -0.157]),
    np.array([-0.3, -2.0, 2.0, -0.3, -1.57, -0.157]),
    np.array([0.0, -2.5, 2.8, 0.3, -1.57, -0.157]),
    np.array([0.0, -2.8, 2.5, -0.3, -1.57, -0.157]),
]


def get_fk_position(qpos, robot_kin):
    """Get end-effector position from FK (joints 1-4)."""
    fk_result = lerobot_FK(qpos[1:5], robot=robot_kin)
    # FK returns [X, Y, Z, gamma, beta, alpha]
    # Apply yaw rotation (joint 0) to the XY position
    yaw = qpos[0]
    x, y, z = fk_result[0], fk_result[1], fk_result[2]
    x_rot = x * np.cos(yaw) - y * np.sin(yaw)
    y_rot = x * np.sin(yaw) + y * np.cos(yaw)
    return np.array([x_rot, y_rot, z])


def main():
    robot_kin = get_robot(ROBOT_TYPE)

    print("=== Camera-to-Robot Calibration ===")
    print(f"ROBOT_TYPE={ROBOT_TYPE}, points={NUM_CALIBRATION_POINTS}")
    print("Attach an orange marker at the end-effector (or set MARKER_OFFSET_IN_ROBOT_FRAME).")
    print("The EE+marker must be IN the camera FOV at each pose; if not, press 's' to skip.")
    print("(See CAMERA_CALIBRATION.md §10 for alternatives when the robot is not in FOV.)")
    print()

    # Initialize camera
    cam = RealSenseCamera()
    cam.start()
    print("Camera started.")

    # Initialize detector
    detector = BallDetector()

    # Connect to robot
    arm = feetech_arm(driver_port=ROBOT_PORT, calibration_file=CALIBRATION_FILE)
    print("Robot connected.")

    camera_points = []
    robot_points = []

    try:
        for i, target_qpos in enumerate(CALIBRATION_POSITIONS[:NUM_CALIBRATION_POINTS]):
            print(f"\n--- Point {i+1}/{NUM_CALIBRATION_POINTS} ---")
            print(f"Moving to: {[f'{q:.2f}' for q in target_qpos]}")

            # Move robot to position
            arm.action(target_qpos)
            time.sleep(2.0)  # Wait for robot to settle

            # Get FK position (EE in robot frame); add marker offset if marker not at EE
            robot_pos = get_fk_position(target_qpos, robot_kin) + MARKER_OFFSET_IN_ROBOT_FRAME
            print(f"Robot point (EE + marker offset): {robot_pos}")

            # Detect marker in camera
            print("Detecting marker... Press 'c' to capture, 's' to skip.")
            captured = False

            while not captured:
                color, depth = cam.get_frames()
                if color is None:
                    continue

                pos_3d, center, radius, mask = detector.detect(color, depth, cam)

                # Draw detection
                display = color.copy()
                if center is not None:
                    cv2.circle(display, center, int(radius), (0, 255, 0), 2)
                    cv2.putText(display, f"3D: [{pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f}]",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(display, f"Point {i+1}/{NUM_CALIBRATION_POINTS} - 'c'=capture, 's'=skip",
                            (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Calibration", display)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('c') and pos_3d is not None:
                    camera_points.append(pos_3d.copy())
                    robot_points.append(robot_pos.copy())
                    print(f"Captured! Camera: {pos_3d}, Robot: {robot_pos}")
                    captured = True
                elif key == ord('s'):
                    print("Skipped.")
                    captured = True
                elif key == ord('q'):
                    print("Aborted.")
                    return

        cv2.destroyAllWindows()

        # Compute transform
        if len(camera_points) < 3:
            print("Need at least 3 points for calibration. Aborting.")
            return

        camera_pts = np.array(camera_points)
        robot_pts = np.array(robot_points)

        T = compute_rigid_transform(camera_pts, robot_pts)
        print(f"\nComputed transform (camera -> robot):")
        print(T)

        # Residuals: per-point and per-axis (helps spot axis flips)
        errors = []
        err_x, err_y, err_z = [], [], []
        for cp, rp in zip(camera_pts, robot_pts):
            p_hom = np.append(cp, 1.0)
            tr = (T @ p_hom)[:3]
            err = np.linalg.norm(tr - rp)
            errors.append(err)
            err_x.append(abs(tr[0] - rp[0]))
            err_y.append(abs(tr[1] - rp[1]))
            err_z.append(abs(tr[2] - rp[2]))
        print(f"Residuals: mean={np.mean(errors)*1000:.1f} mm, max={np.max(errors)*1000:.1f} mm")
        print(f"  per-axis mean |error|: X={np.mean(err_x)*1000:.1f}, Y={np.mean(err_y)*1000:.1f}, Z={np.mean(err_z)*1000:.1f} mm")

        # Save
        save_transform(T, CALIBRATION_FILE_PATH)
        print("Calibration complete!")

    finally:
        cam.stop()
        arm.disconnect()


if __name__ == "__main__":
    main()
