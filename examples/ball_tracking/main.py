"""Real robot entry point: orchestrates vision and control threads with state machine.

Usage:
    python main.py

State machine:
    IDLE -> (ball detected, 3+ frames) -> TRACKING
    TRACKING -> (intercept found in workspace) -> INTERCEPTING
    INTERCEPTING -> (ball passes or lost for 0.5s) -> RETURNING
    RETURNING -> (at ready position) -> IDLE

Two threads:
    1. Vision (60Hz): Camera -> detection -> Kalman filter -> trajectory prediction
    2. Control (50Hz): Read target -> IK -> send joint commands
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import threading
import time

from config import (
    CONTROL_RATE_HZ, DETECTION_FRAMES_THRESHOLD, BALL_LOST_TIMEOUT,
    CALIBRATION_FILE_PATH,
    ARUCO_CALIBRATION_FILE_PATH,
    WORKSPACE_R_MIN, WORKSPACE_R_MAX, WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
)
from camera import RealSenseCamera
from detector import BallDetector
from predictor import BallPredictor
from calibration import load_transform, load_aruco_calibration
from ball_to_robot import VisionToRobotPipeline
from controller import RobotController
from visualizer import Visualizer


def _resolve_calibration_path(path_str: str) -> Path:
    """Resolve calibration file path (try as-is, then next to this script)."""
    p = Path(path_str)
    if p.exists():
        return p
    return Path(__file__).resolve().parent / Path(path_str).name


# Shared state (protected by lock)
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "IDLE"  # IDLE, TRACKING, INTERCEPTING, RETURNING, LOCKED
        self.target_pose = None  # target EE pose for controller
        self.ball_pos_3d = None  # filtered ball position (robot frame)
        self.ball_vel_3d = None  # estimated ball velocity
        self.trajectory = []  # predicted trajectory points
        self.intercept_point = None
        self.detection_count = 0
        self.last_detection_time = 0.0
        self.running = True
        self.locked_position = None  # Locked ball position (spacebar)


def vision_thread(shared, cam, pipeline):
    """
    Vision loop: frames → pipeline (detect → transform → predict → target) → update shared.

    Pipeline: locate in camera → transform to robot frame → Kalman → target pose.
    Sets shared.target_pose so the control thread moves the robot to the ball.
    """
    while shared.running:
        color, depth = cam.get_frames()
        if color is None:
            continue

        r = pipeline.process(color, depth)

        with shared.lock:
            if shared.state == "LOCKED":
                time.sleep(0.01)
                continue

            if r.ball_pos_robot is not None:
                shared.detection_count += 1
                shared.last_detection_time = time.time()
                shared.ball_pos_3d = r.ball_pos_robot.copy()
                shared.ball_vel_3d = r.ball_vel_robot.copy() if r.ball_vel_robot is not None else None

                if shared.detection_count >= DETECTION_FRAMES_THRESHOLD and shared.state == "IDLE":
                    shared.state = "TRACKING"
                    print("[STATE] IDLE -> TRACKING (following ball)")

                if shared.state in ("TRACKING", "INTERCEPTING") and r.target_pose is not None:
                    shared.target_pose = r.target_pose.copy()

            if r.ball_pos_robot is None and shared.state == "TRACKING":
                if (time.time() - shared.last_detection_time) > BALL_LOST_TIMEOUT:
                    shared.state = "RETURNING"
                    shared.target_pose = None
                    shared.detection_count = 0
                    print("[STATE] TRACKING -> RETURNING (ball lost)")

        time.sleep(0.01)


def control_thread(shared, controller):
    """Control loop: execute motion commands based on state."""
    dt = 1.0 / CONTROL_RATE_HZ

    while shared.running:
        loop_start = time.time()

        try:
            with shared.lock:
                state = shared.state
                target = shared.target_pose.copy() if shared.target_pose is not None else None
                locked_pos = shared.locked_position.copy() if shared.locked_position is not None else None

            if state == "IDLE":
                pass  # Stay at current position

            elif state == "LOCKED":
                # Move to locked position
                if locked_pos is not None:
                    # Create target pose from locked position (with neutral orientation)
                    target_pose = np.array([locked_pos[0], locked_pos[1], locked_pos[2], 0.0, 0.0, 0.0])
                    result = controller.move_to_target_pose(target_pose)

            elif state in ("TRACKING", "INTERCEPTING"):
                if target is not None:
                    success = controller.move_to_target_pose(target)
                    if not success[1]:
                        # IK failed, skip this frame
                        pass

            elif state == "RETURNING":
                at_ready = controller.move_to_ready()
                if at_ready:
                    with shared.lock:
                        shared.state = "IDLE"
                        print("[STATE] RETURNING -> IDLE")

        except Exception as e:
            print(f"Control error: {e}")
            time.sleep(0.1)  # Back off on error

        # Maintain control rate
        elapsed = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    print("=== Ball Tracking & Interception (Real Robot) ===")

    # Load camera → robot transform (prefer ArUco calibration if present)
    aruco_path = _resolve_calibration_path(ARUCO_CALIBRATION_FILE_PATH)
    legacy_path = _resolve_calibration_path(CALIBRATION_FILE_PATH)

    if aruco_path.exists():
        cam_to_robot_T = load_aruco_calibration(str(aruco_path))
        print("Loaded camera-to-robot transform (ArUco calibration).")
    elif legacy_path.exists():
        cam_to_robot_T = load_transform(str(legacy_path))
        print("Loaded camera-to-robot transform (legacy calibration).")
    else:
        print("ERROR: No calibration file found.")
        print(f"  ArUco: {aruco_path} (run aruco_calibration.py)")
        print(f"  Legacy: {legacy_path} (run calibrate_camera.py)")
        return

    # Initialize components
    cam = RealSenseCamera()
    cam.start()
    print("Camera started.")

    detector = BallDetector()
    predictor = BallPredictor()
    pipeline = VisionToRobotPipeline(
        detector, cam, cam_to_robot_T, predictor, mode="direct"
    )
    controller = RobotController(use_hardware=True)
    visualizer = Visualizer()

    # Move to ready position
    print("Moving to ready position...")
    for _ in range(50):
        controller.move_to_ready()
        time.sleep(0.02)
    print("Ready.")

    # Shared state
    shared = SharedState()

    # Start threads
    vision_t = threading.Thread(target=vision_thread,
                                args=(shared, cam, pipeline),
                                daemon=True)
    control_t = threading.Thread(target=control_thread,
                                 args=(shared, controller),
                                 daemon=True)

    vision_t.start()
    control_t.start()
    print("Threads started. Robot auto-follows ball after 3 detections.")
    print("'q'=quit, SPACE=lock position (stop follow), 'r'=return to ready.")

    try:
        while shared.running:
            # Main thread handles visualization
            color, depth = cam.get_frames()
            if color is None:
                continue

            with shared.lock:
                state = shared.state
                ball_pos = shared.ball_pos_3d
                ball_vel = shared.ball_vel_3d
                intercept = shared.intercept_point
                det_center = None
                det_radius = None

            # Re-detect for visualization (in main thread for display)
            pos_cam, center, radius, mask = detector.detect(color, depth, cam)

            ee_pos = controller.get_ee_position()

            display = visualizer.draw(
                color, center, radius, mask,
                trajectory_2d=None,  # Would need camera projection for 2D display
                intercept_2d=None,
                state=state,
                ball_pos_3d=ball_pos,
                ball_vel_3d=ball_vel,
                ee_pos=ee_pos
            )

            key = visualizer.show(display)
            if key == ord('q'):
                shared.running = False
                break
            elif key == ord(' '):  # Spacebar - lock current ball position
                with shared.lock:
                    if shared.ball_pos_3d is not None:
                        shared.locked_position = shared.ball_pos_3d.copy()
                        shared.state = "LOCKED"
                        x, y, z = shared.locked_position
                        r = np.sqrt(x**2 + y**2)
                        z_clip = np.clip(z, 0.08, 0.25)  # same as controller
                        in_r = WORKSPACE_R_MIN <= r <= WORKSPACE_R_MAX
                        in_z = WORKSPACE_Z_MIN <= z <= WORKSPACE_Z_MAX
                        print(f"[LOCKED] Ball (robot frame): x={x:.3f} y={y:.3f} z={z:.3f} m")
                        print(f"         r={r:.3f} m (IK uses z_clipped={z_clip:.3f})")
                        print(f"         Workspace: r=[{WORKSPACE_R_MIN}, {WORKSPACE_R_MAX}], z=[{WORKSPACE_Z_MIN}, {WORKSPACE_Z_MAX}]")
                        print(f"         In workspace: r={in_r}, z={in_z}  {'-> IK may fail if outside' if not (in_r and in_z) else ''}")
                    else:
                        print("[LOCKED] No ball detected to lock")
            elif key == ord('r'):  # Reset - unlock and return to ready
                with shared.lock:
                    shared.locked_position = None
                    shared.state = "RETURNING"
                    shared.target_pose = None
                    print("[RESET] Returning to ready position")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        shared.running = False
        vision_t.join(timeout=2.0)
        control_t.join(timeout=2.0)
        cam.stop()
        controller.disconnect()
        visualizer.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
