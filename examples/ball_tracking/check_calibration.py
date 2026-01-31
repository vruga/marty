#!/usr/bin/env python3
"""
Check that camera-to-robot (ArUco) calibration is correct.

Projects known marker positions (from marker_positions.yaml) from robot frame
into the camera image. If calibration is good, the projected circles should
overlay the actual markers. Also prints ball position in robot frame when detected.

Usage:
    python check_calibration.py
    Press 'q' to quit.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import cv2

from config import ARUCO_CALIBRATION_FILE_PATH, MARKER_POSITIONS_FILE
from calibration import load_aruco_calibration
from camera import RealSenseCamera
from detector import BallDetector


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    return Path(__file__).resolve().parent / Path(path_str).name


def load_marker_centers(filepath: str, axis_remap: str = "xyz"):
    """Load marker (x,y,z) in robot frame from YAML.
    axis_remap: "xyz" = no change; "zxy" = use (z,x,y); "xzy" = (x,z,y); "-xyz" = (-x,y,z).
    Use if circles are way off (e.g. wrong frame convention).
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    out = {}
    for mid, pose in data['markers'].items():
        x, y, z = pose['x'], pose['y'], pose['z']
        if axis_remap == "xyz":
            p = np.array([x, y, z], dtype=np.float64)
        elif axis_remap == "zxy":
            p = np.array([z, x, y], dtype=np.float64)  # forward=Z, left=X, up=Y
        elif axis_remap == "xzy":
            p = np.array([x, z, y], dtype=np.float64)
        elif axis_remap == "-xyz":
            p = np.array([-x, y, z], dtype=np.float64)
        elif axis_remap == "x-yz":
            p = np.array([x, -y, z], dtype=np.float64)
        else:
            p = np.array([x, y, z], dtype=np.float64)
        out[int(mid)] = p
    return out


def robot_to_camera(p_robot: np.ndarray, T_cam_to_world: np.ndarray) -> np.ndarray:
    """Transform 3D point from robot/world frame to camera frame."""
    p = np.array([p_robot[0], p_robot[1], p_robot[2], 1.0])
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    q = T_world_to_cam @ p
    return q[:3]


def project_to_image(p_cam: np.ndarray, fx: float, fy: float, ppx: float, ppy: float):
    """Project 3D point in camera frame to (u, v) pixel. Returns None if behind camera."""
    if p_cam[2] <= 0:
        return None
    u = fx * p_cam[0] / p_cam[2] + ppx
    v = fy * p_cam[1] / p_cam[2] + ppy
    return (int(round(u)), int(round(v)))


def main():
    import os
    # If circles are way off-screen, try: export CHECK_CALIB_AXIS=zxy  or  xzy  or  -xyz  or  x-yz
    axis_remap = os.environ.get("CHECK_CALIB_AXIS", "xyz")

    print("=== Calibration check ===")
    print("Known marker positions (robot frame) are projected into the image.")
    print("If calibration is correct, the colored circles should sit ON the markers.")
    print("Press 'q' to quit.\n")

    # Load calibration and marker positions
    aruco_path = _resolve_path(ARUCO_CALIBRATION_FILE_PATH)
    if not aruco_path.exists():
        print(f"ERROR: {aruco_path} not found. Run aruco_calibration.py first.")
        return
    T_cam_to_world = load_aruco_calibration(str(aruco_path))
    cam_pos_world = T_cam_to_world[:3, 3]
    print(f"Camera position in world (robot) frame: x={cam_pos_world[0]:.3f} y={cam_pos_world[1]:.3f} z={cam_pos_world[2]:.3f} m\n")

    markers_path = _resolve_path(MARKER_POSITIONS_FILE)
    if not markers_path.exists():
        print(f"ERROR: {markers_path} not found.")
        return
    marker_centers = load_marker_centers(str(markers_path), axis_remap=axis_remap)
    if axis_remap != "xyz":
        print(f"Using axis remap: {axis_remap}\n")

    cam = RealSenseCamera()
    cam.start()
    detector = BallDetector()

    # Pipeline to get ball in robot frame (same as main)
    from calibration import transform_point
    pipeline_T = T_cam_to_world

    intrinsics = cam.intrinsics
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy

    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # BGR per marker ID
    h, w = None, None
    frame_count = 0
    debug_printed = False

    while True:
        color, depth = cam.get_frames()
        if color is None:
            continue
        if h is None:
            h, w = color.shape[:2]

        # Project each marker position into the image
        display = color.copy()
        for mid, p_robot in marker_centers.items():
            p_cam = robot_to_camera(p_robot, T_cam_to_world)
            uv = project_to_image(p_cam, fx, fy, ppx, ppy)

            # Debug: print once so we can see why only ID3 shows / wrong position
            if not debug_printed and frame_count == 5:
                in_front = p_cam[2] > 0
                uv_str = f"({uv[0]},{uv[1]})" if uv else "behind camera"
                print(f"  ID{mid}: robot=({p_robot[0]:.3f},{p_robot[1]:.3f},{p_robot[2]:.3f}) "
                      f"-> cam=({p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}) Z>0={in_front} -> uv={uv_str}")

            c = colors[mid % len(colors)]
            if uv is not None:
                # Clip to image so we always see something; if off-screen, draw at edge
                u, v = uv[0], uv[1]
                u_clip = max(20, min(w - 20, u))
                v_clip = max(20, min(h - 20, v))
                cv2.circle(display, (u_clip, v_clip), 15, c, 2)
                cv2.putText(display, f"ID{mid}", (u_clip - 10, v_clip - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                if (u, v) != (u_clip, v_clip):
                    cv2.putText(display, "off-screen", (u_clip, v_clip + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
            else:
                # Behind camera: draw label at fixed position so we see all IDs
                cv2.putText(display, f"ID{mid}: behind", (10, 80 + mid * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        if frame_count == 5 and not debug_printed:
            debug_printed = True
            print("\nIf circles are OFF-SCREEN or wrong:")
            print("  1. YAML and calibration must match: marker_positions.yaml must be the exact")
            print("     positions (in meters, robot frame) where the markers were when you ran")
            print("     aruco_calibration.py. If you changed the YAML after calibrating, re-run:")
            print("       python aruco_calibration.py   (place markers at positions in YAML, then S to save)")
            print("  2. Try a different axis convention: CHECK_CALIB_AXIS=zxy  or  xzy  or  -xyz")
            print("     Example: CHECK_CALIB_AXIS=zxy python check_calibration.py\n")
        frame_count += 1

        # Detect ball and show position in robot frame
        pos_cam, center, radius, mask = detector.detect(color, depth, cam)
        if pos_cam is not None:
            pos_robot = transform_point(pipeline_T, pos_cam)
            cv2.putText(display, f"Ball (robot): [{pos_robot[0]:.3f}, {pos_robot[1]:.3f}, {pos_robot[2]:.3f}] m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, "Circles = marker positions from YAML (should overlay markers)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow("Calibration check", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.stop()
    print("Done.")


if __name__ == "__main__":
    main()
