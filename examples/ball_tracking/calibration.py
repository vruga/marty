"""Camera-to-robot rigid transform using SVD-based Kabsch algorithm.

Frames:
  - Camera: X right, Y down, Z forward (RealSense). Meters.
  - Robot:   X forward, Y left, Z up (from FK + base yaw). Meters.

Transform: p_robot = R @ p_camera + t  =>  T @ [p_cam; 1] = [R @ p_cam + t; 1]
"""

import numpy as np
import json
import os


def compute_rigid_transform(camera_points, robot_points):
    """Compute rigid transform (R, t) from camera frame to robot frame using SVD.

    Kabsch: finds R (rotation, det=+1) and t such that p_robot = R @ p_camera + t
    by minimizing sum of squared distances over corresponding pairs.

    Args:
        camera_points: Nx3 array of 3D points in camera frame (meters)
        robot_points: Nx3 array of corresponding 3D points in robot frame (meters)

    Returns:
        np.ndarray: 4x4 homogeneous matrix T (camera -> robot). p_robot = (T @ [p;1])[:3]
    """
    assert camera_points.shape == robot_points.shape
    assert camera_points.shape[0] >= 3

    # Centroids
    centroid_cam = np.mean(camera_points, axis=0)
    centroid_rob = np.mean(robot_points, axis=0)

    # Center the points
    cam_centered = camera_points - centroid_cam
    rob_centered = robot_points - centroid_rob

    # Compute cross-covariance matrix
    H = cam_centered.T @ rob_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Rotation matrix
    R = Vt.T @ sign_matrix @ U.T

    # Translation
    t = centroid_rob - R @ centroid_cam

    # Build 4x4 homogeneous transform: p_robot = T @ [p_cam; 1]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # Sanity: rotation must be proper (det = +1)
    det_r = np.linalg.det(R)
    if abs(det_r - 1.0) > 0.01:
        raise RuntimeError(f"Kabsch produced invalid rotation: det(R)={det_r:.4f} (expected 1.0)")

    return T


def save_transform(T, filepath):
    """Save 4x4 transform matrix to JSON file.

    Args:
        T: 4x4 homogeneous transformation matrix
        filepath: output JSON file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {"transform_4x4": T.tolist()}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved transform to {filepath}")


def load_transform(filepath, validate=True):
    """Load 4x4 transform matrix from JSON file (legacy format: transform_4x4).

    Args:
        filepath: input JSON file path
        validate: if True, warn when det(R) is not ~1 (invalid rotation)

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix (camera -> robot)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    T = np.array(data["transform_4x4"])
    assert T.shape == (4, 4)
    if validate:
        det_r = np.linalg.det(T[:3, :3])
        if abs(det_r - 1.0) > 0.01:
            print(f"[calibration] WARNING: loaded T has det(R)={det_r:.4f} (expected 1.0); recompute with calibrate_camera.py")
    return T


def load_aruco_calibration(filepath, validate=True):
    """Load camera-to-world (robot base) transform from ArUco calibration JSON.

    Use this when you calibrated with aruco_calibration.py (four markers on table).
    World frame = robot base frame, so T_camera_to_world is camera → robot.

    Args:
        filepath: path to camera_to_world.json
        validate: if True, warn when det(R) is not ~1

    Returns:
        np.ndarray: 4x4 T (camera → robot/world)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    T = np.array(data["T_camera_to_world"])
    assert T.shape == (4, 4)
    if validate:
        det_r = np.linalg.det(T[:3, :3])
        if abs(det_r - 1.0) > 0.01:
            print(f"[calibration] WARNING: ArUco T has det(R)={det_r:.4f} (expected 1.0)")
    return T


def transform_point(T, point):
    """Apply 4x4 transform to a 3D point.

    Args:
        T: 4x4 homogeneous transformation matrix
        point: 3D point as array-like [x, y, z]

    Returns:
        np.ndarray: Transformed 3D point [x, y, z]
    """
    p_hom = np.array([point[0], point[1], point[2], 1.0])
    result = T @ p_hom
    return result[:3]
