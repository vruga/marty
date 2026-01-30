#!/usr/bin/env python3
"""
ArUco-based camera-to-world calibration.

Use this when camera is BEHIND the robot and cannot see the end-effector.
Place ArUco markers at known positions (e.g., table corners), run calibration,
then remove markers. The saved transform works without markers.

Usage:
    python aruco_calibration.py              # Run calibration
    python aruco_calibration.py --preview    # Just preview marker detection
    python aruco_calibration.py --generate   # Generate printable markers

Workflow:
    1. Print markers (--generate or use https://chev.me/arucogen/)
    2. Measure marker positions from robot base, edit marker_positions.yaml
    3. Place markers at measured positions
    4. Run calibration (this script)
    5. Remove markers - calibration is saved!
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional

# Import from existing modules
from camera import RealSenseCamera
from config import (
    ARUCO_MARKER_SIZE,
    ARUCO_DICT_TYPE,
    MARKER_POSITIONS_FILE,
    ARUCO_CALIBRATION_FILE_PATH,
)


class ArucoCalibrator:
    """Detects ArUco markers and computes camera-to-world transform."""

    def __init__(
        self,
        marker_size: float = ARUCO_MARKER_SIZE,
        dict_type: int = ARUCO_DICT_TYPE,
    ):
        """
        Args:
            marker_size: Physical marker side length in meters
            dict_type: ArUco dictionary type (0 = DICT_4X4_50)
        """
        self.marker_size = marker_size

        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Will be set from RealSense
        self.camera_matrix = None
        self.dist_coeffs = None

    def set_camera_intrinsics(self, intrinsics):
        """Set camera intrinsics from RealSense."""
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        # RealSense provides distortion coefficients
        self.dist_coeffs = np.array(intrinsics.coeffs)

    def detect_markers(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect ArUco markers and estimate their poses.

        Args:
            image: BGR image

        Returns:
            Dict mapping marker_id -> T_camera_to_marker (4x4 transform)
        """
        if self.camera_matrix is None:
            raise RuntimeError("Camera intrinsics not set. Call set_camera_intrinsics first.")

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(image)

        if ids is None or len(ids) == 0:
            return {}

        # Estimate pose for each marker
        results = {}
        for i, marker_id in enumerate(ids.flatten()):
            # Get rotation and translation vectors
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], self.marker_size, self.camera_matrix, self.dist_coeffs
            )

            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            # Convert to 4x4 transform matrix
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec

            results[marker_id] = T

        return results

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Draw detected markers and their axes on image."""
        output = image.copy()

        # Detect corners for drawing
        corners, ids, _ = self.detector.detectMarkers(image)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)

            # Draw axis for each detection
            for marker_id, T in detections.items():
                rvec, _ = cv2.Rodrigues(T[:3, :3])
                tvec = T[:3, 3]
                cv2.drawFrameAxes(
                    output, self.camera_matrix, self.dist_coeffs,
                    rvec, tvec, self.marker_size * 0.5
                )

                # Draw distance text
                dist = np.linalg.norm(tvec)
                # Find corner for this marker
                for i, mid in enumerate(ids.flatten()):
                    if mid == marker_id:
                        pos = tuple(corners[i][0][0].astype(int))
                        cv2.putText(
                            output, f"ID:{marker_id} d={dist:.2f}m",
                            (pos[0], pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        break

        return output

    def compute_camera_to_world(
        self,
        detections: Dict[int, np.ndarray],
        marker_world_poses: Dict[int, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Compute camera-to-world transform from marker detections.

        Args:
            detections: Dict of marker_id -> T_camera_to_marker
            marker_world_poses: Dict of marker_id -> T_world_to_marker

        Returns:
            Tuple of (T_camera_to_world averaged, list of individual estimates)
        """
        estimates = []

        for marker_id, T_cam_marker in detections.items():
            if marker_id not in marker_world_poses:
                print(f"Warning: Marker {marker_id} detected but not in config, skipping")
                continue

            T_world_marker = marker_world_poses[marker_id]

            # T_camera_to_world = T_world_to_marker @ inv(T_camera_to_marker)
            # Actually: T_cam_to_world = T_world_marker @ inv(T_cam_marker)
            T_cam_to_world = T_world_marker @ np.linalg.inv(T_cam_marker)
            estimates.append(T_cam_to_world)

        if not estimates:
            return None, []

        # Average the transforms
        # For rotation: convert to quaternions, average, normalize
        # For translation: simple average
        translations = np.array([T[:3, 3] for T in estimates])
        avg_translation = np.mean(translations, axis=0)

        rotations = [Rotation.from_matrix(T[:3, :3]) for T in estimates]
        quats = np.array([r.as_quat() for r in rotations])

        # Simple quaternion averaging (works well for similar rotations)
        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)
        avg_rotation = Rotation.from_quat(avg_quat).as_matrix()

        T_avg = np.eye(4)
        T_avg[:3, :3] = avg_rotation
        T_avg[:3, 3] = avg_translation

        return T_avg, estimates


def load_marker_positions(filepath: str) -> Dict[int, np.ndarray]:
    """Load marker world positions from YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    marker_poses = {}
    for marker_id, pose in data['markers'].items():
        marker_id = int(marker_id)

        # Position
        x, y, z = pose['x'], pose['y'], pose['z']

        # Rotation (Euler angles in degrees)
        rx = np.radians(pose.get('rx', 0))
        ry = np.radians(pose.get('ry', 0))
        rz = np.radians(pose.get('rz', 0))

        # Build transform matrix
        R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        marker_poses[marker_id] = T

    return marker_poses


def save_calibration(T_camera_to_world: np.ndarray, filepath: str):
    """Save calibration to JSON file."""
    # Extract rotation as quaternion for compact storage
    R = T_camera_to_world[:3, :3]
    t = T_camera_to_world[:3, 3]
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

    data = {
        'T_camera_to_world': T_camera_to_world.tolist(),
        'rotation_quaternion_xyzw': quat.tolist(),
        'translation_xyz': t.tolist(),
        'description': 'Transform from camera frame to world frame (robot base)',
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Calibration saved to: {filepath}")


def load_calibration(filepath: str) -> np.ndarray:
    """Load calibration from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data['T_camera_to_world'])


def generate_markers(output_dir: str = "aruco_markers", marker_ids: List[int] = [0, 1, 2, 3]):
    """Generate printable ArUco marker images."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Generate each marker
    marker_size_px = 200  # pixels (will print at whatever size you scale to)

    for marker_id in marker_ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

        # Add white border for easier cutting
        border = 50
        bordered = cv2.copyMakeBorder(
            marker_img, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255
        )

        # Add ID label
        cv2.putText(
            bordered, f"ID: {marker_id}",
            (border, bordered.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2
        )

        filepath = output_path / f"aruco_marker_{marker_id}.png"
        cv2.imwrite(str(filepath), bordered)
        print(f"Generated: {filepath}")

    # Generate a single page with all markers
    page_width = 2100  # A4 at ~250 DPI
    page_height = 2970
    page = np.ones((page_height, page_width), dtype=np.uint8) * 255

    # Place markers in a 2x2 grid
    positions = [(200, 200), (1100, 200), (200, 1500), (1100, 1500)]
    marker_print_size = 400  # pixels on page

    for i, marker_id in enumerate(marker_ids[:4]):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_print_size)
        x, y = positions[i]
        page[y:y+marker_print_size, x:x+marker_print_size] = marker_img

        # Label
        cv2.putText(
            page, f"ID: {marker_id}  (50mm side)",
            (x, y + marker_print_size + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2
        )

    # Add instructions
    cv2.putText(page, "ArUco Markers - DICT_4X4_50", (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.putText(page, "Print at 100% scale. Measure actual size after printing!", (200, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    all_markers_path = output_path / "all_markers_A4.png"
    cv2.imwrite(str(all_markers_path), page)
    print(f"Generated: {all_markers_path}")
    print(f"\nPrint 'all_markers_A4.png' at 100% scale on A4 paper.")
    print(f"After printing, MEASURE the actual marker size and update ARUCO_MARKER_SIZE in config.py")


def preview_detection(camera: RealSenseCamera, calibrator: ArucoCalibrator):
    """Preview marker detection without calibrating."""
    print("\n=== ArUco Detection Preview ===")
    print("Press 'q' to quit\n")

    while True:
        color, depth = camera.get_frames()
        if color is None:
            continue

        detections = calibrator.detect_markers(color)
        annotated = calibrator.draw_detections(color, detections)

        # Show detection count
        cv2.putText(
            annotated, f"Detected: {len(detections)} markers",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        for i, (mid, T) in enumerate(detections.items()):
            pos = T[:3, 3]
            cv2.putText(
                annotated, f"M{mid}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m",
                (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        cv2.imshow("ArUco Preview", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_calibration(
    camera: RealSenseCamera,
    calibrator: ArucoCalibrator,
    marker_world_poses: Dict[int, np.ndarray],
    num_samples: int = 30
):
    """
    Run interactive calibration.

    Collects multiple samples and averages for robustness.
    """
    print("\n=== ArUco Calibration ===")
    print(f"Expected markers: {list(marker_world_poses.keys())}")
    print(f"Will collect {num_samples} samples")
    print("\nControls:")
    print("  SPACE - Start/stop collection")
    print("  S     - Save calibration")
    print("  R     - Reset samples")
    print("  Q     - Quit\n")

    all_estimates = []
    collecting = False
    final_transform = None

    while True:
        color, depth = camera.get_frames()
        if color is None:
            continue

        detections = calibrator.detect_markers(color)
        annotated = calibrator.draw_detections(color, detections)

        # Status text
        status = "COLLECTING" if collecting else "PAUSED"
        color_status = (0, 255, 0) if collecting else (0, 165, 255)
        cv2.putText(annotated, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
        cv2.putText(annotated, f"Samples: {len(all_estimates)}/{num_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated, f"Markers visible: {len(detections)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Collect samples
        if collecting and len(detections) >= 2:
            T_avg, estimates = calibrator.compute_camera_to_world(detections, marker_world_poses)
            if T_avg is not None:
                all_estimates.extend(estimates)

                if len(all_estimates) >= num_samples:
                    collecting = False
                    print(f"\nCollected {len(all_estimates)} samples!")

        # Show current estimate if we have samples
        if all_estimates:
            # Recompute average from all samples
            translations = np.array([T[:3, 3] for T in all_estimates])
            avg_t = np.mean(translations, axis=0)
            std_t = np.std(translations, axis=0)

            cv2.putText(annotated, f"Camera pos: [{avg_t[0]:.3f}, {avg_t[1]:.3f}, {avg_t[2]:.3f}]m",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated, f"Std dev: [{std_t[0]:.4f}, {std_t[1]:.4f}, {std_t[2]:.4f}]m",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("ArUco Calibration", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            collecting = not collecting
            print(f"Collection: {'started' if collecting else 'paused'}")

        elif key == ord('r'):
            all_estimates = []
            final_transform = None
            print("Samples reset")

        elif key == ord('s'):
            if len(all_estimates) < 10:
                print("Need at least 10 samples to save!")
                continue

            # Compute final average
            translations = np.array([T[:3, 3] for T in all_estimates])
            avg_translation = np.mean(translations, axis=0)

            rotations = [Rotation.from_matrix(T[:3, :3]) for T in all_estimates]
            quats = np.array([r.as_quat() for r in rotations])
            avg_quat = np.mean(quats, axis=0)
            avg_quat /= np.linalg.norm(avg_quat)
            avg_rotation = Rotation.from_quat(avg_quat).as_matrix()

            final_transform = np.eye(4)
            final_transform[:3, :3] = avg_rotation
            final_transform[:3, 3] = avg_translation

            save_calibration(final_transform, ARUCO_CALIBRATION_FILE_PATH)

            # Print summary
            print("\n" + "="*50)
            print("CALIBRATION COMPLETE")
            print("="*50)
            print(f"Samples used: {len(all_estimates)}")
            print(f"Translation std: {np.std(translations, axis=0) * 1000} mm")
            print(f"\nCamera position in world frame:")
            print(f"  X: {avg_translation[0]:.4f} m")
            print(f"  Y: {avg_translation[1]:.4f} m")
            print(f"  Z: {avg_translation[2]:.4f} m")
            print("="*50)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return final_transform


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ArUco camera calibration')
    parser.add_argument('--preview', action='store_true',
                        help='Preview marker detection only')
    parser.add_argument('--generate', action='store_true',
                        help='Generate printable markers')
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of samples to collect')
    args = parser.parse_args()

    # Generate markers and exit
    if args.generate:
        generate_markers()
        return

    # Initialize camera
    camera = RealSenseCamera()
    camera.start()
    print(f"Camera started: {camera.width}x{camera.height}")

    # Initialize calibrator
    calibrator = ArucoCalibrator()
    calibrator.set_camera_intrinsics(camera.intrinsics)
    print(f"Marker size: {ARUCO_MARKER_SIZE * 100:.1f} cm")

    try:
        if args.preview:
            preview_detection(camera, calibrator)
        else:
            # Load marker positions
            marker_poses = load_marker_positions(MARKER_POSITIONS_FILE)
            print(f"Loaded {len(marker_poses)} marker positions from {MARKER_POSITIONS_FILE}")

            run_calibration(camera, calibrator, marker_poses, args.samples)
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
