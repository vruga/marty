"""
Pipeline: camera frames → ball 3D in robot frame → target pose for the robot.

This is the single place that:
  1. Locate:  detect ball (pixel + depth) → 3D in camera frame
  2. Transform: camera 3D → robot 3D via calibration T (camera_to_robot)
  3. (Optional) filter/predict: Kalman for smooth position + velocity
  4. Produce target: pose [x,y,z,roll,pitch,yaw] so the controller can move the EE to the ball

Use VisionToRobotPipeline.process(color, depth) and use .target_pose to drive the robot.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from calibration import transform_point
from planner import find_intercept_point, compute_paddle_pose


@dataclass
class BallToRobotResult:
    """Result of processing one frame: ball in robot frame and suggested target pose."""

    ball_pos_robot: Optional[np.ndarray]  # [x,y,z] in robot frame (meters), or None
    ball_vel_robot: Optional[np.ndarray]   # [vx,vy,vz] if predictor used, else None
    target_pose: Optional[np.ndarray]     # [x,y,z,roll,pitch,yaw] for controller, or None
    # Raw detection (for viz / debug)
    center: Optional[tuple]               # (cx, cy) pixel or None
    radius: Optional[int]                 # pixels or None
    mask: Optional[np.ndarray]           # binary mask
    pos_camera: Optional[np.ndarray]     # [x,y,z] in camera frame before transform


class VisionToRobotPipeline:
    """
    Process frames: detect ball → transform to robot frame → (optional) filter/predict
    → produce target pose for the robot.

    - detector: BallDetector
    - camera: RealSenseCamera (for deprojection inside detector)
    - cam_to_robot_T: 4x4 transform from calibration (camera → robot)
    - predictor: BallPredictor or None; if set, use Kalman and can do intercept mode
    - mode: "direct" = target is current ball position; "intercept" = target is first
      workspace intercept from predicted trajectory (needs predictor)
    """

    def __init__(
        self,
        detector,
        camera,
        cam_to_robot_T: np.ndarray,
        predictor=None,
        mode: str = "direct",
    ):
        self.detector = detector
        self.camera = camera
        self.T = cam_to_robot_T
        self.predictor = predictor
        self.mode = mode if mode in ("direct", "intercept") else "direct"

    def process(self, color: np.ndarray, depth: np.ndarray) -> BallToRobotResult:
        """
        Run: detect → transform to robot → (predict) → build target pose.

        Returns:
            BallToRobotResult with ball_pos_robot, ball_vel_robot, target_pose, and viz fields.
        """
        pos_cam, center, radius, mask = self.detector.detect(color, depth, self.camera)

        if pos_cam is None:
            return BallToRobotResult(
                ball_pos_robot=None,
                ball_vel_robot=None,
                target_pose=None,
                center=center,
                radius=radius,
                mask=mask,
                pos_camera=None,
            )

        # 2. Transform: camera frame → robot frame (the core calibration step)
        pos_robot = transform_point(self.T, pos_cam)

        ball_vel = None
        if self.predictor is not None:
            state = self.predictor.update(pos_robot)
            ball_pos = state[:3].copy()
            ball_vel = state[3:6].copy()
        else:
            ball_pos = pos_robot.copy()

        # 4. Produce target pose for the controller
        target_pose = self._make_target(ball_pos, ball_vel)

        return BallToRobotResult(
            ball_pos_robot=ball_pos,
            ball_vel_robot=ball_vel,
            target_pose=target_pose,
            center=center,
            radius=radius,
            mask=mask,
            pos_camera=pos_cam.copy(),
        )

    def _make_target(
        self,
        ball_pos: np.ndarray,
        ball_vel: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if self.mode == "direct":
            # Go to current ball position; orientation neutral
            return np.array([ball_pos[0], ball_pos[1], ball_pos[2], 0.0, 0.0, 0.0])

        if self.mode == "intercept" and self.predictor is not None and ball_vel is not None:
            traj = self.predictor.predict_trajectory()
            intercept_pt, idx = find_intercept_point(traj)
            if intercept_pt is not None:
                # Approximate velocity at intercept (use current vel; trajectory doesn't store it)
                return compute_paddle_pose(intercept_pt, ball_vel)
        return None
