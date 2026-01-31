"""Robot control: IK wrapper, motion smoothing, ready position, safety limits."""

import numpy as np
from pathlib import Path
from config import (
    READY_QPOS, CONTROL_QLIMIT, MAX_JOINT_CHANGE,
    ROBOT_PORT, CALIBRATION_FILE, ROBOT_TYPE, RETURN_TOLERANCE
)
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot, feetech_arm
from planner import compute_yaw_from_target


def _resolve_robot_calibration_path(path_str: str) -> str:
    """Resolve robot calibration file path (main_follower.json lives in examples/, not ball_tracking/)."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    # When run from examples/ball_tracking/, config has "examples/main_follower.json" -> look in parent dir
    script_dir = Path(__file__).resolve().parent
    alt = script_dir.parent / Path(path_str).name  # examples/ball_tracking/../main_follower.json
    if alt.exists():
        return str(alt)
    return path_str  # let feetech_arm report the error


class RobotController:
    """Wrapper for SO-100 arm control with IK, smoothing, and safety."""

    def __init__(self, use_hardware=True):
        self.robot_kin = get_robot(ROBOT_TYPE)
        self.current_qpos = READY_QPOS.copy()
        self.use_hardware = use_hardware
        self.arm = None

        if use_hardware:
            calibration_path = _resolve_robot_calibration_path(CALIBRATION_FILE)
            self.arm = feetech_arm(driver_port=ROBOT_PORT, calibration_file=calibration_path)

    def get_current_qpos(self):
        """Get current joint positions (from hardware feedback or internal state)."""
        if self.use_hardware and self.arm is not None:
            feedback = self.arm.feedback()
            self.current_qpos = feedback.copy()
        return self.current_qpos.copy()

    def move_to_target_pose(self, target_pose):
        """Move end-effector to target pose using IK.

        Separates yaw (joint 0) from 4-DOF IK (joints 1-4).

        Args:
            target_pose: np.array([x, y, z, roll, pitch, yaw])

        Returns:
            tuple: (new_qpos, success)
        """
        # Compute joint 0 (yaw) from target XY position
        target_yaw = compute_yaw_from_target(target_pose[:3])
        target_yaw = np.clip(target_yaw, CONTROL_QLIMIT[0][0], CONTROL_QLIMIT[1][0])

        # For IK, we work in the rotated frame (remove yaw from target)
        # Transform target into the frame after yaw rotation
        x, y, z = target_pose[0], target_pose[1], target_pose[2]
        r = np.sqrt(x**2 + y**2)

        # Clamp z to safe range (avoid very low positions that cause IK issues)
        z = np.clip(z, 0.08, 0.25)

        # In the yaw-rotated frame, the target is at (r, 0, z)
        # Use fixed orientation for paddle (facing forward, slight tilt)
        ik_target = np.array([r, 0.0, z, 0.0, 0.0, 0.0])

        # Current joints 1-4 as seed
        q_seed = self.current_qpos[1:5]

        # Run IK
        try:
            q_new, success = lerobot_IK(q_seed, ik_target, robot=self.robot_kin)
        except Exception as e:
            print(f"IK exception: {e}")
            return self.current_qpos.copy(), False

        if not success or np.all(q_new == -1.0):
            print(f"IK failed for target x={x:.3f} y={y:.3f} z={z:.3f} -> r={r:.3f} z_clipped={z:.3f} (workspace r=[0.12,0.32], z~[0.08,0.25])")
            return self.current_qpos.copy(), False

        # Safety check: reject IK solutions too far from current position
        max_jump = np.max(np.abs(q_new[:4] - q_seed))
        if max_jump > 1.0:  # More than 1 radian jump = reject
            print(f"IK solution rejected: jump={max_jump:.2f} rad too large")
            return self.current_qpos.copy(), False

        # Apply smoothing with conservative step size
        q_smoothed = self._smooth_motion(q_seed, q_new[:4])

        # Assemble full 6-DOF qpos
        new_qpos = self.current_qpos.copy()
        # Smooth yaw separately
        yaw_delta = target_yaw - self.current_qpos[0]
        if abs(yaw_delta) > MAX_JOINT_CHANGE:
            yaw_delta = np.sign(yaw_delta) * MAX_JOINT_CHANGE
        new_qpos[0] = self.current_qpos[0] + yaw_delta
        new_qpos[1:5] = q_smoothed

        # Apply joint limits
        new_qpos = self._apply_limits(new_qpos)

        # Send to hardware
        self._send_command(new_qpos)
        self.current_qpos = new_qpos.copy()

        return new_qpos, True

    def move_to_ready(self):
        """Move toward ready position. Returns True when at ready position."""
        target = READY_QPOS.copy()
        diff = target - self.current_qpos

        # Check if already at ready
        if np.max(np.abs(diff)) < RETURN_TOLERANCE:
            return True

        # Smooth motion toward ready
        new_qpos = self.current_qpos.copy()
        for i in range(6):
            delta = diff[i]
            if abs(delta) > MAX_JOINT_CHANGE:
                delta = np.sign(delta) * MAX_JOINT_CHANGE
            new_qpos[i] = self.current_qpos[i] + delta

        new_qpos = self._apply_limits(new_qpos)
        self._send_command(new_qpos)
        self.current_qpos = new_qpos.copy()

        return False

    def set_qpos_direct(self, qpos):
        """Directly set joint positions (for simulation use)."""
        self.current_qpos = qpos.copy()
        self._send_command(qpos)

    def _smooth_motion(self, q_current, q_target):
        """Apply per-joint velocity limiting."""
        q_out = q_current.copy()
        for i in range(len(q_target)):
            delta = q_target[i] - q_current[i]
            if abs(delta) > MAX_JOINT_CHANGE:
                delta = np.sign(delta) * MAX_JOINT_CHANGE
            q_out[i] = q_current[i] + delta
        return q_out

    def _apply_limits(self, qpos):
        """Clamp joint positions to limits."""
        for i in range(6):
            qpos[i] = np.clip(qpos[i], CONTROL_QLIMIT[0][i], CONTROL_QLIMIT[1][i])
        return qpos

    def _send_command(self, qpos):
        """Send joint command to hardware."""
        if self.use_hardware and self.arm is not None:
            self.arm.action(qpos)

    def get_ee_position(self):
        """Get current end-effector position via FK."""
        fk = lerobot_FK(self.current_qpos[1:5], robot=self.robot_kin)
        x, y, z = fk[0], fk[1], fk[2]
        yaw = self.current_qpos[0]
        x_rot = x * np.cos(yaw) - y * np.sin(yaw)
        y_rot = x * np.sin(yaw) + y * np.cos(yaw)
        return np.array([x_rot, y_rot, z])

    def disconnect(self):
        """Disconnect from hardware."""
        if self.arm is not None:
            self.arm.disconnect()
