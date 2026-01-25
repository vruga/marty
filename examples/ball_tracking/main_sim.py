"""MuJoCo simulation mode: simulated ball + arm, no hardware needed.

Usage:
    python main_sim.py

Simulates a table tennis ball launched toward the robot arm.
The full tracking/prediction/interception pipeline runs using
the ball's position read directly from the MuJoCo simulation state.

Press 'space' in the viewer to launch a new ball.
Press 'r' to reset the ball.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import mujoco.viewer
import time

from config import (
    SIM_XML_PATH, SIM_BALL_INITIAL_POS, SIM_BALL_VELOCITY,
    READY_QPOS, CONTROL_RATE_HZ, DETECTION_FRAMES_THRESHOLD,
    BALL_LOST_TIMEOUT, PREDICTION_DT, MAX_JOINT_CHANGE
)
from predictor import BallPredictor
from planner import find_intercept_point, compute_paddle_pose, compute_yaw_from_target
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot

os.environ["MUJOCO_GL"] = "egl"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


class SimController:
    """Simplified controller for simulation (no hardware)."""

    def __init__(self, robot_kin):
        self.robot_kin = robot_kin
        self.current_qpos = READY_QPOS.copy()

    def move_to_target_pose(self, target_pose):
        """Compute IK and smooth motion toward target pose."""
        target_yaw = compute_yaw_from_target(target_pose[:3])
        target_yaw = np.clip(target_yaw, -2.1, 2.1)

        x, y, z = target_pose[0], target_pose[1], target_pose[2]
        r = np.sqrt(x**2 + y**2)
        ik_target = np.array([r, 0.0, z, target_pose[3], target_pose[4], 0.0])

        q_seed = self.current_qpos[1:5]
        try:
            q_new, success = lerobot_IK(q_seed, ik_target, robot=self.robot_kin)
        except Exception:
            return self.current_qpos.copy(), False

        if not success or np.all(q_new == -1.0):
            return self.current_qpos.copy(), False

        # Safety check: reject solutions with large jumps
        if np.max(np.abs(q_new[:4] - q_seed)) > 1.0:
            return self.current_qpos.copy(), False

        # Smooth motion
        new_qpos = self.current_qpos.copy()
        yaw_delta = target_yaw - self.current_qpos[0]
        if abs(yaw_delta) > MAX_JOINT_CHANGE:
            yaw_delta = np.sign(yaw_delta) * MAX_JOINT_CHANGE
        new_qpos[0] = self.current_qpos[0] + yaw_delta

        for i in range(4):
            delta = q_new[i] - self.current_qpos[i + 1]
            if abs(delta) > MAX_JOINT_CHANGE:
                delta = np.sign(delta) * MAX_JOINT_CHANGE
            new_qpos[i + 1] = self.current_qpos[i + 1] + delta

        self.current_qpos = new_qpos.copy()
        return new_qpos, True

    def move_to_ready(self):
        """Move toward ready position."""
        target = READY_QPOS.copy()
        diff = target - self.current_qpos
        if np.max(np.abs(diff)) < 0.05:
            return True

        new_qpos = self.current_qpos.copy()
        for i in range(6):
            delta = diff[i]
            if abs(delta) > MAX_JOINT_CHANGE:
                delta = np.sign(delta) * MAX_JOINT_CHANGE
            new_qpos[i] = self.current_qpos[i] + delta
        self.current_qpos = new_qpos.copy()
        return False


def launch_ball(mjdata, ball_qpos_start, ball_vel_start, ball_pos_idx, ball_vel_idx):
    """Reset ball position and apply initial velocity."""
    mjdata.qpos[ball_pos_idx:ball_pos_idx + 3] = ball_qpos_start
    # Reset quaternion to identity
    mjdata.qpos[ball_pos_idx + 3:ball_pos_idx + 7] = [1, 0, 0, 0]
    # Set velocity
    mjdata.qvel[ball_vel_idx:ball_vel_idx + 3] = ball_vel_start
    mjdata.qvel[ball_vel_idx + 3:ball_vel_idx + 6] = [0, 0, 0]


def main():
    print("=== Ball Tracking Simulation (MuJoCo) ===")
    print("Controls:")
    print("  Space: Launch ball")
    print("  R: Reset ball")
    print("  Q: Quit")

    # Load model
    mjmodel = mujoco.MjModel.from_xml_path(SIM_XML_PATH)
    mjdata = mujoco.MjData(mjmodel)

    # Get joint indices
    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])

    # Get ball body/joint indices
    ball_joint_id = mjmodel.joint("ball_joint").id
    ball_pos_idx = mjmodel.jnt_qposadr[ball_joint_id]  # free joint: 7 DOF (pos + quat)
    ball_vel_idx = mjmodel.jnt_dofadr[ball_joint_id]  # free joint: 6 DOF (vel + angvel)
    ball_body_id = mjmodel.body("ball").id

    # Initialize robot
    robot_kin = get_robot('so100')
    sim_controller = SimController(robot_kin)
    predictor = BallPredictor()

    # State machine
    state = "IDLE"
    detection_count = 0
    last_detection_time = 0.0
    target_pose = None
    ball_launched = False

    # Set initial robot position
    mjdata.qpos[qpos_indices] = READY_QPOS
    mujoco.mj_step(mjmodel, mjdata)

    # Key callback state
    key_pressed = {"space": False, "r": False}

    def key_callback(keycode):
        if keycode == 32:  # space
            key_pressed["space"] = True
        elif keycode == 82 or keycode == 114:  # R or r
            key_pressed["r"] = True

    try:
        with mujoco.viewer.launch_passive(mjmodel, mjdata, key_callback=key_callback) as viewer:
            start_time = time.time()

            while viewer.is_running():
                step_start = time.time()

                # Handle key presses
                if key_pressed["space"]:
                    key_pressed["space"] = False
                    # Add some randomness to ball launch
                    vel_noise = np.random.randn(3) * 0.2
                    ball_vel = SIM_BALL_VELOCITY + vel_noise
                    pos_noise = np.random.randn(3) * 0.05
                    ball_pos = SIM_BALL_INITIAL_POS + pos_noise
                    launch_ball(mjdata, ball_pos, ball_vel, ball_pos_idx, ball_vel_idx)
                    ball_launched = True
                    predictor.reset()
                    detection_count = 0
                    state = "IDLE"
                    print(f"Ball launched! vel={ball_vel}")

                if key_pressed["r"]:
                    key_pressed["r"] = False
                    launch_ball(mjdata, SIM_BALL_INITIAL_POS, np.zeros(3), ball_pos_idx, ball_vel_idx)
                    ball_launched = False
                    predictor.reset()
                    detection_count = 0
                    state = "IDLE"
                    target_pose = None
                    print("Ball reset.")

                # --- Read ball position and velocity from simulation ---
                ball_pos_3d = mjdata.xpos[ball_body_id].copy()
                ball_vel_sim = mjdata.qvel[ball_vel_idx:ball_vel_idx + 3].copy()
                ball_speed = np.linalg.norm(ball_vel_sim)

                # Only process if ball is above ground and in a reasonable range
                ball_visible = (ball_pos_3d[2] > 0.01 and
                                np.linalg.norm(ball_pos_3d[:2]) < 2.0 and
                                ball_launched)

                # Check if ball has passed robot or stopped (for state transitions)
                ball_passed = (ball_pos_3d[0] < -0.1 or  # passed behind robot
                               (ball_pos_3d[2] < 0.05 and ball_speed < 0.1) or  # on ground, stopped
                               np.linalg.norm(ball_pos_3d[:2]) > 1.5)  # too far away

                if ball_visible and not ball_passed:
                    # Update Kalman filter with simulated "detection"
                    filtered_state = predictor.update(ball_pos_3d)
                    last_detection_time = time.time()
                    detection_count += 1

                    # Predict trajectory
                    trajectory = predictor.predict_trajectory()

                    # Find intercept
                    intercept, idx = find_intercept_point(trajectory)

                    # State machine
                    if state == "IDLE":
                        if detection_count >= DETECTION_FRAMES_THRESHOLD:
                            state = "TRACKING"
                            print(f"[STATE] IDLE -> TRACKING")

                    if state == "TRACKING":
                        if intercept is not None:
                            vel = predictor.get_velocity()
                            target_pose = compute_paddle_pose(intercept, vel)
                            state = "INTERCEPTING"
                            print(f"[STATE] TRACKING -> INTERCEPTING at "
                                  f"[{intercept[0]:.3f}, {intercept[1]:.3f}, {intercept[2]:.3f}]")

                    if state == "INTERCEPTING":
                        if intercept is not None:
                            vel = predictor.get_velocity()
                            target_pose = compute_paddle_pose(intercept, vel)

                else:
                    # Ball not visible or has passed
                    if state in ("TRACKING", "INTERCEPTING"):
                        if ball_passed:
                            # Immediate transition when ball has clearly passed
                            state = "RETURNING"
                            target_pose = None
                            detection_count = 0
                            predictor.reset()
                            print("[STATE] -> RETURNING (ball passed)")
                        else:
                            elapsed = time.time() - last_detection_time
                            if elapsed > BALL_LOST_TIMEOUT:
                                state = "RETURNING"
                                target_pose = None
                                detection_count = 0
                                predictor.reset()
                                print("[STATE] -> RETURNING (timeout)")

                # --- Control ---
                if state == "IDLE":
                    pass
                elif state in ("TRACKING", "INTERCEPTING"):
                    if target_pose is not None:
                        new_qpos, success = sim_controller.move_to_target_pose(target_pose)
                        if success:
                            mjdata.qpos[qpos_indices] = new_qpos
                elif state == "RETURNING":
                    at_ready = sim_controller.move_to_ready()
                    mjdata.qpos[qpos_indices] = sim_controller.current_qpos
                    if at_ready:
                        state = "IDLE"
                        print("[STATE] RETURNING -> IDLE")

                # Step simulation
                mujoco.mj_step(mjmodel, mjdata)
                viewer.sync()

                # Maintain step rate
                elapsed = time.time() - step_start
                sleep_time = mjmodel.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nSimulation ended.")

    print("Done.")


if __name__ == "__main__":
    main()
