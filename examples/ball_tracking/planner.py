"""Find intercept point in robot workspace, compute paddle target pose."""

import numpy as np
from config import (
    WORKSPACE_R_MIN, WORKSPACE_R_MAX,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX
)


def is_in_workspace(point):
    """Check if a 3D point is within the robot's reachable workspace.

    Workspace is defined as a cylindrical region:
    - Radial distance (XY plane): [WORKSPACE_R_MIN, WORKSPACE_R_MAX]
    - Height (Z): [WORKSPACE_Z_MIN, WORKSPACE_Z_MAX]

    Args:
        point: np.array([x, y, z]) in robot frame

    Returns:
        bool: True if point is in workspace
    """
    r = np.sqrt(point[0]**2 + point[1]**2)
    z = point[2]
    return (WORKSPACE_R_MIN <= r <= WORKSPACE_R_MAX and
            WORKSPACE_Z_MIN <= z <= WORKSPACE_Z_MAX)


def find_intercept_point(trajectory):
    """Find the first point in the predicted trajectory that enters the workspace.

    Args:
        trajectory: list of np.array([x,y,z]) predicted positions

    Returns:
        tuple: (intercept_point, intercept_index) or (None, -1) if no intercept
    """
    for i, point in enumerate(trajectory):
        if is_in_workspace(point):
            return point.copy(), i
    return None, -1


def compute_paddle_pose(intercept_point, ball_velocity):
    """Compute target end-effector pose for paddle to face the incoming ball.

    The paddle should be oriented perpendicular to the ball's velocity vector
    at the intercept point.

    Args:
        intercept_point: np.array([x, y, z]) target position in robot frame
        ball_velocity: np.array([vx, vy, vz]) ball velocity at intercept

    Returns:
        np.array([x, y, z, roll, pitch, yaw]): Target pose for IK
    """
    x, y, z = intercept_point

    # Compute yaw angle (rotation around Z) to face the ball approach direction
    # The ball is coming from direction of velocity, paddle faces opposite
    if abs(ball_velocity[0]) > 0.01 or abs(ball_velocity[1]) > 0.01:
        approach_yaw = np.arctan2(-ball_velocity[1], -ball_velocity[0])
    else:
        # Default: face forward (positive X)
        approach_yaw = np.arctan2(y, x)

    # Compute pitch to angle paddle based on ball's vertical velocity
    speed_horiz = np.sqrt(ball_velocity[0]**2 + ball_velocity[1]**2)
    if speed_horiz > 0.01:
        approach_pitch = np.arctan2(-ball_velocity[2], speed_horiz)
    else:
        approach_pitch = 0.0

    # Clamp pitch to reasonable range
    approach_pitch = np.clip(approach_pitch, -0.75, 0.75)

    # Roll is typically 0 for a flat paddle
    roll = 0.0

    return np.array([x, y, z, roll, approach_pitch, approach_yaw])


def compute_yaw_from_target(target_point):
    """Compute joint 0 (yaw) angle to point toward the target XY position.

    Args:
        target_point: np.array([x, y, z]) in robot frame

    Returns:
        float: yaw angle in radians for joint 0
    """
    return np.arctan2(target_point[1], target_point[0])


def estimate_time_to_intercept(trajectory, intercept_index, dt):
    """Estimate time until ball reaches the intercept point.

    Args:
        trajectory: predicted trajectory points
        intercept_index: index of intercept point in trajectory
        dt: time step between trajectory points

    Returns:
        float: estimated time in seconds
    """
    return intercept_index * dt
