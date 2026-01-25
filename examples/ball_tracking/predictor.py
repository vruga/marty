"""Kalman filter state estimator + physics-based trajectory prediction (gravity + drag)."""

import numpy as np
import time
from config import (
    KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE,
    GRAVITY, AIR_DRAG_COEFF, PREDICTION_DT, PREDICTION_HORIZON
)


class BallPredictor:
    """Kalman filter for ball state estimation and physics-based trajectory prediction."""

    def __init__(self):
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        # State covariance
        self.P = np.eye(6) * 1.0
        # Measurement noise
        self.R = np.eye(3) * KALMAN_MEASUREMENT_NOISE
        # Process noise (tuned)
        self.Q_base = np.eye(6) * KALMAN_PROCESS_NOISE

        self.last_time = None
        self.initialized = False

    def reset(self):
        """Reset the filter state."""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.last_time = None
        self.initialized = False

    def update(self, position_3d):
        """Update Kalman filter with a new 3D position measurement.

        Args:
            position_3d: np.array([x, y, z]) in robot frame (meters)

        Returns:
            np.array: Filtered state [x, y, z, vx, vy, vz]
        """
        now = time.time()

        if not self.initialized:
            self.state[:3] = position_3d
            self.state[3:] = 0.0
            self.last_time = now
            self.initialized = True
            return self.state.copy()

        dt = now - self.last_time
        if dt <= 0:
            dt = 1.0 / 60.0
        self.last_time = now

        # --- Prediction step ---
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Predicted state (add gravity to vz)
        self.state = F @ self.state
        self.state[5] += GRAVITY * dt  # gravity on z-velocity

        # Process noise scaled by dt
        Q = self.Q_base.copy()
        Q[:3, :3] *= dt
        Q[3:, 3:] *= dt

        self.P = F @ self.P @ F.T + Q

        # --- Measurement update ---
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z = position_3d
        y = z - H @ self.state  # innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        return self.state.copy()

    def get_velocity(self):
        """Get current estimated velocity."""
        return self.state[3:6].copy()

    def get_position(self):
        """Get current estimated position."""
        return self.state[0:3].copy()

    def predict_trajectory(self, max_time=None):
        """Predict future ball trajectory using physics (gravity + air drag).

        Args:
            max_time: Maximum prediction horizon in seconds (default from config)

        Returns:
            list of np.array: Predicted positions [[x,y,z], ...] at PREDICTION_DT intervals
        """
        if not self.initialized:
            return []

        if max_time is None:
            max_time = PREDICTION_HORIZON

        pos = self.state[:3].copy()
        vel = self.state[3:6].copy()

        trajectory = [pos.copy()]
        t = 0.0

        while t < max_time:
            # Air drag (proportional to velocity, opposing motion)
            drag = -AIR_DRAG_COEFF * vel

            # Acceleration: gravity + drag
            accel = np.array([drag[0], drag[1], GRAVITY + drag[2]])

            # Euler integration
            vel = vel + accel * PREDICTION_DT
            pos = pos + vel * PREDICTION_DT

            # Stop if ball hits ground (z <= 0)
            if pos[2] <= 0:
                pos[2] = 0
                trajectory.append(pos.copy())
                break

            trajectory.append(pos.copy())
            t += PREDICTION_DT

        return trajectory
