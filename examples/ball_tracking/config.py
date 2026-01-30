"""Configuration parameters for ball tracking and robot interception."""

import numpy as np

# --- Camera Settings ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# --- Ball Detection (HSV for orange ball) ---
HSV_LOWER = np.array([5, 203, 154])
HSV_UPPER = np.array([76, 255, 255])
MIN_BALL_RADIUS_PX = 8
MAX_BALL_RADIUS_PX = 100
MIN_CIRCULARITY = 0.7
DEPTH_SAMPLE_SIZE = 5  # 5x5 pixel region for depth averaging

# --- Kalman Filter ---
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 5e-3

# --- Trajectory Prediction ---
GRAVITY = -9.81  # m/s^2 (applied to Z axis)
AIR_DRAG_COEFF = 0.1  # optional drag coefficient
PREDICTION_DT = 0.005  # time step for trajectory integration (s)
PREDICTION_HORIZON = 2.0  # max prediction time (s)

# --- Robot Workspace (cylindrical bounds in robot frame) ---
WORKSPACE_R_MIN = 0.12  # meters
WORKSPACE_R_MAX = 0.32  # meters
WORKSPACE_Z_MIN = 0.05  # meters
WORKSPACE_Z_MAX = 0.22  # meters

# --- Robot Control ---
CONTROL_RATE_HZ = 50
ROBOT_PORT = "/dev/ttyACM0"
CALIBRATION_FILE = "examples/main_follower.json"
ROBOT_TYPE = "so100"  # must match your arm: "so100" or "so101" (affects FK in calibration)

# Joint limits (same as reference example)
CONTROL_QLIMIT = [[-2.1, -3.1, -0.0, -1.375, -1.57, -0.15],
                  [2.1, 0.0, 3.1, 1.475, 3.1, 1.5]]

# Ready/home position
READY_QPOS = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])

# Motion smoothing
MAX_JOINT_CHANGE = 0.05  # rad/step (conservative for safety)
GOAL_SPEED = 200  # motor speed register value for fast moves

# --- State Machine ---
DETECTION_FRAMES_THRESHOLD = 3  # frames of detection before entering TRACKING
BALL_LOST_TIMEOUT = 0.5  # seconds before returning to IDLE
RETURN_TOLERANCE = 0.05  # radians, tolerance for "at ready position"

# --- Calibration ---
CALIBRATION_FILE_PATH = "examples/ball_tracking/camera_to_robot.json"
NUM_CALIBRATION_POINTS = 8
# If the orange marker is not at the EE: offset from EE to marker in robot base frame [x,y,z] (m).
# For a marker on the gripper, prefer placing it at the EE; this is a coarse approximation.
MARKER_OFFSET_IN_ROBOT_FRAME = np.array([0.0, 0.0, 0.0])

# --- ArUco Calibration (for camera behind robot, not seeing EE) ---
ARUCO_MARKER_SIZE = 0.201  # meters (201mm = 20.1cm, measured from printed marker)
ARUCO_DICT_TYPE = 0  # cv2.aruco.DICT_4X4_50 = 0
MARKER_POSITIONS_FILE = "examples/ball_tracking/marker_positions.yaml"
ARUCO_CALIBRATION_FILE_PATH = "examples/ball_tracking/camera_to_world.json"

# --- MuJoCo Simulation ---
SIM_XML_PATH = "./examples/ball_tracking/scene_ball.xml"
SIM_BALL_INITIAL_POS = np.array([0.5, 0.0, 0.3])  # meters
SIM_BALL_VELOCITY = np.array([-1.5, 0.0, 0.5])  # m/s toward robot
SIM_BALL_RADIUS = 0.02  # meters (40mm table tennis ball)

# --- Visualization ---
VIS_WINDOW_NAME = "Ball Tracking Debug"
VIS_TRAJECTORY_COLOR = (0, 255, 0)  # green
VIS_DETECTION_COLOR = (0, 0, 255)  # red (BGR)
VIS_INTERCEPT_COLOR = (255, 0, 0)  # blue
