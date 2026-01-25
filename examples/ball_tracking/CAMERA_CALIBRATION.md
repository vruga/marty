# Camera-to-Robot Calibration: How It Works

This doc explains **what** the calibration does, **how** it’s implemented, and **how camera vs robot axes** relate so we can fix bad transforms.

---

## 1. What Gets Calibrated (Intrinsics vs Extrinsics)

### Intrinsics (NOT calibrated by this code)

**Intrinsics** describe how 3D points in the **camera’s optical frame** are projected to 2D pixels: focal lengths `fx`, `fy`, principal point `cx`, `cy`, distortion.

- This pipeline **does not** calibrate intrinsics.
- It uses the **RealSense’s built‑in intrinsics** from the **color stream**:
  - `color_stream.as_video_stream_profile().get_intrinsics()`
- Those are used in `rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)` to turn `(pixel_x, pixel_y, depth_m)` into `[X, Y, Z]` in the **camera frame**.

So: **intrinsics come from the RealSense SDK; we only use them for deprojection.**

### Extrinsics (what we calibrate)

**Extrinsics** are the rigid transform from the **camera frame** to the **robot base frame**: a 3×3 rotation `R` and 3×1 translation `t`:

- `p_robot = R @ p_camera + t`
- Stored as a 4×4 matrix `T = [R|t; 0 0 0 1]` in `camera_to_robot.json`.

We compute `R` and `t` from **N corresponding 3D point pairs**:
- `camera_points[i]`: 3D from “pixel + depth” and `rs2_deproject_pixel_to_point` (camera frame).
- `robot_points[i]`: 3D from **forward kinematics (FK)** at the same pose (robot frame).

---

## 2. How 3D Points Are Obtained in Code

### Camera 3D (camera frame)

1. **Pixel + depth**
   - Detect orange marker (or ball) in the **color** image → `(cx, cy)`.
   - Read depth at `(cx, cy)` from the **depth stream aligned to color**.
   - `depth_m = median(depth_patch) / 1000.0` (depth in mm → meters).

2. **Deproject**
   - `camera.deproject_pixel(cx, cy, depth_m)` calls:
     - `rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)`
   - Returns `[X, Y, Z]` in **meters** in the **camera optical frame**.

### Robot 3D (robot frame)

1. **Joint angles**  
   - At each calibration pose: `qpos = [q0, q1, q2, q3, q4, q5]` (e.g. 6 joints).

2. **FK + base yaw**
   - `lerobot_FK(qpos[1:5], robot)` uses joints 1–4 only (arm, no gripper/yaw).
   - FK gives `[X, Y, Z, gamma, beta, alpha]` in the **arm frame** (before base yaw).
   - Base yaw `q0` is applied to the XY from FK:
     - `x_rot = x*cos(q0) - y*sin(q0)`, `y_rot = x*sin(q0) + y*cos(q0)`, `z` unchanged.
   - `robot_points[i] = [x_rot, y_rot, z]` in the **robot base frame**.

---

## 3. Camera vs Robot Axis Conventions

### RealSense (camera) frame

From Intel’s docs for `rs2_deproject_pixel_to_point`:

| Axis | Direction      | In the scene        |
|------|----------------|---------------------|
| **X** | Right         | To the right in image |
| **Y** | Down          | Down in image       |
| **Z** | Forward       | Into the scene (depth) |

- Origin: optical center of the **color** camera.
- Units: **meters**.

### Robot (FK) frame

From the SO-100/SO-101 kinematics (ET chain: `tx`, `tz`, `Ry`, …) and how it’s used in controller/planner:

| Axis | Direction   | Typical meaning      |
|------|-------------|----------------------|
| **X** | Forward    | In front of the base (at yaw=0) |
| **Y** | Left       | Left of base (right‑hand rule)  |
| **Z** | Up         | Height above base    |

- `arctan2(y, x)` is used as yaw; `r = sqrt(x² + y²)` is radial distance.
- Workspace: `WORKSPACE_R_MIN/MAX` in the XY plane, `WORKSPACE_Z_MIN/MAX` in Z.

So:

- **Camera**: X right, Y down, Z forward.
- **Robot**: X forward, Y left, Z up.

If the camera looks **down** at the table, camera Z points **down** in the world, which is roughly **−Z_robot**. The calibration (Kabsch) will find the correct `R` and `t` as long as:
- Correspondences are correct,
- Poses are sufficiently varied,
- There is no systematic bias (e.g. marker not at the EE).

---

## 4. How the Transform Is Computed (Kabsch / SVD)

We want:

`p_robot = R @ p_camera + t`

with `R` orthogonal, `det(R) = 1`.

### Steps (`calibration.compute_rigid_transform`)

1. **Centroids**
   - `centroid_cam = mean(camera_points)`
   - `centroid_rob = mean(robot_points)`

2. **Center the clouds**
   - `cam_centered = camera_points - centroid_cam`
   - `rob_centered = robot_points - centroid_rob`

3. **Cross‑covariance**
   - `H = cam_centered.T @ rob_centered`
   - (`H = P'^T Q'` with `P'`= camera, `Q'`= robot)

4. **SVD**
   - `U, S, Vt = np.linalg.svd(H)` ⇒ `H = U @ diag(S) @ Vt`

5. **Rotation**
   - `R = Vt.T @ diag(1, 1, sign(det(Vt.T @ U.T))) @ U.T`
   - `Vt.T @ U.T` is the Kabsch solution; the `diag(1,1,±1)` forces `det(R)=+1` (no reflection).

6. **Translation**
   - `t = centroid_rob - R @ centroid_cam`

7. **4×4**
   - `T[:3,:3] = R`, `T[:3,3] = t`, `T[3,:] = [0,0,0,1]`.

### How it’s used

- **At runtime** (e.g. in `main.py`):  
  `pos_robot = transform_point(T, pos_camera)`  
  i.e. `pos_robot = (T @ [x,y,z,1])[:3] = R @ pos_camera + t`.

So `T` is **camera → robot**.

---

## 5. Why the Transform Can Look “Wrong”

### 1) **Marker not at end‑effector**

- FK gives the **EE origin**.
- Detection gives the **orange marker**.
- If the marker is offset (e.g. 2 cm in front of the gripper), every pair `(camera_point, robot_point)` is biased. Kabsch will fit a transform that “works” for those biased pairs but will be **wrong for the ball** (or any other point not at the EE).

**Fix:** Put the marker as close as possible to the EE, or model a **marker offset in the EE frame** and subtract it from `robot_points` before Kabsch.

### 2) **Bad or degenerate poses**

- If many poses are **coplanar** or almost so, `H` is ill‑conditioned and `R` can be unreliable (e.g. strange flips).
- If poses are too similar, the problem is under‑constrained.

**Fix:** Use 6–8 **diverse** poses: different X,Y,Z, different yaws and heights.

### 3) **Wrong robot model**

- `ROBOT_TYPE` (e.g. `"so100"` vs `"so101"`) changes FK.  
- If the real robot is SO-101 but config uses `so100`, `robot_points` are wrong and the transform will be wrong.

**Fix:** Set `ROBOT_TYPE` in `config.py` to match the real robot.

### 4) **Depth / intrinsics**

- Depth in **mm** must be converted to **meters** before `rs2_deproject_pixel_to_point`.
- We must use **color** intrinsics when deprojecting from the **color** image (and depth aligned to color).  
- The code does both; if you change the pipeline (e.g. different alignment), this can break.

### 5) **Gripper / joint 5**

- `get_fk_position` uses `qpos[1:5]` (joints 1–4). Joint 5 (gripper) does not move the EE position in the current FK, so this is consistent.

---

## 6. Quick Sanity Checks

After calibration you can:

1. **det(R) = 1**  
   - `np.linalg.det(T[:3,:3])` should be ≈ 1.

2. **Residuals**  
   - For each pair:  
     `err_i = ||(R @ p_cam_i + t) - p_rob_i||`  
   - Mean and max `err_i` (in mm) are already printed; they should be small (e.g. &lt; 10–20 mm) for a good calibration.

3. **Axis directions**  
   - For a point **above** the table, `Z_robot` should be **positive**.  
   - For a point in **front** of the base, `X_robot` should be **positive** (with the usual X-forward convention).  
   - If these are flipped, the rotation or frame convention is suspect.

4. **Reprojection**  
   - For a few `robot_points`, apply `T_inv` to get camera, then `rs2_project_point_to_pixel` and compare to the detected marker pixels. They should match.

---

## 7. File Roles

| File               | Role |
|--------------------|------|
| `camera.py`        | RealSense streams, **color intrinsics**, `deproject_pixel` (camera 3D). |
| `detector.py`      | Orange detection, depth sampling, calls `camera.deproject_pixel`. |
| `calibration.py`   | `compute_rigid_transform` (Kabsch), `save_transform`, `load_transform`, `transform_point`. |
| `calibrate_camera.py` | Moves robot to poses, collects `(camera_points, robot_points)`, runs `compute_rigid_transform`, saves `camera_to_robot.json`. |
| `camera_to_robot.json` | The 4×4 `T` (camera → robot). |
| `config.py`       | `ROBOT_TYPE`, `CALIBRATION_FILE_PATH`, `NUM_CALIBRATION_POINTS`, etc. |

---

## 8. Summary

- **Intrinsics:** from RealSense color stream; used only for deprojection; **not** calibrated here.
- **Extrinsics:** rigid `T` (camera → robot) from N corresponding 3D pairs via **Kabsch (SVD)**.
- **Camera frame:** X right, Y down, Z forward (RealSense).
- **Robot frame:** X forward, Y left, Z up (from FK + base yaw).
- **Transforms go wrong** mainly due to: marker offset, bad/planar poses, wrong `ROBOT_TYPE`, or depth/intrinsics misuse.

The next step is to re‑run calibration with diverse poses and the correct `ROBOT_TYPE`.

---

## 9. Debugging a bad transform

If `det(R)` is far from 1 when loading: the stored `T` is not a valid rotation (reflection or corruption). Re-run `calibrate_camera.py` and check that you get `det(R)=1` and low residuals.

If residuals are large (>20 mm) or one axis (X/Y/Z) dominates: likely marker not at EE, wrong `ROBOT_TYPE`, or depth/intrinsics (e.g. depth in wrong units or wrong intrinsics stream). Put marker at EE, set `ROBOT_TYPE` correctly, and ensure depth is in meters and color intrinsics are used for the aligned color image.

If the ball in robot frame has the wrong sign on Z (e.g. below the table but Z_robot &lt; 0): the camera or robot frame may be flipped. Check that both use the conventions in section 3; if the camera is mounted upside down or the robot’s “up” is different, you may need an extra fixed flip in code (prefer fixing the convention in one place).

---

## 10. When the Robot Is NOT in the Camera FOV

The **current calibration** (`calibrate_camera.py`) **requires the end-effector (with the orange marker) to be visible in the camera** at each calibration pose.

### What the code does

- At each pose, the detector looks for the **orange marker in the image**.
- **Camera 3D** comes from: detect marker → (pixel, depth) → `rs2_deproject_pixel_to_point`.
- If the marker is **outside the image**, `detector.detect` returns `pos_3d = None`.
- You can press **'c' to capture only when `pos_3d is not None`**. If the marker is not detected, **'c' does nothing**; you must press **'s' to skip**.
- Kabsch needs **at least 3 point pairs**. If you skip too many poses (robot/marker outside FOV), you get &lt; 3 points → **calibration aborts**.

So: **if the robot (or the EE+marker) is not in the camera FOV at the calibration poses, this method does not work** as-is.

### What you can do

| Situation | Option |
|-----------|--------|
| **Camera can't see the robot at the default poses** | **Adjust poses or camera:** Choose `CALIBRATION_POSITIONS` so the **EE + marker** are in the image at each pose, or move/rotate the camera so the workspace is in FOV. The **robot base** can be outside the image; only the **EE with the marker** must be visible. |
| **Robot must stay completely out of view** | Use **scene-based calibration** (pattern in the scene, robot touches it) or **eye-in-hand** (camera on the robot, pattern fixed in the world). See below. |

### Alternative 1: Pattern in the scene + robot touches (robot not in FOV)

- Put a **calibration pattern** (ChArUco or checkerboard) **in the scene** (e.g. on the table) so the **camera sees the pattern**.
- Use `solvePnP` / `cv2.aruco` to get the **pattern pose in the camera frame** → 3D positions of the pattern corners in camera frame.
- **Robot touches** specific corners. When "touching", the **EE position from FK** is the 3D point in **robot frame**. The **same physical point** on the pattern has 3D in the **camera frame**.
- That gives **(camera_point, robot_point)** pairs without the camera ever seeing the robot. Run the same Kabsch: `compute_rigid_transform(camera_points, robot_points)`.
- Requirements: pattern **fixed** during calibration; robot can **reach and touch** it; you need **pattern geometry** and to know **which corner** is being touched.

### Alternative 2: Eye-in-hand (camera on the robot)

- **Camera mounted on the robot**. Move the robot to many poses; at each pose the camera sees a **fixed pattern in the world**.
- From the image: **camera → pattern**. From robot FK: **robot → EE** (and **robot → camera** from mounting). Solve **camera → robot** via hand–eye (e.g. AX = XB). The **robot body is never in the camera FOV**; only the **pattern** is.

### Alternative 3: Poses that keep the EE in FOV

- **Simplest:** define `CALIBRATION_POSITIONS` so the **EE (with marker)** is **in the image** at each pose. The base can be out of view; only the **marker** must be detected.
- If the camera looks down at the table, choose poses that put the EE **above the table, in the central part of the image**. Adjust `CALIBRATION_POSITIONS` in `calibrate_camera.py`.
