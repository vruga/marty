"""Ball detector tuning tool - adjust HSV thresholds interactively.

Usage:
    python tune_detector.py

Controls:
    - Trackbars adjust HSV thresholds in real-time
    - 's' to save current thresholds to config
    - 'q' to quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from camera import RealSenseCamera

# Initial HSV values (orange ball)
hsv_low = [5, 100, 100]
hsv_high = [25, 255, 255]


def nothing(x):
    pass


def main():
    print("=== Ball Detector Tuning ===")
    print("Adjust trackbars to tune HSV thresholds")
    print("'s' = save, 'q' = quit")

    cam = RealSenseCamera()
    cam.start()
    print("Camera started.")

    cv2.namedWindow("Tuning")
    cv2.namedWindow("Mask")

    # Create trackbars
    cv2.createTrackbar("H Low", "Tuning", hsv_low[0], 179, nothing)
    cv2.createTrackbar("S Low", "Tuning", hsv_low[1], 255, nothing)
    cv2.createTrackbar("V Low", "Tuning", hsv_low[2], 255, nothing)
    cv2.createTrackbar("H High", "Tuning", hsv_high[0], 179, nothing)
    cv2.createTrackbar("S High", "Tuning", hsv_high[1], 255, nothing)
    cv2.createTrackbar("V High", "Tuning", hsv_high[2], 255, nothing)
    cv2.createTrackbar("Min Radius", "Tuning", 3, 100, nothing)
    cv2.createTrackbar("Max Radius", "Tuning", 100, 300, nothing)
    cv2.createTrackbar("Min Circ x10", "Tuning", 3, 10, nothing)  # 0.3 default

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    try:
        while True:
            color, depth = cam.get_frames()
            if color is None:
                continue

            # Get trackbar values
            h_low = cv2.getTrackbarPos("H Low", "Tuning")
            s_low = cv2.getTrackbarPos("S Low", "Tuning")
            v_low = cv2.getTrackbarPos("V Low", "Tuning")
            h_high = cv2.getTrackbarPos("H High", "Tuning")
            s_high = cv2.getTrackbarPos("S High", "Tuning")
            v_high = cv2.getTrackbarPos("V High", "Tuning")
            min_r = cv2.getTrackbarPos("Min Radius", "Tuning")
            max_r = cv2.getTrackbarPos("Max Radius", "Tuning")

            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])

            # Convert and threshold
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphological cleanup
            mask_clean = cv2.erode(mask, kernel_erode, iterations=1)
            mask_clean = cv2.dilate(mask_clean, kernel_dilate, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display = color.copy()
            best_detection = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < np.pi * min_r**2:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(contour)

                if radius < min_r or radius > max_r:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Draw all candidates in yellow
                cv2.circle(display, (int(cx), int(cy)), int(radius), (0, 255, 255), 1)
                cv2.putText(display, f"c={circularity:.2f}", (int(cx)+10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                if circularity > 0.7:
                    if best_detection is None or radius > best_detection[2]:
                        best_detection = (cx, cy, radius, circularity)

            # Draw best detection in green
            if best_detection is not None:
                cx, cy, radius, circ = best_detection
                cv2.circle(display, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.circle(display, (int(cx), int(cy)), 3, (0, 255, 0), -1)

                # Get depth
                depth_val = depth[int(cy), int(cx)]
                depth_m = depth_val / 1000.0 if depth_val > 0 else 0

                # Deproject
                if depth_m > 0:
                    point_3d = cam.deproject_pixel(cx, cy, depth_m)
                    cv2.putText(display, f"3D: [{point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(display, f"DETECTED r={radius:.0f} c={circ:.2f} d={depth_m:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "NO DETECTION", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show HSV values
            cv2.putText(display, f"HSV Low: [{h_low}, {s_low}, {v_low}]",
                        (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"HSV High: [{h_high}, {s_high}, {v_high}]",
                        (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Tuning", display)
            cv2.imshow("Mask", mask_clean)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n=== Save these values to config.py ===")
                print(f"HSV_LOWER = np.array([{h_low}, {s_low}, {v_low}])")
                print(f"HSV_UPPER = np.array([{h_high}, {s_high}, {v_high}])")
                print(f"MIN_BALL_RADIUS_PX = {min_r}")
                print(f"MAX_BALL_RADIUS_PX = {max_r}")
                print()

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
