import cv2
import numpy as np

# 1. Initialize Webcam (CAP_DSHOW prevents camera light freeze on Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 2. Define HSV Color Range for a BLUE object (Default)
# Tip: Adjust lower/upper bounds if using a Red/Green object!
LOWER_COLOR = np.array([100, 150, 0])  # Lower bound for Blue in HSV
UPPER_COLOR = np.array([140, 255, 255])  # Upper bound for Blue in HSV

canvas = None
prev_point = None

print("Air Canvas Started! Wave a BLUE marker in front of the camera to draw.")
print("Press 'c' to clear canvas. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  # Flip frame horizontally for intuitive "mirror" drawing
  frame = cv2.flip(frame, 1)

  # Initialize black canvas to match camera size on first frame
  if canvas is None:
    canvas = np.zeros_like(frame)

  # --- CORE CV CONCEPTS FOR STUDENTS ---

  # A. Convert BGR to HSV color space (easier to isolate specific colors)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # B. Create binary mask (White = color detected, Black = background)
  mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

  # Clean noise using morphological erosion and dilation
  mask = cv2.erode(mask, None, iterations=1)
  mask = cv2.dilate(mask, None, iterations=1)

  # C. Find contours (outlines of detected shapes)
  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  curr_point = None

  if contours:
    # Get the largest contour (prevents small background noise from drawing)
    largest_contour = max(contours, key=cv2.contourArea)

    # Filter out tiny detections by minimum area threshold
    if cv2.contourArea(largest_contour) > 500:
      # Calculate centroid (center x, y coordinates) of detected marker
      M = cv2.moments(largest_contour)
      if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        curr_point = (cX, cY)

        # Draw a visual tracking dot around object tip
        cv2.circle(frame, curr_point, 8, (0, 255, 255), -1)

  # D. Draw lines on the canvas when object moves
  if curr_point and prev_point:
    cv2.line(canvas, prev_point, curr_point, (0, 0, 255), 5)  # Red ink line

  prev_point = curr_point

  # Combine live webcam feed and drawing canvas together
  combined = cv2.add(frame, canvas)

  # Display UI instructions
  cv2.putText(
      combined,
      "Press 'c' to Clear | Press 'q' to Quit",
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (255, 255, 255),
      2,
  )

  cv2.imshow('BCA AI Demo - Virtual Air Canvas', combined)

  key = cv2.waitKey(1) & 0xFF
  if key == ord('c'):
    canvas = np.zeros_like(frame)  # Clear canvas
  elif key == ord('q'):
    break

# Clean up webcam resources cleanly
cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)