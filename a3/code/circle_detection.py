import cv2
import numpy as np
import math

# constants
G_KERNEL_SIZE = 7
THRESHOLD_LOW = 35
THRESHOLD_HIGH = 210

OUTER_CIRCLE_COLOR = (0, 255, 0)  # Green color for the outer circle
CENTER_COLOR = (0, 0, 255)  # Red color for the center point
OUTER_CIRCLE_THICKNESS = 2
CENTER_THICKNESS = 3
CENTER_RADIUS = 2

def eye_detection(img):
  # Blur the image to reduce noise
  eye_blurred = cv2.GaussianBlur(img, (G_KERNEL_SIZE, G_KERNEL_SIZE), 1.5)
  pupil_eye = pupil_detection(eye_blurred)

  if pupil_eye is None or len(pupil_eye[0]) == 0:
    print("No pupil detected")
    return None

  x, y, rad = pupil_eye[0][0]
  x_out = int(x + 1.1 * rad)
  y_out = int(y)
  threshold_val = img[y_out, x_out]
  _, thresholded_eye = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)


  more_circles = iris_detection(thresholded_eye, pupil_eye[0][0])
  if more_circles is None:
    print("No iris detected")
    return draw_circles(pupil_eye, img)
  return draw_circles(more_circles, draw_circles(pupil_eye, img))

def iris_detection(img, pupil, thresh_low=THRESHOLD_LOW, thresh_high=THRESHOLD_HIGH):
  # Need pupil values when checking if iris circle is valid
  p_x, p_y, p_rad = pupil

  while True:
    print(f"thresh_low: {thresh_low}, thresh_high: {thresh_high}")
    circles = cv2.HoughCircles(
                        img, 
                        cv2.HOUGH_GRADIENT, 
                        dp=1, # Smaller value = more accurate but slower
                        minDist=40,  # Minimum distance between circles
                        param1=thresh_high,  # Upper threshold for Canny edge detector
                        param2=thresh_low, # Threshold for center detection
                        minRadius=20, # Minimum radius of circle to be detected 
                        maxRadius=0 # Maximum radius of circle to be detected
                      )
    valid_circles = [[]]

    if circles is not None and len(circles[0]) > 0:
      for circle in circles[0]:
        x, y, rad = circle
        # Check if the circle is valid
        valid = check_circle_overlap(p_x, p_y, p_rad, x, y, rad)

        if valid:
          valid_circles[0].append(circle)
      
    valid_circles = np.array(valid_circles)

    # Preview the valid circles
    img_copy = img.copy()
    increment = _preview_eye(img_copy, valid_circles)
    if increment is None:
      break
    thresh_low = max(thresh_low + increment, 0)
  return valid_circles

# Show the image with the detected circle, and wait for the user to press a key
def _preview_eye(img, circle):
  drawn_img = draw_circles(circle, img)
  if drawn_img is None:
    cv2.imshow("Eye", img)  
  else:
    cv2.imshow("Eye", drawn_img)
  key = cv2.waitKey(0)

  cv2.destroyAllWindows()
  if key == 0:
    return 1
  elif key == 1:
    return -1
  return None

def pupil_detection(eye):
  i, temp_low = 0, THRESHOLD_LOW

  while i < 20:
    i += 1
    pupil_circle = cv2.HoughCircles(
                      eye, 
                      cv2.HOUGH_GRADIENT, 
                      dp=0.1, # Smaller value = more accurate but slower
                      minDist=40,  # Minimum distance between circles
                      param1=THRESHOLD_HIGH,  # Upper threshold for Canny edge detector
                      param2=temp_low, # Threshold for center detection
                      minRadius=20, # Minimum radius of circle to be detected 
                      maxRadius=0 # Maximum radius of circle to be detected
                    )
    if pupil_circle is None or len(pupil_circle[0]) == 0:
      temp_low = max(temp_low - 2, 0)
    elif len(pupil_circle[0]) > 1:
      temp_low += 2
    else:
      break
  if len(pupil_circle[0]) == 0 or len(pupil_circle[0]) > 1:
    return None
  return pupil_circle

def draw_circles(circles, img):
  if circles is None:
    return None
  circles = np.uint16(np.around(circles))
  for circle in circles[0,:]:
    center_x, center_y, radius = circle[0], circle[1], circle[2]
    # Draw the outer circle
    cv2.circle(img, (center_x, center_y), radius, OUTER_CIRCLE_COLOR, OUTER_CIRCLE_THICKNESS)
    # Draw the center of the circle
    cv2.circle(img, (center_x, center_y), CENTER_RADIUS, CENTER_COLOR, CENTER_THICKNESS)
  return img

def check_circle_overlap(x1, y1, r1, x2, y2, r2):
  # Distance between centers of circles
  distance_centers = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  
  if distance_centers > r1 + r2:
    # Radius don't overlap
    return False 
  elif distance_centers <= abs(r1 - r2):
    if r1 == r2 and distance_centers == 0:
      return False # Circles are coincident
    else:
      return True # One circle is inside the other
  else:
    return False # Circles overlap