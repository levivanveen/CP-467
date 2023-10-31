import cv2
import numpy as np

# constants
G_KERNEL_SIZE = 5
THRESHOLD_LOW = 39
THRESHOLD_HIGH = 200
OUTER_CIRCLE_COLOR = (0, 255, 0)  # Green color for the outer circle
CENTER_COLOR = (0, 0, 255)  # Red color for the center point
OUTER_CIRCLE_THICKNESS = 2
CENTER_THICKNESS = 3
CENTER_RADIUS = 2

def circle_detection(eye):
  #eye_blurred = cv2.GaussianBlur(eye, (G_KERNEL_SIZE, G_KERNEL_SIZE), 0)
  eye_circles = cv2.HoughCircles(
                    eye, 
                    cv2.HOUGH_GRADIENT, 
                    dp=0.1, # Smaller value = more accurate but slower
                    minDist=40,  # Minimum distance between circles
                    param1=THRESHOLD_HIGH,  # Upper threshold for Canny edge detector
                    param2=THRESHOLD_LOW, # Threshold for center detection
                    minRadius=20, # Minimum radius of circle to be detected 
                    maxRadius=0 # Maximum radius of circle to be detected
                  )
  if eye_circles is not None:
    # We want to draw all the circles detected onto the original image
    eye_circles = np.uint16(np.around(eye_circles))
    for circle in eye_circles[0,:]:
      center_x, center_y, radius = circle[0], circle[1], circle[2]
      # Draw the outer circle
      cv2.circle(eye, (center_x, center_y), radius, OUTER_CIRCLE_COLOR, OUTER_CIRCLE_THICKNESS)
      # Draw the center of the circle
      cv2.circle(eye, (center_x, center_y), CENTER_RADIUS, CENTER_COLOR, CENTER_THICKNESS)

  return eye