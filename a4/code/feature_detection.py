import cv2
import numpy as np

C_THRESH_LOW = 80
C_THRESH_HIGH = 255
MIN_LENGTH = 5
HOUGH_LINE_THRESH = 200

def feat_detection(img, circle_params):
  grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # First, detect edges
  edges = cv2.Canny(grey, C_THRESH_LOW, C_THRESH_HIGH)

  # Detect lines using hough transform
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_LINE_THRESH)

  blurred = cv2.GaussianBlur(grey, (5, 5), 1.5)
  circles_img = circles(blurred, img, circle_params)
  lines_img = draw_lines(line_filter(lines, MIN_LENGTH), circles_img)
  return lines_img

def draw_circles(circles, img):
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
      center = (circle[0], circle[1])
      radius = circle[2]
      cv2.circle(img, center, radius, (0, 255, 0), 2)

  return img

def draw_lines(lines, img):
  if lines is not None:
    for i in range(0, len(lines)):
      l = lines[i][0]
      cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3)
  return img

def circles(blurred, img, params):
  # Detect circles using hough transform
  circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    1, 
    40, 
    param1=params[0],
    param2=params[1],
    minRadius=params[2],
    maxRadius=params[3],
  )
  img_with_circles = np.copy(img)
  circles_img = draw_circles(circles, img_with_circles)
  return circles_img

def line_filter(lines, min_length=0):
  filtered_lines = []

  for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if length > min_length:
      filtered_lines.append(line)

  return filtered_lines

def _preview_img(img):
  assert img is not None
  cv2.imshow("Preview", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()