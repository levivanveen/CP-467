import cv2
import numpy as np

def shrink_one_quarter(img):
  height, width = img.shape[:2] # img.shape returns (height, width, channel)
  new_height = height // 2
  new_width = width // 2

  img_rescaled = np.zeros((new_height, new_width), dtype=np.uint8)

  for i in range(new_height):
    for j in range(new_width):
      img_rescaled[i][j] = img[i*2][j*2]

  return img_rescaled

def nearest_neighbor(img):
  height, width = img.shape[:2]
  new_height = height * 2
  new_width = width * 2

  img_nearest = np.zeros((new_height, new_width), dtype=np.uint8)

  for i in range(height):
    for j in range(width):
      img_nearest[i*2 + 1][j*2 + 1] = img[i][j]
      img_nearest[i*2][j*2] = img[i][j]
      
  return img_nearest

def bilinear(img):
  height, width = img.shape[:2]
  new_height = height * 2
  new_width = width * 2

  img_bilinear = np.zeros((new_height, new_width), dtype=np.uint8)

  for i in range(new_height):
    for j in range(new_width):
      img_bilinear[i][j] = _bilinear_interpolation(img, j, i, height, width)

  return img_bilinear

def bicubic(img):
  height, width = img.shape[:2]
  new_height = height * 2
  new_width = width * 2
  
  return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def _bilinear_interpolation(img, x, y, height, width):
  # given position (x, y) in the image, return the new pixel value using bilinear interpolation
  x_og = x // 2
  y_og = y // 2

  x0 = int(x_og)
  y0 = int(y_og)
  
  x1 = x0 + 1 if x0 < width - 1 else x0
  y1 = y0 + 1 if y0 < height - 1 else y0

  alpha = x_og - x0
  beta = y_og - y0

  f00 = img[y0][x0]
  f01 = img[y0][x1]
  f10 = img[y1][x0]
  f11 = img[y1][x1]

  return (1 - alpha) * (1 - beta) * f00 + (1 - alpha) * beta * f01 + alpha * (1 - beta) * f10 + alpha * beta * f11