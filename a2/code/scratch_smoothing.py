import numpy as np
import cv2
import math

AVERAGE_FILTER_SIZE = 3
GAUSSIAN_FILTER_SIZE = 7
SOBEL_FILTER_SIZE = 3
SIGMA = 1
MEAN = 0

HORIZONTAL_SOBEL_KERNEL = np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
])

VERTICAL_SOBEL_KERNEL = np.array([
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
])

# 3x3 smoothing average filter
def averaging_filter(img):
  filtered_img = np.zeros_like(img)
  # Define filter as a 3x3 matrix of 1/9
  kernel = np.ones((AVERAGE_FILTER_SIZE, AVERAGE_FILTER_SIZE), np.float32) / 9

  # Apply filter to image
  for i in range(len(img)):
    for j in range(len(img[0])):
      # I and J passed in are the center of the kernel, so we need to offset by 1
      offset = AVERAGE_FILTER_SIZE // 2
      filtered_img[i][j] = _apply_kernel(img, i - offset, j - offset, kernel)
  return filtered_img

# 7x7 smoothing gaussian filter
def gaussian_filter(img):
  filtered_img = np.zeros_like(img)
  # Calculate kernel values
  kernel = _calculate_gaussian_kernel()

  # Apply filter to image
  for i in range(len(img)):
    for j in range(len(img[0])):
      # I and J passed in are the center of the kernel, so we need to offset by 3
      offset = GAUSSIAN_FILTER_SIZE // 2
      filtered_img[i][j] = _apply_kernel(img, i - offset, j - offset, kernel)
  return filtered_img

def sobel_filter(img):
  filtered_img = np.zeros_like(img) 
  for i in range(len(img)):
    for j in range(len(img[0])):
      # I and J passed in are the center of the kernel, so we need to offset by 1
      offset = SOBEL_FILTER_SIZE // 2
      horizontal = _apply_kernel(img, i - offset, j - offset, HORIZONTAL_SOBEL_KERNEL)
      vertical = _apply_kernel(img, i - offset, j - offset, VERTICAL_SOBEL_KERNEL)
      filtered_img[i][j] = math.sqrt(horizontal ** 2 + vertical ** 2)
  return filtered_img

""" 
  Helper function to apply the kernel to the image
  kernel must be a square matrix
"""
def _apply_kernel(img, row, col, kernel):
  offset = len(kernel) // 2
  new_pixel = 0
  top_idx = offset - 1
  bottom_idx = len(img) - offset
  left_idx = offset - 1
  right_idx = len(img[0]) - offset

  for i in range(len(kernel)):
    for j in range(len(kernel)):
      if i + row < top_idx or i + row > bottom_idx or j + col < left_idx or j + col > right_idx:
        new_pixel += 0
      else:
        new_pixel += img[row + i][col + j] * kernel[i][j]
  return int(new_pixel)

"""
  Helper function to calculate the gaussian kernel
  kernel must be a square matrix
"""
def _calculate_gaussian_kernel():
  kernel = np.zeros((GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIZE), np.float32)
  for i in range(GAUSSIAN_FILTER_SIZE):
    for j in range(GAUSSIAN_FILTER_SIZE):
      x = i - (GAUSSIAN_FILTER_SIZE // 2)
      y = j - (GAUSSIAN_FILTER_SIZE // 2)
      kernel[i][j] = math.exp(-(x**2 + y**2) / (2 * SIGMA**2))
  
  # Normalize kernel
  kernel = kernel / kernel.sum()
  return kernel
