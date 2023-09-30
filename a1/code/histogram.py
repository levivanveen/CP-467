import cv2
import numpy as np

def equalize(img, level_range):
  height, width = img.shape[:2]
  img_equalized = np.zeros((height, width), dtype=np.uint8)

  # Calculate the cumulative probability of each intensity level
  pixel_cumulative_arr = cumulative_histogram(img, level_range)

  # Calculate the new intensity level for each intensity level
  pixel_new_intensity_arr = np.round(pixel_cumulative_arr * (level_range - 1)).astype(np.uint8)

  # Assign the new intensity level to each pixel
  img_equalized = pixel_new_intensity_arr[img]

  return img_equalized

def specify(img1, img2, level_range):
  height, width = img1.shape[:2]
  img_specified = np.zeros((height, width), dtype=np.uint8)
  mapping = [0] * level_range

  # Calculate the cumulative probability of each intensity level
  img1_cumulative_prob = cumulative_histogram(img1, level_range)
  img2_cumulative_prob = cumulative_histogram(img2, level_range)

  img1_intensity_arr = np.round(img1_cumulative_prob * (level_range - 1)).astype(np.uint8)
  img2_intensity_arr = np.round(img2_cumulative_prob * (level_range - 1)).astype(np.uint8)

  for i in range(len(img1_cumulative_prob)):
    # find the index of the closest value in img2_cumulative_prob to img1_cumulative_prob[i]
    mapping[i] = find_value_index(img2_intensity_arr, img1_intensity_arr[i])
  
  # Assign the new intensity level to each pixel
  for i in range(height):
    for j in range(width):
      img_specified[i][j] = mapping[img1[i][j]]

  return img_specified

def cumulative_histogram(img, level_range):
  pixel_count_hist = np.histogram(img, bins=np.arange(level_range+1))[0]

  # Calculate the probability of each intensity level
  pixel_prob_arr = pixel_count_hist / (img.shape[0] * img.shape[1])

  # Calculate the cumulative probability of each intensity level
  pixel_cumulative_arr = np.cumsum(pixel_prob_arr)

  return pixel_cumulative_arr

def find_value_index(arr, target):
  min_diff = float('inf')
  nearest_index = None

  for i, value in enumerate(arr):
    diff = abs(int(value) - int(target))
    if diff < min_diff:
      min_diff = diff
      nearest_index = i
  
  return nearest_index

