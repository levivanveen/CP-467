import numpy as np

def negative(img, level_range):
  height, width = img.shape[:2]
  img_negative = np.zeros((height, width), dtype=np.uint8)

  for i in range(height):
    for j in range(width):
      img_negative[i][j] = level_range - 1 - img[i][j]
  return img_negative

def power_law(img, gamma):
  height, width = img.shape[:2]
  img_power = np.zeros((height, width), dtype=np.uint8)

  for i in range(height):
    for j in range(width):
      img_power[i][j] = (img[i][j]) ** gamma
      if (img_power[i][j] > 255):
        img_power[i][j] = 255
      elif (img_power[i][j] < 0):
        img_power[i][j] = 0
  
  return img_power

def contrast_stretch(img, s1, s2):
  height, width = img.shape[:2]

  # Set r1 and r2 to the min and max pixel values in the image
  r1 = 256
  r2 = 0
  for i in range(height):
    for j in range(width):
      if img[i][j] < r1:
        r1 = img[i][j]
      if img[i][j] > r2:
        r2 = img[i][j]

  img_contrast = np.zeros((height, width), dtype=np.uint8)

  for i in range(height):
    for j in range(width):
      # new pixel = (old pixel - min pixel) * (new max - new min) / (max pixel - min pixel) + new min
      img_contrast[i][j] = (img[i][j] - r1) * ((s2 - s1) / (r2 - r1)) + s1
      if (img_contrast[i][j] > 255):
        img_contrast[i][j] = 255
      elif (img_contrast[i][j] < 0):
        img_contrast[i][j] = 0

  return img_contrast