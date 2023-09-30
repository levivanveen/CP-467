import os
import cv2

import interpolation
import point_operation
import histogram

CWD = os.getcwd()
# Input image paths
CAMERAMAN = os.path.abspath(os.path.join(CWD, '..', 'input-images', 'cameraman.tif'))
EINSTEIN = os.path.abspath(os.path.join(CWD, '..', 'input-images', 'einstein.tif'))
CHEST_XRAY1 = os.path.abspath(os.path.join(CWD, '..', 'input-images', 'chest_x-ray1.jpeg'))
CHEST_XRAY2 = os.path.abspath(os.path.join(CWD, '..', 'input-images', 'chest_x-ray2.jpeg'))

# Output image paths (adjust the paths as needed)
CAMERAMAN_RESCALED = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_rescaled.tif'))
CAMERAMAN_NEAREST = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_nearest.tif'))
CAMERAMAN_BILINEAR = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_bilinear.tif'))
CAMERAMAN_BICUBIC = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_bicubic.tif'))
CAMERAMAN_NEGATIVE = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_negative.tif'))
CAMERAMAN_POWER = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_power.tif'))
CAMERAMAN_CONTRAST = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'cameraman_contrast.tif'))
EINSTEIN_EQUALIZED = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'einstein_equalized.tif'))
CHEST_XRAY3 = os.path.abspath(os.path.join(CWD, '..', 'output-images', 'chest_x-ray3.jpeg'))

# Operation values
GAMMA = 1.4
LEVEL_RANGE = 256

def main():
  function_type = input("""What function would you like to run?
  (1) Image Interpolation
  (2) Point Operations
  (3) Histrogram Processing
  Enter to quit
  """)
  if function_type == "1":
    image_interpolation()
  elif function_type == "2":
    point_operations()
  elif function_type == "3":
    histogram_processing()
  elif function_type == "":
    return
  else:
    print("Invalid input")
  main()

def image_interpolation():
  question = input("""Which question would you like to run?
  1. Rescale by 1/4
  2. Rescale by 4x
  """)
  if question == "1":
    img = cv2.imread(CAMERAMAN, cv2.IMREAD_UNCHANGED)
    if img is not None:
      img_rescaled = interpolation.shrink_one_quarter(img)
      cv2.imwrite(CAMERAMAN_RESCALED, img_rescaled)
      print("Rescaled image saved to {}".format(CAMERAMAN_RESCALED))
    else:
      print("Invalid path for cameraman")
      
  elif question == "2":
    img = cv2.imread(CAMERAMAN_RESCALED, cv2.IMREAD_UNCHANGED)
    if img is None:
      print("Please run question 1 first")
      return

    letter_question = input("""Which method would you like to use?
    a. Nearest Neighbor
    b. Bilinear Interpolation
    c. Bicubic Interpolation
    """)
    if letter_question == "a":
      img_nearest = interpolation.nearest_neighbor(img)
      cv2.imwrite(CAMERAMAN_NEAREST, img_nearest)
      print("Rescaled image saved to {}".format(CAMERAMAN_NEAREST))
    elif letter_question == "b":
      img_bilinear = interpolation.bilinear(img)
      cv2.imwrite(CAMERAMAN_BILINEAR, img_bilinear)
      print("Rescaled image saved to {}".format(CAMERAMAN_BILINEAR))
    elif letter_question == "c":
      img_bicubic = interpolation.bicubic(img)
      cv2.imwrite(CAMERAMAN_BICUBIC, img_bicubic)
      print("Rescaled image saved to {}".format(CAMERAMAN_BICUBIC))
    else: 
      print("Invalid input")
  else:
    print("Invalid input")
  return

def point_operations():
  question = input("""Which question would you like to run?
  (1) Negative
  (2) Power Law Transform
  (3) Contrast Stretching
  """)
  if question == "1":
    img = cv2.imread(CAMERAMAN, cv2.IMREAD_UNCHANGED)
    if img is not None:
      img_negative = point_operation.negative(img, LEVEL_RANGE)
      cv2.imwrite(CAMERAMAN_NEGATIVE, img_negative)
      print("Negative image saved to {}".format(CAMERAMAN_NEGATIVE))
    else:
      print("Invalid path for cameraman")
  elif question == "2":
    img = cv2.imread(CAMERAMAN, cv2.IMREAD_UNCHANGED)
    if img is not None:
      img_power = point_operation.power_law(img, GAMMA)
      cv2.imwrite(CAMERAMAN_POWER, img_power)
      print("Power law image saved to {}".format(CAMERAMAN_POWER))
    else:
      print("Invalid path for cameraman")
  elif question == "3":
    img = cv2.imread(CAMERAMAN, cv2.IMREAD_UNCHANGED)
    if img is not None:
      img_contrast = point_operation.contrast_stretch(img, 0, LEVEL_RANGE - 1)
      cv2.imwrite(CAMERAMAN_CONTRAST, img_contrast)
      print("Contrast image saved to {}".format(CAMERAMAN_CONTRAST))
    else:
      print("Invalid path for cameraman")
  else:
    print("Invalid input")
  return

def histogram_processing():
  question = input("""Which question would you like to run?
  (1) Histogram Equalization
  (2) Histogram Specification
  """)
  if question == "1":
    img = cv2.imread(EINSTEIN, cv2.IMREAD_UNCHANGED)
    if img is not None:
      img_equalized = histogram.equalize(img, LEVEL_RANGE)
      cv2.imwrite(EINSTEIN_EQUALIZED, img_equalized)
      print("Equalized image saved to {}".format(EINSTEIN_EQUALIZED))
    else:
      print("Invalid path for einstein")
  elif question == "2":
    img1 = cv2.imread(CHEST_XRAY1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(CHEST_XRAY2, cv2.IMREAD_UNCHANGED)
    if img1 is not None and img2 is not None:
      img_specified = histogram.specify(img1, img2, LEVEL_RANGE)
      cv2.imwrite(CHEST_XRAY3, img_specified)
      print("Specified image saved to {}".format(CHEST_XRAY3))
    else:
      print("Invalid path for chest_x-ray1 or chest_x-ray2")
  else:
    print("Invalid input")
  return

if __name__ == "__main__":
  main()
