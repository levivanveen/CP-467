import os
import cv2

import scratch_smoothing
import opencv_smoothing
from edge_detection import marr_hildreth, canny, edge_map

CWD = os.getcwd()
# Input image paths
CAMERAMAN = os.path.abspath(os.path.join(CWD, '..', 'input-images', 'cameraman.tif'))

#Output image paths (adjust the paths as needed)
AVG_SMOOTH_SCRATCH = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't1a.tif'))
GAUSS_SMOOTH_SCRATCH = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't1b.tif'))
SOBEL_SHARPEN_SCRATCH = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't1c.tif'))
AVG_SMOOTH_OPENCV = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't2a.tif'))
GAUSS_SMOOTH_OPENCV = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't2b.tif'))
SOBEL_SHARPEN_OPENCV = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't2c.tif'))
MARR_HILDRETH = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't3a.tif'))
CANNY = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't3b.tif'))
EDGE_MAP_MARR_HILDRETH = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't4a.tif'))
EDGE_MAP_CANNY = os.path.abspath(os.path.join(CWD, '..', 'output-images', 't4b.tif'))


def main():
  function_type = input("""What function would you like to run?
  (1) Smoothing filters from scratch
  (2) OpenCV built-in smoothing filter functions
  (3) Marr-Hildreth & Canny edge detection
  (4) Edge maps from scratch
  Enter to quit
  """)
  # Load cameraman image to be used in most questions
  if function_type != "4" or function_type != "":
    img = cv2.imread(CAMERAMAN, cv2.IMREAD_UNCHANGED)
    if img is None:
      print("Invalid path for cameraman")
      return

  if function_type == "1":
    smoothing_filter(scratch_smoothing, img)
  elif function_type == "2":
    smoothing_filter(opencv_smoothing, img)
    return
  elif function_type == "3":
    edge_detection(img)
  elif function_type == "4":
    edge_maps()
  elif function_type == "":
    return
  else:
    print("Invalid input")
  main()

def smoothing_filter(smooth_funcs, img):
  question = input("""Which question would you like to run?
  1. Averaging smoothing filter (filter size: 3x3)
  2. Gaussian smoothing filter (filter size: 7x7, sigma=1, mean=0)
  3. Sobel sharpening filter (filter size: 3x3, both horizontal and vertical)
  """)

  if question == "1":
    new_img = smooth_funcs.averaging_filter(img)
    if smooth_funcs == scratch_smoothing:
      cv2.imwrite(AVG_SMOOTH_SCRATCH, new_img)
    else:
      cv2.imwrite(AVG_SMOOTH_OPENCV, new_img)

  elif question == "2":
    new_img = smooth_funcs.gaussian_filter(img)
    if smooth_funcs == scratch_smoothing:
      cv2.imwrite(GAUSS_SMOOTH_SCRATCH, new_img)
    else:
      cv2.imwrite(GAUSS_SMOOTH_OPENCV, new_img)

  elif question == "3":
    new_img = smooth_funcs.sobel_filter(img)
    if smooth_funcs == scratch_smoothing:
      cv2.imwrite(SOBEL_SHARPEN_SCRATCH, new_img)
    else:
      cv2.imwrite(SOBEL_SHARPEN_OPENCV, new_img)

  else:
    print("Invalid input")
  return

def edge_detection(img):
  question = input("""Which question would you like to run?
  1. Marr-Hildreth edge detector
  2. Canny edge detector
  """)
  if question == "1":
    cv2.imwrite(MARR_HILDRETH, marr_hildreth(img))

  elif question == "2":
    cv2.imwrite(CANNY, canny(img))

  else:
    print("Invalid input")
  return

def edge_maps():
  question = input("""Which edge detection image would you like to use?
  1. Marr-Hildreth
  2. Canny
  """)

  if question == "1":
    img = cv2.imread(MARR_HILDRETH, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(EDGE_MAP_MARR_HILDRETH, edge_map(img))
    
  elif question == "2":
    img = cv2.imread(CANNY, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(EDGE_MAP_CANNY, edge_map(img))
  else:
    print("Invalid input")
  return

if __name__ == "__main__":
  main()
