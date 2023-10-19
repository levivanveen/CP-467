import cv2
import numpy as np

THRESHOLD_HIGH = 255
THRESHOLD_LOW = 120

ZERO_CROSSSING_THRESHOLD = 10

GAUSSIAN_FILTER_SIZE = 7
SIGMA = 1

def marr_hildreth(img):
  # Apply gaussian filter
  blurred = cv2.GaussianBlur(img, (GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIZE), SIGMA)
  # Apply laplacian filter
  laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
  # Find zero crossings
  edge = _zero_crossings(laplacian, ZERO_CROSSSING_THRESHOLD)
  return edge.astype(np.uint8)

def _zero_crossings(img, threshold=0):
  rows, cols = img.shape
  edge_img = np.zeros_like(img)
  
  for i in range(1, rows - 1):
    for j in range(1, cols - 1):
      neighbours = [
        img[i + 1, j], # pixel below
        img[i - 1, j], # pixel above
        img[i, j + 1], # pixel to the right
        img[i, j - 1], # pixel to the left
        img[i + 1, j + 1], # pixel bottom right
        img[i + 1, j - 1], # pixel bottom left
        img[i - 1, j + 1], # pixel top right
        img[i - 1, j - 1] # pixel top left
      ]
      pos_neighbours = sum(1 for neighbour in neighbours if neighbour > threshold)
      neg_neighbours = sum(1 for neighbour in neighbours if neighbour < -threshold)
      if pos_neighbours > 0 and neg_neighbours > 0:
        edge_img[i, j] = 255
  return edge_img

def canny(img):
  edges = cv2.Canny(img, THRESHOLD_LOW, THRESHOLD_HIGH)
  return edges

def edge_map(img):
  curlab = 1
  rows, cols = img.shape
  queue = []
  # Store labels in a dictionary
  labels = {}

  for i in range(rows):
    for j in range(cols):
      if img[i,j] == 0 or (i,j) in labels:
        continue

      # pixel is foreground
      queue.append((i,j))
      labels[(i,j)] = curlab
      # Set all connected pixels to the same label
      while len(queue) > 0:
        pixel = queue.pop(0)
        neighbours = _get_neighbours(pixel, img)
        for neighbour in neighbours:
          row = neighbour[0]
          col = neighbour[1]
          if (row, col) not in labels and img[row,col] == 255:
            labels[(row,col)] = curlab
            queue.append((row,col))
      # Increment label
      curlab += 1

  
  # Create the colour map image
  colours = []
  for _ in range(curlab):
    colours.append(np.random.randint(0, 255, size=3))

  colour_map = np.zeros((rows, cols, 3), dtype=np.uint8)
  for i in range(rows):
    for j in range(cols):
      if (i, j) in labels:
        colour_map[i, j] = colours[labels[(i, j)]]
  return colour_map

def _get_neighbours(pixel, img):
  i, j = pixel
  rows, cols = img.shape
  neighbours = [
    (i - 1, j - 1), # top left
    (i - 1, j), # above
    (i - 1, j + 1), # top right
    (i, j - 1), # left
    (i, j + 1), # right
    (i + 1, j - 1), # bottom left
    (i + 1, j), # below
    (i + 1, j + 1) # bottom right
  ]
  valid_neighbours = [(x,y) for x, y in neighbours if 0 <= x < rows and 0 <= y < cols]
  return valid_neighbours
