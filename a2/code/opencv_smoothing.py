import cv2

AVERAGE_FILTER_SIZE = 3
GAUSSIAN_FILTER_SIZE = 7
SOBEL_FILTER_SIZE = 3
SIGMA = 1
MEAN = 0

# 3x3 smoothing average filter
def averaging_filter(img):
    return cv2.blur(img, (AVERAGE_FILTER_SIZE, AVERAGE_FILTER_SIZE))  # Change kernel size as needed

# 7x7 smoothing gaussian filter
def gaussian_filter(img):
    return cv2.GaussianBlur(img, (GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIZE), SIGMA)  # Change kernel size and sigma as needed

# Sobel Filter
def sobel_filter(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=SOBEL_FILTER_SIZE)  # Gradient in x-direction
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=SOBEL_FILTER_SIZE)  # Gradient in y-direction
    magnitude = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)  # Combine x and y gradients
    return magnitude