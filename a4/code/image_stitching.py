import cv2
import numpy as np

def stitch_images(img1, img2):
  # Convert images to grayscale
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  # Use the ORB (Oriented FAST and Rotated BRIEF) feature detector
  orb = cv2.ORB_create()

  # Find the keypoints and descriptors with ORB
  keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

  # Use the Brute Force Matcher to find the best matches
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(descriptors1, descriptors2)

  # Sort them in ascending order of distance
  matches = sorted(matches, key=lambda x: x.distance)

  # Select the first 10% of matches
  num_matches = int(len(matches) * 0.4)
  matches = matches[:num_matches]

  # Extract the matched keypoints
  points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
  points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

  # Find the homography matrix
  H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Warp the second image using the homography matrix
  warped_image = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

  # Combine the two images
  if len(warped_image.shape) == 2:
    stitched_image = np.zeros_like(warped_image)
    stitched_image[:img2.shape[0], :img2.shape[1]] = img2
  else:
    stitched_image = np.zeros_like(warped_image)
    stitched_image[:img2.shape[0], :img2.shape[1], :] = img2[:, :, :warped_image.shape[2]]

  stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_image, 0.5, 0)

  return stitched_image