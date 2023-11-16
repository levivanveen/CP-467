import numpy as np
import cv2

def obj_addition(scene_img, obj_img, mask):
  # Convert the scene and object images to float32
  scene_32 = scene_img.astype(np.float32)
  obj_32 = obj_img.astype(np.float32)

  # Check the number of channels in each image
  if scene_32.shape[2] == 4 and obj_32.shape[2] == 3:
    # If the scene has 4 channels and the object has 3, remove the alpha channel from the scene
    scene_32 = scene_32[:, :, :3]
  elif scene_32.shape[2] == 3 and obj_32.shape[2] == 4:
    # If the scene has 3 channels and the object has 4, remove the alpha channel from the object
    obj_32 = obj_32[:, :, :3]

  result_img = np.copy(scene_32)
  mask_indices = np.nonzero(mask)

  # Iterate over non-zero pixels in the mask
  for y, x in zip(*mask_indices):
    result_img[y, x] = 0
    for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
      # Check that the pixel is within the bounds of the image
      if nx >= 0 and nx < scene_32.shape[1] and ny >= 0 and ny < scene_32.shape[0]:
        result_img[y, x] += obj_32[y, x] - obj_32[ny, nx]

        if mask[ny, nx] == 0:
          result_img[y, x] += scene_32[ny, nx]
        else:
          result_img[y, x] += obj_32[ny, nx]

  # Convert the result image back to uint8
  result_img = result_img.astype(np.uint8)
  return result_img

def mask_creation(image):
    # Display the image and let the user draw the mask manually
    clone = image.copy()
    cv2.namedWindow("Scene")
    cv2.imshow("Scene", clone)

    # Initialize the mask
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

    # Create a callback function for mouse events
    def draw_mask(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(mask, (x, y), 3, 255, -1)
            cv2.imshow("Scene", clone)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(clone, (x, y), 10, (0, 255, 0), -1)
            cv2.circle(mask, (x, y), 10, 255, -1)
            cv2.imshow("Scene", clone)

    # Set the callback function for mouse events
    cv2.setMouseCallback("Scene", draw_mask)

    # Wait for the user to draw the mask and press 'Enter'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break

    # Cleanup and close the window
    cv2.destroyAllWindows()

    return mask
