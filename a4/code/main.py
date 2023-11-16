import os
import cv2
from feature_detection import feat_detection
from image_manipulation import obj_addition, mask_creation
from image_stitching import stitch_images

#input images
CHESSBOARD = 'chessboard.png'
SHOES = 'shoes.png'
LIGHTNING = 'lightning.png'
SCENE = 'scene.png'
MASK = 'mask.png'
STITCH_1 = 'stitch_1.png'
STITCH_2 = 'stitch_2.png'

#output images
CHESSBOARD_FEAT = 'chessboard_feat.png'
SHOES_FEAT = 'shoes_feat.png'
LIGHTNING_ADDED = 'lightning_added.png'
STICHED = 'stitched.png'

#feature detection parameters
CIRCLE_PARAMS = {
  'chess': (95, 45, 20, 100),
  'shoes': (110, 35, 20, 100),
}

def main():
  update_paths()
  function_type = input("""What task would you like to run?
  (1) Feature Detection
  (2) Image Manipulation
  (3) Image Stitching
  Enter to quit
  """)

  if function_type == "1":
    feature_detection()
  elif function_type == "2":
    image_manipulation()
  elif function_type == "3":
    image_stitching()
  elif function_type == "":
    return
  else: 
    print("Invalid input")
  main()

def feature_detection():
  # Load images to be used in feature detection
  chessboard = cv2.imread(CHESSBOARD, cv2.IMREAD_UNCHANGED)
  shoes = cv2.imread(SHOES, cv2.IMREAD_UNCHANGED)

  # Run feature detection functions
  new_chess = feat_detection(chessboard, CIRCLE_PARAMS['chess'])
  new_shoes = feat_detection(shoes, CIRCLE_PARAMS['shoes'])
  
  # Save the new images
  cv2.imwrite(CHESSBOARD_FEAT, new_chess)
  cv2.imwrite(SHOES_FEAT, new_shoes)
  return

def image_manipulation():
  lightning = cv2.imread(LIGHTNING, cv2.IMREAD_UNCHANGED)
  scene = cv2.imread(SCENE, cv2.IMREAD_UNCHANGED)
  mask = cv2.imread(MASK, cv2.IMREAD_UNCHANGED)
  if mask is None:
    mask = mask_creation(lightning)
    cv2.imwrite(MASK, mask)

  lightning_added = obj_addition(scene, lightning, mask)
  cv2.imwrite(LIGHTNING_ADDED, lightning_added)
  return

def image_stitching():
  stitch_1 = cv2.imread(STITCH_1, cv2.IMREAD_UNCHANGED)
  stitch_2 = cv2.imread(STITCH_2, cv2.IMREAD_UNCHANGED)

  # Stitch the images together
  stitched = stitch_images(stitch_2, stitch_1)
  cv2.imwrite(STICHED, stitched)
  return

def update_paths():
  global CHESSBOARD, SHOES, CHESSBOARD_FEAT, SHOES_FEAT, LIGHTNING, SCENE, LIGHTNING_ADDED, MASK, STITCH_1, STITCH_2, STICHED
  current_directory = os.getcwd()
  # Input images
  MASK = os.path.join(current_directory, '..', 'input-images', 'mask.png')
  LIGHTNING = os.path.join(current_directory, '..', 'input-images', 'lightning.png')
  SCENE = os.path.join(current_directory, '..', 'input-images', 'scene.png')
  CHESSBOARD = os.path.join(current_directory, '..', 'input-images', 'chessboard.png')
  SHOES = os.path.join(current_directory, '..', 'input-images', 'shoes.png')
  STITCH_1 = os.path.join(current_directory, '..', 'input-images', 'stitch_1.png')
  STITCH_2 = os.path.join(current_directory, '..', 'input-images', 'stitch_2.png')
  # Ouput images
  CHESSBOARD_FEAT = os.path.join(current_directory, '..', 'output-images', 'chessboard_feat.png')
  SHOES_FEAT = os.path.join(current_directory, '..', 'output-images', 'shoes_feat.png')
  LIGHTNING_ADDED = os.path.join(current_directory, '..', 'output-images', 'lightning_added.png')
  STICHED = os.path.join(current_directory, '..', 'output-images', 'stitched.png')


if __name__ == "__main__":
  curr_dir = os.getcwd()
  file_dir = os.path.dirname(os.path.abspath(__file__))

  if curr_dir != file_dir:
    os.chdir(file_dir)
    print(f"Changed the current working directory to: {file_dir}")
  main()