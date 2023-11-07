import os
import cv2
from circle_detection import eye_detection

def main():
  # File path constants
  INPUT_IMAGES = os.path.join(os.getcwd(), '..', 'input-images')
  EYE_FILE = lambda filename: os.path.join(os.getcwd(), '..', 'input-images', filename)
  OUTPUT_IMAGE = lambda filename: os.path.join(os.getcwd(), '..', 'Segmented_Results', filename)

  eye_images = []
  for filename in os.listdir(INPUT_IMAGES):
    eye_images.append(EYE_FILE(filename))
  
  for eye_file in eye_images:
    eye_img = cv2.imread(eye_file, cv2.IMREAD_UNCHANGED)
    if eye_img is None:
      print("Invalid path for eye image")
      return
    # Run the circle detection function 
    circle = eye_detection(eye_img)
    if circle is None:
      print("No circle detected")
    else:
      cv2.imwrite(OUTPUT_IMAGE(os.path.basename(eye_file)), circle)
  return

if __name__ == "__main__":
  curr_dir = os.getcwd()
  file_dir = os.path.dirname(os.path.abspath(__file__))

  if curr_dir != file_dir:
    os.chdir(file_dir)
    print(f"Changed the current working directory to: {file_dir}")
  else:
    print("The current working directory is already the same as the directory of the current file.")
  main()