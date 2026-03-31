import os
import glob
import cv2
from keras_xception import preprocess_ham10000_image

base_dir = os.path.dirname(os.path.abspath(__file__))
jpg_paths = glob.glob(os.path.join(base_dir, "*.jpg"))

for path in jpg_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    preprocess_ham10000_image(img, plot_comparison=True)
