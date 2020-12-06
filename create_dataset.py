# Create dataset of randomly rotated images
# Use this to create a dataset of rotated images for testing or supply your own
import cv2
from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input directory of images")
    ap.add_argument("-o", "--output", required=True, help="Path to output directory of rotated images")
    args = vars(ap.parse_args())

    # grabs paths to input images and limit to 10000
    img_paths = list(paths.list_images(args["dataset"]))[:10000]
    random.shuffle(img_paths)

    angles = {}
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(img_paths), widgets=widgets).start()

    for i, path in enumerate(img_paths):
        angle = np.random.choice([0, 90, 180, 270])
        img = cv2.imread(path)

        if img is None:
            continue

        img = imutils.rotate_bound(img, angle)
        base = os.path.sep.join([args["output"], str(angle)])

        if not os.path.exists(base):
            os.makedirs(base)

        ext = path[path.rfind("."):]
        output_path = [base, "image_{}{}".format(str(angles.get(angle, 0)).zfill(5), ext)]
        output_path = os.path.sep.join(output_path)

        cv2.imwrite(output_path, img)

        c = angles.get(angle, 0)
        angles[angle] = c + 1
        pbar.update(i)
    pbar.finish()

    for angle in sorted(angles.keys()):
        print("[INFO] Angle={}: {:,}".format(angle, angles[angle]))