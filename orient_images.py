# Applies trained LG model to rotate images
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--db", required=True, help="Path to HDF5 features")
    ap.add_argument("-i", "--dataset", required=True, help="Path to rotated images")
    ap.add_argument("-m", "--model", required=True, help="Path to saved model")
    args = vars(ap.parse_args())

    # Load labels from dataset
    db = h5py.File(args["db"], "r")
    label_names = [int(angle) for angle in db["label_names"][:]]
    db.close()

    print("[INFO] Sampling images...")
    image_paths = list(paths.list_images(args["dataset"]))
    image_paths = np.random.choice(image_paths, size=(10, ), replace=False)

    print("[INFO] Loading VGG 16 WEIGHTS...")
    vgg = VGG16(weights="imagenet", include_top=False)

    print("[INFO] Loading model...")
    with open(args["model"], "rb") as f:
        model = pickle.loads(f.read())

    for img_path in image_paths:
        orig = cv2.imread(img_path)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)

        img = np.expand_dims(img, axis=0)
        img = imagenet_utils.preprocess_input(img)

        features = vgg.predict(img)
        features = features.reshape((features.shape[0], 512*7*7))

        angle = model.predict(features)
        angle = label_names[angle[0]]

        rotated = imutils.rotate_bound(orig, 360 - angle)

        cv2.imshow("Original", orig)
        cv2.imshow("Corrected", rotated)
        cv2.waitKey(0)