# Uses pretrained VGG16 model as a feature extractor

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from io2.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
    ap.add_argument("-o", "--output", required=True, help="Path to output dataset")
    ap.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size of image through network")
    ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="Feature extraction buffer size")
    args = vars(ap.parse_args())

    bs = args["batch_size"]

    print("[INFO] Loading images...")
    img_paths = list(paths.list_images(args["dataset"]))
    random.shuffle(img_paths)

    labels = [p.split(os.path.sep)[-2] for p in img_paths]
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print("[INFO] Loading network...")
    model = VGG16(weights="imagenet", include_top=False)
    model.summary()

    # init the dataset writer 
    dataset = HDF5DatasetWriter((len(img_paths), 512*7*7), args["output"], data_key="features", buf_size=args["buffer_size"])
    dataset.store_class_labels(le.classes_)

    widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(img_paths), widgets=widgets).start()

    for i in np.arange(0, len(img_paths), bs):
        batch_paths = img_paths[i:i+bs]
        batch_labels = labels[i:i+bs]
        batch_imgs = []

        for (j, path) in enumerate(batch_paths):
            img = load_img(path, target_size=(224, 224))
            img = img_to_array(img)
            # preprocess by expanding dimensions and subtracting mean RGB from imagenet dataset
            img = np.expand_dims(img, axis=0)
            img = imagenet_utils.preprocess_input(img)
            batch_imgs.append(img)

        batch_imgs = np.vstack(batch_imgs)
        features = model.predict(batch_imgs, batch_size=bs)
        # reshape flattened feature vector of MaxPooling2D output
        features = features.reshape((features.shape[0], 512*7*7))

        dataset.add(features, batch_labels)

        pbar.update(i)

    dataset.close()
    pbar.finish()