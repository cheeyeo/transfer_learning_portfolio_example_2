# Training script for using the extracted features and building a LogisticRegression model to learn the features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
import argparse
import pickle
import h5py

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--db", required=True, help="Path to features HDF5 database")
    ap.add_argument("-m", "--model", required=True, help="Path to output model")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="Nos of jobs to run when tuning hyper parameter")
    args = vars(ap.parse_args())

    db = h5py.File(args["db"], "r")
    # get index of train/test split
    i = int(db["labels"].shape[0] * 0.75)

    print("[INFO] Tuning hyperparameters...")
    params = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    lg = LogisticRegression(max_iter=1000, random_state=1)

    model = GridSearchCV(lg, params, cv=cv, n_jobs=args["jobs"], refit=True)

    model.fit(db["features"][:i], db["labels"][:i])

    print("[INFO] Best hyperparams: {}".format(model.best_params_))

    print("[INFO] Evaluating model...")
    preds = model.predict(db["features"][i:])
    print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

    print("[INFO] Saving model...")
    with open(args["model"], "wb") as f:
        f.write(pickle.dumps(model.best_estimator_))

    db.close()