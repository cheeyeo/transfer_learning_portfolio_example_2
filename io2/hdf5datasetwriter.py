import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, output_path, data_key="images", buf_size=1000):
        if os.path.exists(output_path):
          raise ValueError("The supplied `output_path` already exists and cannot be overwritten. Manually delete the file before continuing.", output_path)

        # open hdf5 database for writing and create 2 datasets: features, labels
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        self.buf_size = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) >= 0:
            self.flush()

        self.db.close()