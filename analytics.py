import os

import numpy as np

import tensorflow as tf

from collections import Counter

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss

from flags import FLAGS
from helpers import load_npy_files_from_folder

ROOT = FLAGS.ROOT
PREPROCESSED_FOLDER = FLAGS.PREPROCESSED_FOLDER
TRAIN_READY_FOLDER = FLAGS.TRAIN_READY_FOLDER


def count_classes(y_list):
    c = Counter([int(y) for y in y_list])
    print("".center(20, "_"))
    print("Class: count")
    for k, v in c.items():
        print(f"{k}: {v}")
    print("".center(20, "_"))


if __name__ == '__main__':

    x_list, x_paths = load_npy_files_from_folder(os.path.join(ROOT, TRAIN_READY_FOLDER, "train", "x"))
    y_list, y_paths = load_npy_files_from_folder(os.path.join(ROOT, TRAIN_READY_FOLDER, "train", "y"))

    count_classes(y_list)

    x = np.array(x_list)
    idx_x = np.expand_dims(np.arange(0, x.shape[0]), axis=1).astype(int)
    print(idx_x.shape)
    y = np.array(y_list)
    y_tf = tf.one_hot(y, depth=FLAGS.NUM_CLASSES)

    print(x.shape, y.shape, y_tf.shape)

    for entry, path, entry_tf in zip(y_list, y_paths, y_tf):
        print(entry, entry_tf, path)

    training_generator = BalancedBatchGenerator(idx_x, y, sampler=NearMiss(), batch_size=10)

    c = Counter()

    for idx_x, y in training_generator:
        c.update(y.tolist())

    print(c)
    print("Is it balanced?")