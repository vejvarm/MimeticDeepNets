import os

from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from helpers import load_npy_files_from_folder, calc_class_weights, make_train_and_test_from_dict_of_datasets
from flags import FLAGS

ROOT = FLAGS.ROOT
XVAL_SPLIT_FOLDER = os.path.join(ROOT, FLAGS.XVAL_SPLIT_FOLDER)


if __name__ == '__main__':
    xval_ds_dict = {}
    for xval_group in FLAGS.XVAL_GROUPS:
        arr_list_train_x, _ = load_npy_files_from_folder(os.path.join(XVAL_SPLIT_FOLDER, str(xval_group), "x"))
        arr_list_train_y, _ = load_npy_files_from_folder(os.path.join(XVAL_SPLIT_FOLDER, str(xval_group), "y"))

        ds = Dataset.from_tensor_slices((np.array(arr_list_train_x),
                                         tf.one_hot(np.array(arr_list_train_y).astype(int), depth=FLAGS.NUM_CLASSES)))

        xval_ds_dict[xval_group] = ds

    for test_group in FLAGS.XVAL_GROUPS:

        c_test = Counter()
        c_train = Counter()

        # create ds_test and ds_train
        ds_train, ds_test = make_train_and_test_from_dict_of_datasets(xval_ds_dict,
                                                                      keys_for_test_set=[test_group],
                                                                      verbosity=1)

        c = calc_class_weights(ds_train)

        for x, y in ds_test:
            c_test.update([tuple(y.numpy().tolist())])

        print(c_test)

        for x, y in ds_train:
            c_train.update([tuple(y.numpy().tolist())])

        print(c_train)