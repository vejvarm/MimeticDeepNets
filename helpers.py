import json
import os
import re
import pickle

from collections import Counter

import numpy as np

import tensorflow as tf

from flags import FLAGS

def load_npy_files_from_folder(folder):
    folder, _, files = next(os.walk(folder))

    path_list = []
    array_list = []

    for file in files:
        pth = os.path.join(folder, file)
        path_list.append(pth)
        array_list.append(np.load(pth))

    return array_list, path_list


def check_ds_balance(ds):
    """ Check label distribution in dataset

    :param ds: tensorflow data.Dataset structure with (x, y) data
    :return:
    """
    #
    c = Counter()

    for x, y in ds:
        c.update([str(y_.numpy()) for y_ in y])

    return c


def calc_class_weights(ds_train):
    """ Calculate class weights for countering imbalanced distribution of classes in training data
    Class weights are inversely proportional to the number of samples of given class versus the total number of samples
    (useful: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)

    :param ds_train:
    :return: Dict[] (1/class_samples)*(total_samples/2) for each class
    """

    c = Counter()
    total_samples = len(ds_train)
    class_weights = dict()
    initial_bias = []

    # Calculate number of samples for each class
    for x, y in ds_train:
        c.update([tuple(y.numpy().tolist())])

    # Calculate weight ratio for each class
    for k, class_samples in c.items():
        class_weights[np.argmax(k)] = 1/class_samples*total_samples/2
        initial_bias.append(class_samples/total_samples)

    return class_weights, initial_bias


def initialize_callbacks(results_folder,
                         checkpoint_dir=FLAGS.CHECKPOINT_FOLDER,
                         log_dir=FLAGS.LOG_FOLDER,
                         grid_search_subfolders="",
                         checkpoint_name="weights_ep{epoch:03d}-ac{val_accuracy:.2f}-rc{val_recall:.2f}-fs{val_f1score:.2f}",
                         checkpoint_metric='val_f1score',
                         early_stop_metric='val_auc',
                         patience=100,
                         restore_best_weights=True):

    checkpoint_path = os.path.join(results_folder, checkpoint_dir, grid_search_subfolders, checkpoint_name)
    log_path = os.path.join(results_folder, log_dir, grid_search_subfolders)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=checkpoint_metric, verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=True, mode='max', save_freq='epoch',
                                                          options=None)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=early_stop_metric,
                                                      verbose=1,
                                                      patience=patience,
                                                      mode='max',
                                                      restore_best_weights=restore_best_weights)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    return [model_checkpoint, early_stopping, tensorboard]


def save_to_pickle(path_to_file, obj, verbosity=0):
    with open(path_to_file, "wb") as f:
        pickle.dump(obj, f)
        if verbosity:
            print(f"Object saved to {path_to_file}")


def load_from_pickle(path_to_file):
    with open(path_to_file, "rb") as f:
        return pickle.load(f)


def make_train_and_test_from_dict_of_datasets(ds_dict, keys_for_test_set=tuple(), verbosity=0):
    ds_train = None
    ds_test = None

    train_keys = []
    test_keys = []
    for key, ds in ds_dict.items():
        if key in keys_for_test_set:
            if ds_test:
                ds_test = ds_test.concatenate(ds)
            else:
                ds_test = ds
            test_keys.append(key)
        else:
            if ds_train:
                ds_train = ds_train.concatenate(ds)
            else:
                ds_train = ds
            train_keys.append(key)

    if verbosity:
        print("Key distribution:")
        print("train ds keys: " + str(train_keys))
        print("test ds keys: " + str(test_keys))

    return ds_train, ds_test if ds_test else tf.data.Dataset.from_tensor_slices([])


def iterative_ds_concatenate(full_ds, ds):
    if full_ds:
        full_ds = full_ds.concatenate(ds)
    else:
        full_ds = ds

    return full_ds


def decode_class(class_idx):
    """

    :param class_idx: index of class from network prediction (from 0 to 3)
    :return: original labels from practitioners (1, 2, 3 or 6)
    """
    class_idx += 1  # back to indexing from 1

    if class_idx == 4:
        class_idx = 6

    return class_idx


def list_all_eval_results(base_folder):

    path_gen = os.walk(base_folder)

    paths_to_eval_results = []

    for abspath, subfolders, files in path_gen:
        for sub in subfolders:
            if re.match(r"\b\d{10}.\d+\b", sub):
                full_path = os.path.join(abspath, sub, "eval_results.json")
                if os.path.exists(full_path):
                    paths_to_eval_results.append(os.path.join(abspath, sub, "eval_results.json"))

    return paths_to_eval_results


def load_all_eval_results(paths_to_eval_results):
    eval_results_list = []
    for path in paths_to_eval_results:
        with open(path, "r") as f:
            eval_results_list.append(json.load(f))

    return eval_results_list


def sort_results_by_metric(checkpoint_folder, metric="f1score"):
    paths_to_eval_results = list_all_eval_results(checkpoint_folder)
    list_of_eval_results = load_all_eval_results(paths_to_eval_results)

    score_path_list = []

    for path, entry in zip(paths_to_eval_results, list_of_eval_results):
        score_sum = 0.
        n = 0
        for xval_id, metric_dict in entry.items():
            score_sum += metric_dict[metric]
            n += 1

        score_average = score_sum/n
        score_path_list.append((score_average, os.path.split(path)[0]))

    # sort by score_average
    score_path_list.sort(key=lambda x: x[0], reverse=True)

    timestamp_list = [os.path.split(p)[-1] for s, p in score_path_list]

    return score_path_list, timestamp_list
