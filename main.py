import json
import os
import time

import numpy as np
import tensorflow as tf
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from tensorflow.data import Dataset

from flags import FLAGS
from helpers import load_npy_files_from_folder, calc_class_weights, initialize_callbacks, \
    make_train_and_test_from_dict_of_datasets, iterative_ds_concatenate, save_to_pickle, sort_results_by_metric,\
    time_stamp_to_full_path
from model import compile_model

ROOT = FLAGS.ROOT
RESULTS_FOLDER = os.path.join(ROOT, FLAGS.RESULTS_FOLDER)
PREPROCESSED_FOLDER = os.path.join(ROOT, FLAGS.PREPROCESSED_FOLDER)
TRAIN_READY_FOLDER = os.path.join(ROOT, FLAGS.TRAIN_READY_FOLDER)
XVAL_SPLIT_FOLDER = os.path.join(ROOT, FLAGS.XVAL_SPLIT_FOLDER)

# TODO:
#      DONE: Rozdělit data do Cross-Validačních skupin (1-5 podle tabulky excel od Pavla)
#      DONE: Jedna skupina je vždy testovací sada, zbytek jsou trénovací
#      DONE: Váhování tříd v závislosti na zastoupení v trénovací sadě (při aktualizaci gradientů)
#      DONE: OUTPUT: for each Xval_test group output trained class (only after model is trained)
#


def create_balanced_data_pipeline(x, y, batch_size=8):
    """

    :param x: (List[2Darray]) list of numpy array input samples
    :param y: (List[1Darray]) list of numpy array target values (labels)
    :param batch_size: (int) size of the training minibatches
    :return: data generator returning x, y with balanced label distribution
    """
    arr_x = tf.convert_to_tensor(np.array(x), dtype=tf.float32)
    idx_x = np.expand_dims(np.arange(0, len(arr_x)), axis=-1).astype(int)
    arr_y = np.array(y)

    train_gen = BalancedBatchGenerator(idx_x, arr_y, sampler=NearMiss(), batch_size=1)
    train_gen_tf = Dataset.from_generator(lambda: train_gen, output_types=(tf.int32, tf.int32), output_shapes=((1, 1), (1, )))
    train_gen_tf = train_gen_tf.map(lambda ix, lab: (arr_x[tf.squeeze(ix), :, :],
                                                     tf.squeeze(tf.one_hot(lab, depth=FLAGS.NUM_CLASSES))),
                                    num_parallel_calls=4)
    train_gen_tf = train_gen_tf.shuffle(FLAGS.SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(-1)

    return train_gen_tf


def create_balanced_oversampled_pipeline(x, y, batch_size=8, oversampler=SMOTE()):
    """

    :param x: (List[2Darray]) list of numpy array input samples
    :param y: (List[1Darray]) list of numpy array target values (labels)
    :param batch_size: (int) size of the training minibatches
    :param oversampler: (imblearn sampler object) (https://imbalanced-learn.org/stable/references/over_sampling.html)
    :return: data generator returning x, y with balanced label distribution
    """
    arr_x = tf.convert_to_tensor(np.array(x), dtype=tf.float32)
    idx_x = np.expand_dims(np.arange(0, len(arr_x)), axis=-1).astype(int)
    arr_y = np.array(y)

    idx_res, y_res = oversampler.fit_resample(idx_x, arr_y)
    ds = Dataset.from_tensor_slices((idx_res, y_res))
    ds = ds.map(lambda ix, lab: (arr_x[tf.squeeze(ix), :, :],
                                                     tf.squeeze(tf.one_hot(lab, depth=FLAGS.NUM_CLASSES))),
                                    num_parallel_calls=4)
    ds = ds.shuffle(FLAGS.SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(-1)

    return ds


def train(model_type: str = FLAGS.CHOSEN_MODEL):
    # Load unbalanced data into tensorflow datasets
    xval_ds_dict = {}
    for xval_group in FLAGS.XVAL_GROUPS:
        arr_list_train_x, _ = load_npy_files_from_folder(os.path.join(XVAL_SPLIT_FOLDER, str(xval_group), "x"))
        arr_list_train_y, _ = load_npy_files_from_folder(os.path.join(XVAL_SPLIT_FOLDER, str(xval_group), "y"))

        ds = Dataset.from_tensor_slices((np.array(arr_list_train_x).astype(float),
                                        tf.one_hot(np.array(arr_list_train_y).astype(int), depth=FLAGS.NUM_CLASSES)))

        xval_ds_dict[xval_group] = ds

    # GRID SEARCH:
    for lr in FLAGS.LR:
        for bs in FLAGS.BATCH_SIZE:
            ts = time.time()
            eval_results_dict = {}
            relative_path_to_current_run = os.path.join(f"lr-{lr}", f"bs-{bs}", str(ts))
            for test_group in FLAGS.XVAL_GROUPS:
                print(f"Current xVal test group: {test_group}")

                # create ds_test and ds_train for current xval group
                ds_train, ds_test = make_train_and_test_from_dict_of_datasets(xval_ds_dict,
                                                                              keys_for_test_set=[test_group],
                                                                              verbosity=0)

                # Create balanced generator for training data
                # train_ds_balanced = create_balanced_data_pipeline(arr_list_train_x, arr_list_train_y, bs)
                # train_ds_balanced = create_balanced_oversampled_pipeline(arr_list_train_x, arr_list_train_y, bs)
                # DONE: instead of arr_train_x, generate indices and then sample the data from arr_train_x

                # Calculate current class weights for imbalanced train set
                class_weights, initial_bias = calc_class_weights(ds_train)

                # Shuffle and batch datasets
                ds_train = ds_train.shuffle(FLAGS.SHUFFLE_BUFFER_SIZE).batch(bs)
                ds_test = ds_test.batch(FLAGS.TEST_BATCH_SIZE)

                # DEFINE CALLBACKS
                callbacks = initialize_callbacks(results_folder=RESULTS_FOLDER,
                                                 grid_search_subfolders=os.path.join(relative_path_to_current_run,
                                                                                     f"xval-{test_group}"))

                # COMPILE MODEL
                model = compile_model(model_type=model_type, lr=lr, initial_output_bias=initial_bias)

                # TRAIN MODEL (weighted classes)
                model.fit(ds_train, validation_data=ds_test, epochs=FLAGS.NUM_EPOCHS, callbacks=callbacks,
                          class_weight=class_weights, verbose=FLAGS.TRAIN_VERBOSE)

                # EVALUATE MODEL for current xval test group (indicative of overall RUN performance)
                eval_results_dict[test_group] = model.evaluate(ds_test, return_dict=True)

            # SAVE EVAL Results to current run folder
            path_to_current_run_checkpoint_folder = os.path.join(RESULTS_FOLDER,
                                                                 FLAGS.CHECKPOINT_FOLDER,
                                                                 relative_path_to_current_run)
            with open(os.path.join(path_to_current_run_checkpoint_folder, "eval_results.json"), "w") as f:
                json.dump(eval_results_dict, f, indent=4)
            with open(os.path.join(path_to_current_run_checkpoint_folder, "model_params.json"), "w") as f:
                json.dump(FLAGS.PARAMS[model_type], f, indent=4)

            print("TEST GROUP ACCURACY".center(30, "-"))
            print("|"+f"lr: {lr:.5f} | bs: {bs:03d}".center(28, " ")+"|")
            print("\n".join(["|"+f"{xval_grp:01d}: {d['accuracy']:.3f}".center(28, " ")+"|"
                             for xval_grp, d in eval_results_dict.items()]))
            print("".center(30, "-"))


def predict(time_stamp_list=FLAGS.PREDICTION_TIME_STAMPS, score_path_list=None, model_type: str = FLAGS.CHOSEN_MODEL):
    # Load unbalanced data into tensorflow datasets
    full_ds = None
    full_ds_info = None

    for xval_group in FLAGS.XVAL_GROUPS:
        arr_list_train_x, paths_x = load_npy_files_from_folder(
            os.path.join(XVAL_SPLIT_FOLDER, str(xval_group), "x"))
        arr_list_train_y, paths_y = load_npy_files_from_folder(
            os.path.join(XVAL_SPLIT_FOLDER, str(xval_group), "y"))

        ds = Dataset.from_tensor_slices((np.array(arr_list_train_x).astype(float),
                                         tf.one_hot(np.array(arr_list_train_y).astype(int),
                                                    depth=FLAGS.NUM_CLASSES)))

        ds_info = Dataset.from_tensor_slices(
            ([os.path.split(p)[-1] for p in paths_x], [xval_group] * len(ds)))  # names and xval group

        full_ds = iterative_ds_concatenate(full_ds, ds)
        full_ds_info = iterative_ds_concatenate(full_ds_info, ds_info)

    # batch datasets
    full_ds = full_ds.batch(1)
    full_ds_info = full_ds_info.batch(1)

    # GRID LOAD:
    full_result_dict = {}
    # print(current_checkpoint_folder)
    # print(time_stamps)
    for run_id, ts in enumerate(time_stamp_list):
        current_run_folder = time_stamp_to_full_path(os.path.join(RESULTS_FOLDER, FLAGS.CHECKPOINT_FOLDER), ts)
        result_dict = {}
        for test_group in FLAGS.XVAL_GROUPS:
            result_list = []
            print(f"Current xVal test group: {test_group}")

            # load model from checkpoint
            model = compile_model(model_type=model_type,
                                  checkpoint_folder=os.path.join(current_run_folder,
                                                                 f"xval-{test_group}"))

            # predict probabilities from loaded model
            probs = model.predict(full_ds)

            # infer most probable classes from probabilities
            predictions = np.argmax(probs, axis=1)

            # append to list of tuples: (file_name, xval_group, true_label, predicted_label, class_probs)
            for (fn, xgrp), (_, tl), pl, cprob in zip(full_ds_info, full_ds, predictions, probs):
                result_list.append((fn.numpy()[0], xgrp.numpy()[0], np.argmax(tl), pl, cprob))

            # sort list by name and date:
            result_list.sort(key=lambda tup: tup[0])

            # add sorted list to dict for current xval group
            result_dict[test_group] = result_list

        # save list to pickle file
        save_to_pickle(os.path.join(current_run_folder, "prediction_results.p"), result_dict,
                       verbosity=1)
        # add new entry to full results dict
        if score_path_list:
            full_result_dict[f"{score_path_list[run_id][0]:.2f}_ts-{ts}"] = result_dict
        else:
            full_result_dict[f"{run_id:03}_ts-{ts}"] = result_dict

    # Save full_result_dict
    save_to_pickle(os.path.join(RESULTS_FOLDER, "checkpoints", f"full_result_dict_{ts}.p"), full_result_dict,
                   verbosity=1)


if __name__ == "__main__":
    train()
    score_path_list, timestamp_list = sort_results_by_metric(os.path.join(RESULTS_FOLDER, FLAGS.CHECKPOINT_FOLDER),
                                                             metric="f1score")
    predict(timestamp_list, score_path_list)

    # TODO: run evaluation after this finishes