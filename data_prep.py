import os
import shutil
import json

import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt

from flags import FLAGS
from helpers import load_npy_files_from_folder

ROOT = FLAGS.ROOT
DATA_FOLDER = FLAGS.DATA_FOLDER
IMAGE_FOLDER = FLAGS.IMAGE_FOLDER
PREPROCESSED_FOLDER = FLAGS.PREPROCESSED_FOLDER
CHOSEN_MAX_LEN_FOLDER = FLAGS.CHOSEN_MAX_LEN_FOLDER
TRAIN_READY_FOLDER = FLAGS.TRAIN_READY_FOLDER
CROSS_VAL_FILE = FLAGS.CROSS_VAL_FILE
XVAL_SPLIT_FOLDER = FLAGS.XVAL_SPLIT_FOLDER
JSON_OUT = FLAGS.JSON_OUT

MAX_LEN = FLAGS.MAX_LEN

#  DONE: we need to aggregate the data from csv files for each id+date to one sample in dataset
#  DONE: Refer to preprocessed/x /y
#   define GRU network for processing the data
#  TODO: Continue in main


def create_paths_json(data_folder_path, json_output_path):
    file_paths = next(os.walk(data_folder_path))[-1]
    file_dict = {}

    for fp in file_paths:
        fp_list = fp.split(" ")
        id = fp_list[0][-3:]
        date = fp_list[1]
        label = fp_list[3][-1]
        exercise = fp_list[5][0:3]
        if id in file_dict.keys():
            if date in file_dict[id].keys():
                file_dict[id][date].append((fp, label))
            else:
                file_dict[id][date] = [(fp, label)]
        else:
            file_dict[id] = {date: [(fp, label)]}

    # checksum if exactly 5 entries
    for key in file_dict.keys():
        for k, v in file_dict[key].items():
            if len(v) != 5:
                print(f"WARNING: Entry ({key}, {k}) has {len(v)} exercise(s). You might consider deleting them.")

    with open(json_output_path, "w") as f:
        json.dump(file_dict, f, indent=4)


def norm_and_set_max_len(arr_list, max_len=None):
    if not max_len:
        max_len = max(a.shape[0] for a in arr_list)
    arrs = []
    for arr in arr_list:
        norm_arr = (arr - arr.mean(axis=0)) / arr.std(axis=0)
        if max_len - len(norm_arr) > 0:
            new_arr = np.pad(norm_arr, ((0, max_len-len(norm_arr)), (0, 0)))
        else:
            new_arr = norm_arr[:max_len, :]
        arrs.append(new_arr)

    return np.hstack(arrs)


def aggregate_samples_to_npy(data_path, json_path, preprocessed_path, max_len=600, plot_path=""):
    with open(json_path, "r") as f:
        paths = json.load(f)

    max_len_folder = f"max_len_{max_len}"
    sample_path = os.path.join(preprocessed_path, max_len_folder, "x")
    label_path = os.path.join(preprocessed_path, max_len_folder, "y")

    # create required folders if needed
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    if plot_path:
        full_plot_path = os.path.join(plot_path, max_len_folder)
        os.makedirs(full_plot_path, exist_ok=True)

    for pid, person in paths.items():
        for date, measurements in person.items():
            arr_list = []
            label_list = []
            for msrmnt in measurements:
                df = pd.read_csv(os.path.join(data_path, msrmnt[0]))
                df = df.iloc[:, 1:]  # remove time info
                arr_list.append(df.to_numpy())
                label_list.append(aggregate_labels(msrmnt[1]))

            # normalize features, pad to max length and convert to one big numpy array
            sample = norm_and_set_max_len(arr_list, max_len=max_len)

            assert np.mean(label_list) == label_list[0], "labels do not match"
            label = np.mean(label_list, dtype=int)

            specific_path = f"\\{pid}_{date}"

            np.save(sample_path+specific_path+".npy", sample)
            np.save(label_path+specific_path+".npy", label)

            # plotting
            if plot_path:
                fig = plt.figure()
                plt.pcolormesh(sample)
                plt.title(f"person: {pid} | date: {date}")
                plt.xlabel("features")
                plt.ylabel("time index")
                fig.savefig(full_plot_path+specific_path+".png", dpi=150)
                plt.close(fig)


def aggregate_labels(label):
    label = int(label)
    if label == 1:
        return 0
    elif label == 2:
        return 1
    elif label == 3:
        return 2
    elif label == 4:
        return 2
    elif label == 5:
        return 3
    elif label == 6:
        return 3
    else:
        raise ValueError("Not a valid label value")


def generate_balanced_subset(full_dataset_folder_path, subset_folder_path, samples_per_class=3, move=False):

    os.makedirs(os.path.join(subset_folder_path, "x"), exist_ok=True)
    os.makedirs(os.path.join(subset_folder_path, "y"), exist_ok=True)

    x_list, x_paths = load_npy_files_from_folder(os.path.join(full_dataset_folder_path, "x"))
    y_list, y_paths = load_npy_files_from_folder(os.path.join(full_dataset_folder_path, "y"))

    x = np.array(x_list)
    idx_x = np.expand_dims(np.arange(0, x.shape[0]), axis=-1).astype(int)
    y = np.array(y_list)

    rus = RandomUnderSampler(random_state=42)
    idx_x_res, y_res = rus.fit_resample(idx_x, y)

    class_dict = {r: [] for r in set(y_res)}

    for xr, yr in zip(idx_x_res, y_res):
        class_dict[yr].append(xr[0])

    x_sub_path_list = []
    y_sub_path_list = []

    # choose random subset of unique samples from each class
    for k, idx_list in class_dict.items():
        choices = np.random.choice(idx_list, samples_per_class, replace=False)
        x_sub_path_list.extend([x_paths[i] for i in choices])
        y_sub_path_list.extend([y_paths[i] for i in choices])

    if move:
        # move chosen subset to target subset folder
        for x_sp, y_sp in zip(x_sub_path_list, y_sub_path_list):
            x_name = os.path.split(x_sp)[-1]
            y_name = os.path.split(y_sp)[-1]
            assert x_name == y_name
            os.rename(x_sp, os.path.join(subset_folder_path, "x", x_name))
            os.rename(y_sp, os.path.join(subset_folder_path, "y", y_name))

    return x_sub_path_list, y_sub_path_list


def preload_data_from_excel(path_to_file):
    """ Load important data for creating the cross-validation dataset (group_id, path_to_file)

    :param path_to_file:
    :return:
    """
    with open(path_to_file, "rb") as f:
        df = pd.read_excel(f)

    return df


def get_xVal_info(df):
    """

    :param df: xVal dataframe with the relevant columns
    :return: xVal_info (Set[Tuple(str, str, int, int)]) of (path to x, path to y, doctor eval, cross-validation group)
    """
    names = df.iloc[:, 2]
    doctor_labels = df.iloc[:, 4]
    xVal_group = df.iloc[:, -3]

    xVal_info = []

    for name, lab, group in set(zip(names, doctor_labels, xVal_group)):
        id, date = name.split()
        label = aggregate_labels(lab)

        file_name = f"{id[-3:]}_{date}.npy"
        xVal_info.append((os.path.join(chosen_max_len_path, "x", file_name),
                          os.path.join(chosen_max_len_path, "y", file_name),
                          label,
                          group))

    return xVal_info


def split_and_copy_to_xval_folders(xVal_info, target_folder):
    """ Split to folders 1-5 based on cross-validation test set group (given group is test set, rest is train set)

    :param xVal_info:
    :param target_folder:
    :return:
    """

    for i in range(1, 6):
        os.makedirs(os.path.join(target_folder, str(i), "x"), exist_ok=True)
        os.makedirs(os.path.join(target_folder, str(i), "y"), exist_ok=True)

    for i, (x_old_path, y_old_path, doc_eval, xval_group) in enumerate(xVal_info):
        y = np.load(y_old_path)
        assert int(y) == doc_eval, "Doctor labels in npy files must match with the excel sheet!"
        x_name = os.path.split(x_old_path)[-1]
        y_name = os.path.split(y_old_path)[-1]
        assert x_name == y_name, "x and y file names must match!"
        x_new_path = os.path.join(target_folder, str(xval_group), "x", x_name)
        y_new_path = os.path.join(target_folder, str(xval_group), "y", y_name)
        shutil.copy(x_old_path, x_new_path)
        shutil.copy(y_old_path, y_new_path)
        print(f"Copied {x_name} to xval group {xval_group} ({i+1}/{len(xVal_info)}) \n\tx: {x_new_path}\n\ty: {y_new_path}")
    print(f"DONE".center(100, "_"))


if __name__ == '__main__':
    data_path = os.path.join(ROOT, DATA_FOLDER)
    json_path = os.path.join(ROOT, JSON_OUT)
    preprocessed_path = os.path.join(ROOT, PREPROCESSED_FOLDER)
    chosen_max_len_path = os.path.join(ROOT, PREPROCESSED_FOLDER, CHOSEN_MAX_LEN_FOLDER)
    validation_subset_path = os.path.join(ROOT, TRAIN_READY_FOLDER, "valid")
    cross_val_file_path = os.path.join(ROOT, CROSS_VAL_FILE)
    target_xval_split_path = os.path.join(ROOT, XVAL_SPLIT_FOLDER)
    plot_path = os.path.join(ROOT, IMAGE_FOLDER)

    # create_paths_json(data_path, json_path)

    # aggregate_samples_to_npy(data_path, json_path, preprocessed_path, max_len=MAX_LEN, plot_path=plot_path)

    # x_subset, y_subset = generate_balanced_subset(chosen_max_len_path, validation_subset_path, 3, move=True)

    xVal_df = preload_data_from_excel(cross_val_file_path)

    # TODO: get file names from patient date column
    # names = df_xVal.iloc[:, 0]
    xVal_info = get_xVal_info(xVal_df)

    assert len(xVal_info) == 122

    # TODO: copy files to individual folders based on xVal group (xVal_gid/x, xVal_gid/y)
    split_and_copy_to_xval_folders(xVal_info, target_xval_split_path)

    # TODO: check if eval checks out with validation numpy files
