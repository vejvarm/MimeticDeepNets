import os

from flags import FLAGS
from helpers import list_all_eval_results, load_all_eval_results, sort_results_by_metric, time_stamp_to_full_path

BASE_FOLDER = os.path.join(FLAGS.ROOT, FLAGS.RESULTS_FOLDER, FLAGS.CHECKPOINT_FOLDER)


if __name__ == '__main__':
    # paths_to_eval_results = list_all_eval_results(BASE_FOLDER)
    # list_of_eval_results = load_all_eval_results(paths_to_eval_results)

    sorted_list, timestamp_list = sort_results_by_metric(BASE_FOLDER, "f1score")

    print(sorted_list)
    print(timestamp_list)

    for ts in timestamp_list:
        full_path_to_ts = time_stamp_to_full_path(BASE_FOLDER, ts)

        print(ts, full_path_to_ts)