import os
from typing import Union, Tuple, List

import pandas as pd

from flags import FLAGS
from helpers import load_from_pickle, decode_class, sort_results_by_metric

ROOT = FLAGS.ROOT
RESULTS_FOLDER = FLAGS.RESULTS_FOLDER

FULL_PATH_TO_CHECKPOINTS = os.path.join(ROOT, RESULTS_FOLDER, "checkpoints")


def eval_results(time_stamps: Union[Tuple, List],
                 excel_file_path=os.path.join(FULL_PATH_TO_CHECKPOINTS, f"xVal_results.xlsx")):
    with pd.ExcelWriter(excel_file_path, mode="w") as writer:
        for ts in time_stamps:
            print(f"Evaluating results for time stamp: {ts}")
            full_results_dict_path = os.path.join(FULL_PATH_TO_CHECKPOINTS, f"full_result_dict_{ts}.p")

            full_results_dict = load_from_pickle(full_results_dict_path)

            for run_id, results_dict in full_results_dict.items():
                only_eval_dict = {cur_xval: [decode_class(data[3]) for data in data_list]
                                  for cur_xval, data_list in results_dict.items()}
                # convert to pandas dataframe
                df = pd.DataFrame(only_eval_dict)
                df.to_csv(os.path.join(FULL_PATH_TO_CHECKPOINTS, f"xVal_results_{run_id}.csv"), index=False, header=False)
                df.to_excel(writer, run_id)


if __name__ == '__main__':
    time_stamps_to_eval = ["1616007514.9154973"]
    eval_results(time_stamps_to_eval)

    metric = "f1score"

    score_path_list, _ = sort_results_by_metric(os.path.join(ROOT, RESULTS_FOLDER, "checkpoints"), metric)

    print(f"{metric}: {[s for s, p in score_path_list]}")
