import os

import numpy as np

import tensorflow as tf

from helpers import load_npy_files_from_folder
from model import compile_model
from flags import FLAGS

RUN_EVAL_OUTPUT_EXAMPLE = {
 1: {'loss': 1.3367092609405518, 'accuracy': 0.4000000059604645, 'recall': 0.0, 'auc': 0.630133330821991},
 2: {'loss': 1.3860043287277222, 'accuracy': 0.2800000011920929, 'recall': 0.0, 'auc': 0.5453332662582397},
 3: {'loss': 1.4508723020553589, 'accuracy': 0.1666666716337204, 'recall': 0.0, 'auc': 0.477430522441864},
 4: {'loss': 1.4121192693710327, 'accuracy': 0.0833333358168602, 'recall': 0.0, 'auc': 0.5095486044883728},
 5: {'loss': 1.4277423620224, 'accuracy': 0.1666666716337204, 'recall': 0.0, 'auc': 0.4939235746860504}
}

XVAL_SPLIT_FOLDER = os.path.join(FLAGS.ROOT, FLAGS.XVAL_SPLIT_FOLDER)
CHECKPOINT_FOLDER = "b:/!Cloud/OneDrive - VŠCHT/VŠCHT/FaceMedical/results/checkpoints/lr-5e-05/bs-16/xval-4/1615393473.1869242"

if __name__ == '__main__':
    model = compile_model(checkpoint_folder=CHECKPOINT_FOLDER)

    # LOAD DATASET:
    arr_list_train_x, _ = load_npy_files_from_folder(os.path.join(XVAL_SPLIT_FOLDER, "1", "x"))
    arr_list_train_y, _ = load_npy_files_from_folder(os.path.join(XVAL_SPLIT_FOLDER, "1", "y"))

    ds = tf.data.Dataset.from_tensor_slices((np.array(arr_list_train_x),
                                            tf.one_hot(np.array(arr_list_train_y).astype(int), depth=FLAGS.NUM_CLASSES)))
    ds = ds.batch(8)

    probs = model.predict(ds, )
    classes = np.argmax(probs, axis=1)

    print(probs.shape, probs)
    print(classes.shape, classes)