import tensorflow as tf

from tensorflow_addons.metrics import F1Score

class FLAGS:
    tf.keras.backend.set_floatx('float32')

    ROOT = "B:\\!Cloud\\OneDrive - VŠCHT\\VŠCHT\\FaceMedical"
    DATA_FOLDER = "full"
    IMAGE_FOLDER = "images"
    RESULTS_FOLDER = "results"
    PREPROCESSED_FOLDER = "preprocessed"
    CHOSEN_MAX_LEN_FOLDER = "max_len_230"
    TRAIN_READY_FOLDER = "train_ready"
    CROSS_VAL_FILE = "xVal_info.xlsx"
    XVAL_SPLIT_FOLDER = "xval"
    LOG_FOLDER = "logs"
    CHECKPOINT_FOLDER = "checkpoints"
    JSON_OUT = "paths.json"

    XVAL_GROUPS = list(range(1, 6))

    MAX_LEN = 230

    CHOSEN_MODEL = "cnn"  # currenty implemented "gru" and "cnn"

    NUM_CLASSES = 4
    GRUPARAMS = {"gru1": 128,
                 "gru2": 64,
                 "dense": 32,
                 "drop_rate": 0.5}

    CNNPARAMS = {"cnl1": (8, (16, 8), (1, 1)),
                 "cnl2": (16, (8, 4), (1, 1)),
                 "cnl3": (32, (4, 2), (1, 1)),
                 "dense": 32,
                 "drop_rate": 0.5}

    NUM_EPOCHS = 400
    LR = [1e-3, 1e-4, 1e-5, 1e-6]  # GRID SEARCH param
    BATCH_SIZE = [4, 8, 16]  # GRID SEARCH param
    SHUFFLE_BUFFER_SIZE = 220

    TEST_BATCH_SIZE = 4

    METRICS = ["accuracy",
               tf.keras.metrics.Recall(name='recall'),
               F1Score(NUM_CLASSES, average="weighted", name="f1score"),
               tf.keras.metrics.AUC(name='auc')]

    TRAIN_VERBOSE = 0

    # Parameters for predicting and evaluating final results from checkpointed models in given folders:
    PRED_LR = [1e-4, ]
    PRED_BATCH_SIZE = [1, 4, 8, 16]
