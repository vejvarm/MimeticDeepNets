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

    CHOSEN_MODEL = "cnn-gru"  # currenty implemented "gru" and "cnn"

    NUM_CLASSES = 4
    PARAMS = {"gru": {"gru1": 512,
                      "gru2": 256,
                      "gru3": 128,
                      "dense": 32,
                      "drop_rate": 0.5},
              "cnn": {"cnl1": (16, (16, 8), (1, 1)),
                      "drop1": 0.0,
                      "cnl2": (32, (8, 4), (1, 1)),
                      "drop2": 0.0,
                      "cnl3": (64, (4, 2), (1, 1)),
                      "drop3": 0.0,
                      "dense": 32,
                      "drop_rate": 0.5},
              "cnn-gru": {"cnl1": (32, (16, 8), (2, 1)),
                          "drop1": 0.0,
                          "cnl2": (64, (8, 4), (2, 1)),
                          "drop2": 0.0,
                          "reshape_size": 20*64,
                          "gru1": 128,
                          "gru2": 64,
                          "dense": 32,
                          "drop_rate": 0.5}
              }

    NUM_EPOCHS = 500
    LR = [1e-3, 5e-4, 1e-4, 1e-5]  # GRID SEARCH param
    BATCH_SIZE = [8, 16, 32]  # GRID SEARCH param
    SHUFFLE_BUFFER_SIZE = 220

    TEST_BATCH_SIZE = 4

    METRICS = ["accuracy",
               tf.keras.metrics.Recall(name='recall'),
               F1Score(NUM_CLASSES, average="weighted", name="f1score"),
               tf.keras.metrics.AUC(name='auc')]

    TRAIN_VERBOSE = 0

    # Time stamps to evaluate in prediction
    PREDICTION_TIME_STAMPS = ['1615815770.256345', '1615816668.2819135', '1615817678.6957397', '1615818294.862513', '1615820067.391902', '1615820672.612466', '1615817135.3869963', '1615819294.1088831', '1615818864.5320156']
