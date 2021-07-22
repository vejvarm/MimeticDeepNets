from abc import ABC

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, GRU, Conv2D, Dropout, BatchNormalization, Flatten, Reshape
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.data import Dataset

from helpers import random_dataset
from flags import FLAGS

NSAMPLES = 20
NCLASSES = FLAGS.NUM_CLASSES


class CNNModel(tf.keras.Model, ABC):
    """
    input_shape: (bs, width, height, channels)
    """
    cnl1_filters, cnl1_krnl_sz, cnl1_strd = FLAGS.PARAMS["cnn"]["cnl1"]
    drop_rate_1 = FLAGS.PARAMS["cnn"]["drop1"]
    cnl2_filters, cnl2_krnl_sz, cnl2_strd = FLAGS.PARAMS["cnn"]["cnl2"]
    drop_rate_2 = FLAGS.PARAMS["cnn"]["drop2"]
    cnl3_filters, cnl3_krnl_sz, cnl3_strd = FLAGS.PARAMS["cnn"]["cnl3"]
    drop_rate_3 = FLAGS.PARAMS["cnn"]["drop3"]
    dense_hidden = FLAGS.PARAMS["cnn"]["dense"]
    dense_drop_rate = FLAGS.PARAMS["cnn"]["drop_rate"]

    def __init__(self, output_bias=None, **kwargs):
        super(CNNModel, self).__init__(**kwargs)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.cnl1 = Conv2D(self.cnl1_filters, self.cnl1_krnl_sz, self.cnl1_strd)
        self.drop1 = Dropout(self.drop_rate_1)
        self.cnl2 = Conv2D(self.cnl2_filters, self.cnl2_krnl_sz, self.cnl2_strd)
        self.drop2 = Dropout(self.drop_rate_2)
        self.cnl3 = Conv2D(self.cnl3_filters, self.cnl3_krnl_sz, self.cnl3_strd)
        self.drop3 = Dropout(self.drop_rate_3)
        self.flat = Flatten()
        self.fc = Dense(self.dense_hidden)
        self.drop = Dropout(self.dense_drop_rate)
        self.out = Dense(FLAGS.NUM_CLASSES, bias_initializer=output_bias)

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, -1)
        x = relu(self.cnl1(x))
        x = self.drop1(x)
        x = relu(self.cnl2(x))
        x = self.drop2(x)
        x = relu(self.cnl3(x))
        x = self.drop3(x)
        x = self.flat(x)
        x = relu(self.fc(x))
        x = self.drop(x)
        return softmax(self.out(x))


class GRUModel(tf.keras.Model, ABC):
    def __init__(self, output_bias=None, **kwargs):
        super(GRUModel, self).__init__(**kwargs)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.gru1 = GRU(FLAGS.PARAMS["gru"]['gru1'], return_sequences=True)
        self.gru2 = GRU(FLAGS.PARAMS["gru"]['gru2'], return_sequences=True)
        self.gru3 = GRU(FLAGS.PARAMS["gru"]['gru3'])
        self.fc = Dense(FLAGS.PARAMS["gru"]['dense'])
        self.drop = Dropout(FLAGS.PARAMS["gru"]['drop_rate'])
        self.out = Dense(NCLASSES, bias_initializer=output_bias)

    def call(self, inputs, **kwargs):
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.gru3(x)
        x = relu(self.fc(x))
        x = self.drop(x)
        return softmax(self.out(x))


class CNNGRUModel(tf.keras.Model, ABC):
    """
    input_shape: (bs, width, height, channels)
    """
    cnl1_filters, cnl1_krnl_sz, cnl1_strd = FLAGS.PARAMS["cnn-gru"]["cnl1"]
    drop_rate_1 = FLAGS.PARAMS["cnn-gru"]["drop1"]
    cnl2_filters, cnl2_krnl_sz, cnl2_strd = FLAGS.PARAMS["cnn-gru"]["cnl2"]
    drop_rate_2 = FLAGS.PARAMS["cnn-gru"]["drop2"]
    # cnl3_filters, cnl3_krnl_sz, cnl3_strd = FLAGS.PARAMS["cnn-gru"]["cnl3"]
    # drop_rate_3 = FLAGS.PARAMS["cnn-gru"]["drop3"]
    reshape_size = FLAGS.PARAMS["cnn-gru"]["reshape_size"]
    gru1_hidden = FLAGS.PARAMS["cnn-gru"]['gru1']
    gru2_hidden = FLAGS.PARAMS["cnn-gru"]['gru2']
    # gru3_hidden = FLAGS.PARAMS["cnn-gru"]['gru3']
    dense_hidden = FLAGS.PARAMS["cnn-gru"]["dense"]
    dense_drop_rate = FLAGS.PARAMS["cnn-gru"]["drop_rate"]

    def __init__(self, output_bias=None, **kwargs):
        super(CNNGRUModel, self).__init__(**kwargs)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.cnl1 = Conv2D(self.cnl1_filters, self.cnl1_krnl_sz, self.cnl1_strd)
        self.drop1 = Dropout(self.drop_rate_1)
        self.cnl2 = Conv2D(self.cnl2_filters, self.cnl2_krnl_sz, self.cnl2_strd)
        self.drop2 = Dropout(self.drop_rate_2)
        # self.cnl3 = Conv2D(self.cnl3_filters, self.cnl3_krnl_sz, self.cnl3_strd)
        # self.drop3 = Dropout(self.drop_rate_3)
        self.reshape = Reshape((-1, self.reshape_size))
        self.gru1 = GRU(self.gru1_hidden, return_sequences=True)
        self.gru2 = GRU(self.gru2_hidden)
        # self.gru3 = GRU(self.gru3_hidden)
        self.fc = Dense(self.dense_hidden)
        self.drop = Dropout(self.dense_drop_rate)
        self.out = Dense(FLAGS.NUM_CLASSES, bias_initializer=output_bias)

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, -1)
        x = relu(self.cnl1(x))
        x = self.drop1(x)
        x = relu(self.cnl2(x))
        x = self.drop2(x)
        # x = relu(self.cnl3(x))
        # x = self.drop3(x)
        x = self.reshape(x)  # reshape
        x = self.gru1(x)
        x = self.gru2(x)
        # x = self.gru3(x)
        x = relu(self.fc(x))
        x = self.drop(x)
        return softmax(self.out(x))



def compile_model(model_type=None, lr=1e-4, initial_output_bias=None, checkpoint_folder=None):
    if model_type is None:
        model_type = FLAGS.CHOSEN_MODEL

    if model_type.lower() == "gru":
        model = GRUModel(output_bias=initial_output_bias)
    elif model_type.lower() == "cnn":
        model = CNNModel(output_bias=initial_output_bias)
    elif model_type.lower() == "cnn-gru":
        model = CNNGRUModel(output_bias=initial_output_bias)
    else:
        raise NotImplementedError("model_type must be either 'gru', 'cnn' or 'cnn-gru'")

    optimizer = Adam(learning_rate=lr)
    loss = CategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=FLAGS.METRICS)

    if checkpoint_folder:
        checkpoint_full_path = tf.train.latest_checkpoint(checkpoint_folder)
        model.load_weights(filepath=checkpoint_full_path)
        print(f"Weights restored from checkpoint: {checkpoint_full_path}")

    return model


def print_model_summary(model_type="gru", input_shape=(None, FLAGS.MAX_LEN, 30)):
    model = compile_model(model_type)
    model.build(input_shape)
    print(model.summary())


if __name__ == '__main__':

    # ds_train = random_dataset(NSAMPLES, NCLASSES)
    # ds_test = random_dataset(5, NCLASSES)

    # model = compile_model()

    print_model_summary("cnn-gru")

    # history = model.fit(ds_train, validation_data=ds_test, epochs=2)

    # print(history)
