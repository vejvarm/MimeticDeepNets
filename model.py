from abc import ABC

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, GRU, Conv2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.data import Dataset

from flags import FLAGS

NSAMPLES = 20
NCLASSES = FLAGS.NUM_CLASSES


class CNNModel(tf.keras.Model, ABC):
    """
    input_shape: (bs, width, height, channels)
    """
    cnl1_filters, cnl1_krnl_sz, cnl1_strd = FLAGS.CNNPARAMS["cnl1"]
    cnl2_filters, cnl2_krnl_sz, cnl2_strd = FLAGS.CNNPARAMS["cnl2"]
    cnl3_filters, cnl3_krnl_sz, cnl3_strd = FLAGS.CNNPARAMS["cnl3"]
    dense_hidden = FLAGS.CNNPARAMS["dense"]
    dense_drop_rate = FLAGS.CNNPARAMS["drop_rate"]

    def __init__(self, output_bias=None, **kwargs):
        super(CNNModel, self).__init__(**kwargs)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.cnl1 = Conv2D(self.cnl1_filters, self.cnl1_krnl_sz, self.cnl1_strd)
        self.cnl2 = Conv2D(self.cnl2_filters, self.cnl2_krnl_sz, self.cnl2_strd)
        self.cnl3 = Conv2D(self.cnl3_filters, self.cnl3_krnl_sz, self.cnl3_strd)
        self.flat = Flatten()
        self.fc = Dense(self.dense_hidden)
        self.drop = Dropout(self.dense_drop_rate)
        self.out = Dense(FLAGS.NUM_CLASSES, bias_initializer=output_bias)

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, -1)
        x = relu(self.cnl1(x))
        x = relu(self.cnl2(x))
        x = relu(self.cnl3(x))
        x = self.flat(x)
        x = relu(self.fc(x))
        x = self.drop(x)
        return softmax(self.out(x))


class GRUModel(tf.keras.Model, ABC):
    def __init__(self, output_bias=None, **kwargs):
        super(GRUModel, self).__init__(**kwargs)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.gru1 = GRU(FLAGS.GRUPARAMS['gru1'], return_sequences=True)
        self.gru2 = GRU(FLAGS.GRUPARAMS['gru2'])
        self.fc = Dense(FLAGS.GRUPARAMS['dense'])
        self.drop = Dropout(FLAGS.GRUPARAMS['drop_rate'])
        self.out = Dense(NCLASSES, bias_initializer=output_bias)

    def call(self, inputs, **kwargs):
        x = self.gru1(inputs)
        x = self.gru2(x)
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
    else:
        raise NotImplementedError("model_type must be either 'gru' or 'cnn'")

    optimizer = Adam(learning_rate=lr)
    loss = CategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=FLAGS.METRICS)

    if checkpoint_folder:
        checkpoint_full_path = tf.train.latest_checkpoint(checkpoint_folder)
        model.load_weights(filepath=checkpoint_full_path)
        print(f"Weights restored from checkpoint: {checkpoint_full_path}")

    return model


def random_dataset(nsamples, nclasses, batch_size=2):
    x = np.random.uniform(size=(nsamples, 600, 30))
    y = np.random.choice(np.arange(0, nclasses), size=(nsamples, ))
    y_one_hot = tf.one_hot(y, nclasses)

    data = Dataset.from_tensor_slices((x, y_one_hot))
    data = data.shuffle(nsamples).batch(batch_size)
    return data


if __name__ == '__main__':

    ds_train = random_dataset(NSAMPLES, NCLASSES)
    ds_test = random_dataset(5, NCLASSES)

    model = compile_model()

    history = model.fit(ds_train, validation_data=ds_test, epochs=2)

    print(history)
