import tensorflow as tf
import numpy as np
import keras.backend as K
import math
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from keras.regularizers import l2
import logging as log


class SiameseNN:
    def __init__(self, learning_rate=1e-2, batch_size=128, l2_regularization=1e-2, name=''):
        log.info(f"Run {name} started. Params: learning_rate-{learning_rate}, batch_size-{batch_size}, l2_regularization-{l2_regularization}")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.input_shape = (250, 250, 1)
        initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1e-2)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.99)

        self.conv_net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(10, 10), activation='relu', input_shape=self.input_shape,
                                   kernel_regularizer=l2(1e-2), kernel_initializer=initializer,
                                   name='Conv1'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(8, 8), activation='relu',
                                   kernel_regularizer=l2(1e-2), kernel_initializer=initializer,
                                   name='Conv2'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
                                    kernel_regularizer=l2(1e-2), kernel_initializer=initializer,
                                   name='Conv3'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), activation='relu',
                                   kernel_regularizer=l2(1e-2), kernel_initializer=initializer,
                                   name='Conv4'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), activation='relu',
                                   kernel_regularizer=l2(1e-2), kernel_initializer=initializer,
                                   name='Conv5'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4096, activation='sigmoid',
                                  kernel_regularizer=l2(1e-4), kernel_initializer=initializer,
                                  name='Dense1'),
        ])

        img1_inp = tf.keras.layers.Input(self.input_shape)
        img2_inp = tf.keras.layers.Input(self.input_shape)

        embedded_img1 = self.conv_net(img1_inp)
        embedded_img2 = self.conv_net(img2_inp)

        l1_layer = tf.keras.layers.Lambda(lambda eis: K.abs(eis[0] - eis[1]))
        l1_dist = l1_layer([embedded_img1, embedded_img2])

        prediction = tf.keras.layers.Dense(1, activation='sigmoid',
                                           name='Prediction')(l1_dist)

        self.model = tf.keras.models.Model(inputs=[img1_inp, img2_inp], outputs=[prediction])

        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                           metrics=['accuracy'])

    def fit(self, X, y, epochs=5):
        log.info(f"Fit start - epochs:{epochs}")
        training_history = self.model.fit(X, y, batch_size=self.batch_size, epochs=epochs, validation_split=0.2,
                                          verbose=1,
                                          callbacks=[TensorBoard(log_dir=f"logs/tb/{str(datetime.now()).replace(' ', '_')}",
                                                                 update_freq='batch')])
        return training_history

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred = np.round(y_pred)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y == y_pred).mean()

    def iter_batches(self, X, y, batch_size):
        """
        X - Matrix of samples and features. Shape - (samples_dimension, number_of_samples).
        Y - Labeling of samples. Shape - (number_of_classes, number_of_samples).
        batch_size - size of wanted batch.
        idx - number of batch wanted.
        returns -
            X_curr - current batch of samples.
            Y_curr - current batch labels.
        """
        m = X.shape[0]
        for j in range(math.ceil(m / batch_size)):
            l_idx = j * batch_size
            u_idx = (j + 1) * batch_size
            if u_idx > m:
                u_idx = m
            yield X[l_idx:u_idx], y[l_idx:u_idx]

