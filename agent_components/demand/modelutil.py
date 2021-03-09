import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, Embedding, RNN, GRUCell, LSTMCell, SimpleRNNCell, StackedRNNCells

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('WARNING')
# tf.compat.v1.disable_eager_execution()
y_vars = ["UsePower-{}".format(i) for i in range(7*24)]

def file2dataset_pipeline(filepath):
    df = pd.read_csv(filepath)
    X = np.array(df.drop(columns=y_vars))
    y = np.array(df[y_vars])
    return X, y

class DNN_Structure(Model):
    def __init__(self, XX_train, yy_train):
        super(DNN_Structure, self).__init__()
        self.d1 = Dense(256, activation='relu', input_shape=(XX_train.shape[1], ))
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(1024, activation='relu')
        self.d4 = Dense(512, activation='relu')
        self.out = Dense(yy_train.shape[1], activation='linear')

    def call(self, x):
        intermediates = []
        for layer in [self.d1, self.d2, self.d3, self.d4, self.out]:
            x = layer(x)
            intermediates.append(x)
        return intermediates, x


# def init_train_test_step():
#     @tf.function
#     def train_step(XX_train, yy_train, model, loss_object, optimizer, train_loss, seed):
#         # Seed
#         tf.random.set_seed(seed)
#         np.random.seed(seed)
#
#         with tf.GradientTape() as tape:
#             _, predictions = model(XX_train)
#             loss = loss_object(yy_train, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#         train_loss(yy_train, predictions)
#
#     @tf.function
#     def test_step(XX_test, yy_test, model, loss_object, test_loss, seed):
#         # Seed
#         tf.random.set_seed(seed)
#         np.random.seed(seed)
#
#         _, predictions = model(XX_test)
#         t_loss = loss_object(yy_test, predictions)
#
#         test_loss(yy_test, predictions)
#
#     return train_step, test_step
#
#
#
#
# def train_model(model, XX_train, yy_train,test_step, epochs, seed, optimizer, lr, momentum=0, verbose=True):
#
#     # Data
#     train_ds = tf.data.Dataset.from_tensor_slices((XX_train, yy_train)).batch(1)
#
#     # Metrics
#     train_loss = tf.keras.metrics.MeanAbsoluteError(name='train_loss')
#     test_loss = tf.keras.metrics.MeanAbsoluteError(name='test_loss')
#
#     # Loss and Optmizer
#     loss = tf.keras.losses.MeanAbsoluteError()
#     if momentum:
#         optimizer = optimizer(learning_rate=lr,momentum=momentum)
#     else:
#         optimizer = optimizer(learning_rate=lr)
#
#     for epoch in range(epochs):
#         train_loss.reset_states()
#         test_loss.reset_states()
#
#         for _XX_train, _yy_train in train_ds:
#             test_step(_XX_train, _yy_train, model, loss, test_loss, seed)
#     return model


def load_model_init(model_ckpt_file, data_csv_file):

    # _, test_step = init_train_test_step()
    # EPOCHS = 1
    # LEARNING_RATE = 0.001
    # OPTIMIZER = tf.keras.optimizers.SGD
    # SEED = 42

    # MODEL_CKPT_FILE = 'models/test.ckpt'
    X, y = file2dataset_pipeline(data_csv_file)
    model = DNN_Structure(X, y)
    # model = train_model(model, X, y, test_step, EPOCHS, SEED, OPTIMIZER, LEARNING_RATE)
    model.load_weights(model_ckpt_file)
    preds = model.predict(X)[1][0]
    # print(preds)
    return model


# if __name__ == '__main__':
#     model = load_model_init("models/test.ckpt", "test.csv")

