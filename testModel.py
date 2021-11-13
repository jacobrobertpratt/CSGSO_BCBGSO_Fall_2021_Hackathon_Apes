
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras

new_model = tf.keras.models.load_model('model.h5')
new_model.summary()
def img_label_load(data_path, num_classes=None):
    data = pd.read_csv(data_path, header=None)
    data_rows = len(data)
    if not num_classes:
        num_classes = len(data[0].unique())

    # this assumes square imgs. Should be 28x28
    img_size = int(np.sqrt(len(data.iloc[0][1:])))

    # Images need to be transposed. This line also does the reshaping needed.
    imgs = np.transpose(data.values[:, 1:].reshape(data_rows, img_size, img_size, 1),
                        axes=[0, 2, 1, 3])  # img_size * img_size arrays

    labels = keras.utils.to_categorical(data.values[:, 0], num_classes)  # one-hot encoding vectors

    return imgs / 255., labels
X_test, y_test = img_label_load('B:\data\csv\emnist-balanced-test.csv')
loss, acc = new_model.evaluate(X_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
