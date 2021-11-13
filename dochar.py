import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pathlib
import scipy.io
import torch


import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
train = pd.read_csv('B:\data\csv\emnist-balanced-train.csv', header=None)
test = pd.read_csv('B:\data\csv\emnist-balanced-test.csv', header=None)
train.head()
train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]
train_labels = pd.get_dummies(train_labels)
test_labels = pd.get_dummies(test_labels)
train_labels.head()
train_data = train_data.values
train_labels = train_labels.values
test_data = test_data.values
test_labels = test_labels.values
del train, test

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])
train_dataI = np.apply_along_axis(rotate, 1, train_data)
test_dataI = np.apply_along_axis(rotate, 1, test_data)
plt.imshow(test_data[2].reshape([28, 28]), cmap='Greys_r')
#plt.show()
num_classes = 47
img_size = 28


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

#bogus matlab stuff
#xs = tf.placeholder(tf.float32, [None, 784])
#ys = tf.placeholder(tf.float32, [None, 47])
#keep_prob = tf.placeholder(tf.float32)
img_size = 28
num_classes = 47
tenTrain = torch.from_numpy(train_data)
print(tenTrain.shape)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=12, kernel_size=(5,5), strides=2, activation='relu', input_shape=(img_size,img_size,1)))
model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Conv2D(filters=18, kernel_size=(3,3) , strides=2, activation='relu'))
model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Conv2D(filters=24, kernel_size=(2,2), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=150, activation='relu'))
model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())
print("t")
X, y = img_label_load('B:\data\csv\emnist-balanced-train.csv')
testX, testY = img_label_load('B:\data\csv\emnist-balanced-test.csv')
print(X.shape)
print(tenTrain.shape)
print((torch.from_numpy(test_data)).shape)
print("aftershape")


#training_data_generator = data_generator.flow(X, y, subset='training')
#validation_data_generator = data_generator.flow(X, y, subset='validation')
#history = model.fit_generator(training_data_generator,
 #                             steps_per_epoch=500, epochs=5, # can change epochs to 10
 #                             validation_data=validation_data_generator)
dataset = tf.data.Dataset.from_tensor_slices(X)
data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=.2)
test_data_generator = data_generator.flow(X, y)
training_data_generator = data_generator.flow(X, y, subset='training')
validation_data_generator = data_generator.flow(X, y, subset='validation')
history = model.fit(training_data_generator, epochs=5, verbose=2, steps_per_epoch=500)
model.evaluate(test_data_generator)
test_data = pd.read_csv('B:\data\csv\emnist-balanced-test.csv', header=None)
X_test, y_test = img_label_load('B:\data\csv\emnist-balanced-test.csv')
model.save('model.h5')
print("saved")
def show_img(data, row_num):
    img_flip = np.transpose(data.values[row_num,1:].reshape(28, 28), axes=[1,0]) # img_size * img_size arrays
    plt.title('Class: ' + str(data.values[row_num,0]) + ', Label: ' + str(class_mapping[data.values[row_num,0]]))
    plt.imshow(img_flip, cmap='Greys_r')

def run_prediction(idx):
    result = np.argmax(model.predict(X_test[idx:idx+1]))
    print('Prediction: ', result ,' Char: ', class_mapping[result])
    print('Label: ', test_data.values[idx,0])
    show_img(test_data, idx)

#for i in range(20):
#    run_prediction(i)
#    plt.imshow(test_dataI[i].reshape([28, 28]), cmap='Greys_r')
 #   plt.show()
#result = np.argmax(model.predict(test_data[12].reshape[28,28]))

print("end")
#for images, labels in train_data.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")


