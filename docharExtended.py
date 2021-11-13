from __future__ import print_function
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.misc import *
from os import listdir
from os.path import isfile, join
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


path = "B:\\data\\curated.tar\\curated\\"


character_curated = [ord(c) for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt()*+,-0123456789:<=>"_[]' ]
#character_curated = [ord(c) for c in '!"#$%&' + "'" + '()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_' + "'" +'abcdefghijklmnopqrstuvwxyz{|}~' ]
def img_label_load(data_path, num_classes=None):
    data = pd.read_csv(data_path, header=None)
    data_rows = len(data)
    if not num_classes:
        num_classes = len(data[0].unique())
    img_size = int(np.sqrt(len(data.iloc[0][1:])))
    imgs = np.transpose(data.values[:, 1:].reshape(data_rows, img_size, img_size, 1),
                        axes=[0, 2, 1, 3])
    labels = keras.utils.to_categorical(data.values[:, 0], len(character_curated))  # one-hot encoding vectors
    return imgs / 255., labels
eX, ey = img_label_load('B:\data\csv\emnist-balanced-train.csv')
etestX, etestY = img_label_load('B:\data\csv\emnist-balanced-test.csv')

print([chr(i) for i in character_curated])
img_size = 28
X = []
y = []
for i in character_curated:
    path_img = path + str(i) + '/'
    for file_name in [f for f in listdir(path_img) if isfile(join(path_img, f))]:
        img = cv2.imread(path_img + file_name, 0)
        img = cv2.resize(img,(img_size, img_size), interpolation = cv2.INTER_AREA)
        X += [img/255.]
        t = []
        for j in character_curated:
            if j == i:
                t.append(1)
            else:
                t.append(0)
        y += [t]

X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float32)
np.append(X,eX)
np.append(y,ey)

'''Definition: Create train test data to model it
    - shuffle
    - separate train test
    - encode target to dummy vars
Parameters:
    X: images[n_images, x_size, y_size]
    y: characters[n_images, 1]
    char_select: dictionary of characters selected
    augmentation_fuction

usage:
    prepare_data_char_subset(X, y, set(['1','0']), augmentation_fuction = lambda: augmentate01(param1, param2=p2) )

return:
    X_train, y_train, X_test, y_test, labels_dictionary
'''
def prepare_data_char_subset(X, y):

    X_select = X
    y_select = y
    #y_select = np.reshape(y_select[0],len(character_curated))
    # Shuffle
    X_select, y_select = shuffle(X_select, y_select, random_state=1)

    # Separate train test
    X_train, X_test, y_train, y_test = train_test_split(X_select, y_select, test_size=0.20, stratify=y)
    print(X_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0],  X_train.shape[1], X_train.shape[2],1))
    X_test = np.reshape(X_test, (X_test.shape[0],  X_test.shape[1], X_test.shape[2],1))

    print('Train shape: ', X_train.shape, y_train.shape)
    print('Test shape: ', X_test.shape, y_test.shape)
    #print('Num classes: ', len(set(y_train)))
    #print('Classes: ', set(y_train))

    return X_train, y_train, X_test, y_test

X, y, X_test, y_test = prepare_data_char_subset(X, y)
f = plt.figure()
a = f.add_subplot(1,4,2)
max = 0
print(X.shape)
print(y.shape)
print("afterprepared")
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=12, kernel_size=(5,5), strides=2, activation='relu', input_shape=(img_size,img_size,1)))
model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Conv2D(filters=18, kernel_size=(3,3) , strides=2, activation='relu'))
model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Conv2D(filters=24, kernel_size=(2,2), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=150, activation='relu'))
model.add(keras.layers.Dense(units=len(character_curated), activation='softmax')) #this is probably current error units =

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

dataset = tf.data.Dataset.from_tensor_slices(X)
data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=.2)
test_data_generator = data_generator.flow(X, y)
training_data_generator = data_generator.flow(X, y, subset='training')
validation_data_generator = data_generator.flow(X_test, y_test, subset='validation')
history = model.fit(training_data_generator, epochs=500, verbose=2, steps_per_epoch=500)
model.evaluate(test_data_generator)