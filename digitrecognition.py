import cv2
import numpy as np
# from keras.datasets import mnist
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import random
import pickle

def get_best_shift(image):
    cy, cx = ndimage.measurements.center_of_mass(image)  # returns coordinates of centre of image

    rows, cols = image.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def shift_according_to_centre_of_mass(img):
    img = cv2.bitwise_not(img)

    # centralize the image according to centre of mass
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img


# creating the training data
data_dir = "myData"
CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

# read training data
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (28, 28))
            new_array = shift_according_to_centre_of_mass(new_array)
            training_data.append([new_array, class_num])


create_training_data()

# mix data up
random.shuffle(training_data)


# split 80-20
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(len(training_data) * 8 // 10):
    X_train.append(training_data[i][0])
    y_train.append(training_data[i][1])
for i in range(len(training_data) * 8 // 10, len(training_data)):
    X_test.append(training_data[i][0])
    y_test.append(training_data[i][1])

X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the data
X_train /= 255
X_test /= 255




#image data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.3,
                             shear_range=0.2,
                             rotation_range=20)
dataGen.fit(X_train)


# displaying dataset using the matploltib
'''plt.imshow(X_train[0], cmap = "gray")
plt.show()
print(y_train[0])'''

# step 0: data preprocessing
# checking the shape involved in dataset

# after watching the shapes we are going to reshape
# last no 0 signifies image is in grayscale
# one hot encoding of dependent variable
'''For example, if the image is of the number 6, 
then the label instead of being = 6,
 it will have a value 0 in column 7 and 0 
 in rest of the columns, like [0,0,0,0,0,0,0,0,0].'''

y_train = to_categorical(y_train, 9)
y_test = to_categorical(y_test, 9)

# building the model
model = Sequential()
layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
layer_2 = Conv2D(64, kernel_size=3, activation='relu')
layer_3 = MaxPooling2D(pool_size=(2, 2))
layer_9 = Conv2D(64, kernel_size=3, activation='relu')
layer_10 = MaxPooling2D(pool_size=(2, 2))
layer_4 = Dropout(0.25)
layer_5 = Flatten()  # Flatten serves as a connection between convolutional and dense layers.
layer_6 = Dense(128, activation='relu')
layer_7 = Dropout(0.5)
layer_8 = Dense(9, activation='softmax')

# adding the layers to the model
model.add(layer_1)  # the first layer takes an input of shape being (28, 28, 0)
model.add(layer_2)  #
model.add(layer_3)
model.add(layer_9)
model.add(layer_10)
model.add(layer_4)
model.add(layer_5)
model.add(layer_6)
model.add(layer_7)
model.add(layer_8)
# compiling the model
'''Optimizer — It controls the learning rate. We will be using ‘adam’ optimizer. It is a very good optimizer as it utilises the perks of both Stochastic gradient and RMSprop optimizers.
Loss function — We will be using ‘categorical_crossentropy’ loss function. It is the most common choice for classification. A lower score corresponds to better performance.
Metrics — To make things easier to interpret, we will be using ‘accuracy’ metrix to see the accuracy score on the validation set while training the model.'''

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the model
model.fit(dataGen.flow(X_train, y_train, batch_size=128), epochs=70, validation_data=(X_test, y_test), shuffle=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy : %.2f%%" % (scores[1]*100))
pickle_out = open("model_trained.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()
