import os
import numpy as np
from random import randint
import random
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

x_train = []
y_train = []

for _ in range(6242):
    c = randint(0, 1)
    if c == 0:
        img_dir = random.choice(os.listdir("data_cl/wait"))
        x = np.array(cv2.imread("data_cl/wait/" + img_dir, 0)).reshape(32, 32, 1)
        x = np.interp(x, (x.min(), x.max()), (-1, +1))
        y = np.array([1, 0])
    else:
        img_dir = random.choice(os.listdir("data_cl/up"))
        x = np.array(cv2.imread("data_cl/up/" + img_dir, 0)).reshape(32, 32, 1)
        x = np.interp(x, (x.min(), x.max()), (-1, +1))
        y = np.array([0, 1])
    x_train.append(x)
    y_train.append(y)

x_train = np.array(x_train)
y_train = np.array(y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (32, 32, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=512, epochs=10)

model.save('final.h5')
