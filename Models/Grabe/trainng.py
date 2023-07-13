from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np

x = pickle.load(open("X_train_grape.pickle", "rb"))
y = pickle.load(open("Y_train_grape.pickle", "rb"))
print(len(x))
pre_trained_model = InceptionV3(input_shape=[200, 200] + [3], include_top=False, weights="imagenet")

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

flat1 = layers.Flatten()(last_output)

class1 = Dense(128, activation='relu')(flat1)

#class6 = Dropout(.1)(class1)

class2 = Dense(64, activation='relu')(class1)

#class7 = Dropout(.1)(class2)

class3 = Dense(32, activation='relu')(class2)

#class8 = Dropout(.1)(class3)

class4 = Dense(16, activation='relu')(class3)
class5 = Dense(8, activation='relu')(class4)
output = Dense(4, activation='softmax')(class5)

model = Model(pre_trained_model.input, output)
'''
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


'''
for layer in vgg.layers:
    layer.trainable = False

flat1 = Flatten()(vgg.layers[-1].output)
class1 = Dense(128, activation='relu')(flat1)
class2 = Dense(64, activation='relu')(class1)
class3 = Dense(32, activation='relu')(class2)
class4 = Dense(16, activation='relu')(class3)
class5 = Dense(8, activation='relu')(class4)
output = Dense(4, activation='softmax')(class5)

model = Model(inputs=vgg.inputs, outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
'''
'''
dataAugmentaion = ImageDataGenerator(rotation_range = 30, zoom_range = 0.20,
fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True,
width_shift_range=0.1, height_shift_range=0.1)

history = model.fit_generator((x, y, batch_size = 32),

                              verbose=2)
'''
'''
aug = Sequential([
    layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    layers.experimental.preprocessing.RandomRotation(factor=.2),
    layers.experimental.preprocessing.RandomZoom(height_factor=.2, width_factor=.2),
    layers.experimental.preprocessing.RandomTranslation(height_factor=.2, width_factor=.2),
])

newx = []
newy = []
for i in range(0, len(x)):
    w = aug(x[i].reshape(-1, 180, 180, 3))
    newx.append(x[i])
    newx.append(w[0])
    newy.append(y[i])
    newy.append(y[i])

train_data = []
for i in range(0, len(newx)):
    train_data.append([newx[i], newy[i]])

random.shuffle(train_data)

newx = []
newy = []

for i in train_data:
    newx.append(i[0])
    newy.append(i[1])


x = np.array(newx).reshape(-1, 180, 180, 3)
y = np.array(newy)
'''
model.fit(x, y, batch_size=64, epochs=12, validation_batch_size=.1)

model.save("grape_diseases_inception_V3_1")


