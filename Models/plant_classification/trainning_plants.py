from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
import pickle

x = pickle.load(open("X_train.pickle", "rb"))
y = pickle.load(open("Y_train.pickle", "rb"))

pre_trained_model = InceptionV3(input_shape=[200, 200] + [3], include_top=False, weights="imagenet")

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

flat1 = layers.Flatten()(last_output)

class1 = Dense(128, activation='relu')(flat1)

class2 = Dense(64, activation='relu')(class1)

class3 = Dense(32, activation='relu')(class2)

class4 = Dense(16, activation='relu')(class3)

output = Dense(8, activation='softmax')(class4)

model = Model(pre_trained_model.input, output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, batch_size=64, epochs=12, validation_batch_size=.1)

model.save("plants_classification_inception_V3_2")

