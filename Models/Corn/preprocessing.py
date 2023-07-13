import numpy as np
import os
import random
import pickle
from cv2 import cv2
from tensorflow.keras import layers, Sequential

training_data = [[], [], [], []]
'''
aug = Sequential([
    layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    layers.experimental.preprocessing.RandomRotation(factor=.2),
    layers.experimental.preprocessing.RandomZoom(height_factor=.2, width_factor=.2),
    layers.experimental.preprocessing.RandomTranslation(height_factor=.2, width_factor=.2),
])
'''
Data_dir = "D:\data_set\GRADUATION\plantvillage dataset"

categories = ["color", "grayscale", "segmented"]
types = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
         "Corn_(maize)___Northern_Leaf_Blight"]

for category in categories:
    path = os.path.join(Data_dir, category)
    for type1 in types:
        path1 = os.path.join(path, type1)
        class_num = types.index(type1)
        for img in os.listdir(path1):
            try:
                img_array = cv2.imread(os.path.join(path1, img))
                new_image = cv2.resize(img_array, (200, 200))
                training_data[class_num].append([new_image, class_num])
            except Exception as e:
                pass
Train = []
Test = []
X_train = []
Y_train = []
X_test = []
Y_test = []

print("1")
'''
print(len(training_data[0]))

le = len(training_data[0])
for i in range(0, le):
    for j in range(1):
        w = aug(training_data[0][i][0].reshape(-1, 200, 200, 3))
        training_data[0].append([w[0], 0])
print("2")
'''
random.shuffle(training_data[0])
random.shuffle(training_data[1])
random.shuffle(training_data[2])
random.shuffle(training_data[3])
print("3")

MAX = 1224
for i in range(4):
    for j in range(0, MAX):
        Train.append(training_data[i][j])
    for j in range(MAX, 1530):
        Test.append(training_data[i][j])
print("4")

random.shuffle(Train)
random.shuffle(Test)

for feuture, laple in Train:
    X_train.append(feuture)
    Y_train.append(laple)

for feuture, laple in Test:
    X_test.append(feuture)
    Y_test.append(laple)
print("5")

X_train = np.array(X_train).reshape(-1, 200, 200, 3)
Y_train = np.array(Y_train)
X_train = X_train / 255
X_test = np.array(X_test).reshape(-1, 200, 200, 3)
Y_test = np.array(Y_test)
X_test = X_test/255

pickle_out = open("X_train_corn.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
print("6")

pickle_out = open("Y_train_corn.pickle", "wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()
print("7")

pickle_out = open("X_test_corn.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
print("8")

pickle_out = open("Y_test_corn.pickle", "wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()
print("9")
