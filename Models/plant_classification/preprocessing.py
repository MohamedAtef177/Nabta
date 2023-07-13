import numpy as np
import os
import random
import pickle
from cv2 import cv2
from tensorflow.keras import layers, Sequential

training_data = [[], [], [], [], [], [], [], []]


Data_dir = "G:\data"

types = ["Apple", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Potato", "Strawberry"]


for type1 in types:
    path1 = os.path.join(Data_dir, type1)
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
for i in range(0, 8):
    random.shuffle(training_data[i])

MAX = 800
for i in range(8):
    for j in range(0, MAX):
        Train.append(training_data[i][j])
    for j in training_data[i][MAX:]:
        Test.append(j)

print(len(Train))
random.shuffle(Train)
random.shuffle(Test)

for feuture, laple in Train:
    X_train.append(feuture)
    Y_train.append(laple)

for feuture, laple in Test:
    X_test.append(feuture)
    Y_test.append(laple)

X_train = np.array(X_train).reshape(-1, 200, 200, 3)
Y_train = np.array(Y_train)
X_train = X_train / 255

print("1")

X_test = np.array(X_test).reshape(-1, 200, 200, 3)
Y_test = np.array(Y_test)
X_test = X_test / 255

print("2")

pickle_out = open("Y_train.pickle", "wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()

print("3")

pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

print("4")

pickle_out = open("Y_test.pickle", "wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()

print("5")

pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()























'''

import numpy as np
import os
import random
import pickle
from cv2 import cv2
from tensorflow.keras import layers, Sequential

training_data = [[], [], [], []]

aug = Sequential([
    layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    layers.experimental.preprocessing.RandomRotation(factor=.2),
    layers.experimental.preprocessing.RandomZoom(height_factor=.2, width_factor=.2),
    layers.experimental.preprocessing.RandomTranslation(height_factor=.2, width_factor=.2),
])

Data_dir = "D:\data_set\GRADUATION\plantvillage dataset"

categories = ["color", "grayscale", "segmented"]
types = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]

for category in categories:
    path = os.path.join(Data_dir, category)
    for type1 in types:
        path1 = os.path.join(path, type1)
        class_num = types.index(type1)
        for img in os.listdir(path1):
            try:
                img_array = cv2.imread(os.path.join(path1, img))
                new_image = cv2.resize(img_array, (180, 180))
                training_data[class_num].append([new_image, class_num])
            except Exception as e:
                pass

Train = []
Test = []
X_train = []
Y_train = []
X_test = []
Y_test = []

num_of_augmentation = [1, 1, 3]
for k in range(3):
    le = len(training_data[k])
    for i in range(0, le):
        for j in range(num_of_augmentation[k]):
            w = aug(training_data[k][i][0].reshape(-1, 180, 180, 3))
            training_data[k].append([w[0], k])

print(len(training_data[0]))
print(len(training_data[1]))
print(len(training_data[2]))

random.shuffle(training_data[0])
random.shuffle(training_data[1])
random.shuffle(training_data[2])
random.shuffle(training_data[3])

for i in range(4):
    for j in training_data[i][:2750]:                # 70%
        Train.append(j)
    for j in training_data[i][2750:]:
        Test.append(j)


MAX = 1500
for i in range(4):
    if i == 3:
        MAX = 2500
    else:
        MAX = 1500
    for j in range(0, MAX):
        Train.append(training_data[i][j])
    for j in training_data[i][MAX:]:
        Test.append(j)

#print(len(Train))
random.shuffle(Train)
random.shuffle(Test)

for feuture, laple in Train:
    X_train.append(feuture)
    Y_train.append(laple)

for feuture, laple in Test:
    X_test.append(feuture)
    Y_test.append(laple)

X_train = np.array(X_train).reshape(-1, 180, 180, 3)
Y_train = np.array(Y_train)
X_train = X_train / 255

print("1")

X_test = np.array(X_test).reshape(-1, 180, 180, 3)
Y_test = np.array(Y_test)
X_test = X_test / 255

print("2")

pickle_out = open("Y_train_apple_1.pickle", "wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()

print("3")

pickle_out = open("X_test_apple_1.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

print("4")

pickle_out = open("Y_test_apple_1.pickle", "wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()

print("5")

pickle_out = open("X_train_apple_1.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
'''