import enum
import os
import numpy as np
from cv2 import cv2
from tensorflow.keras import models

from graduation_project.settings import MEDIA_ROOT


ML_MODELS_PATH = "ml_models\\models\\"

RESIZE_DIR = os.path.join(MEDIA_ROOT, 'resized/')


PLANTS_TYPES_NAMES = [['Apple', 4], ['Charry', 1], ['Corn', 1], ['Grape', 3], ['Peach', 1], ['Pepper', 5], ['Potato', 2], ['Strawberry', 1]]


DISEASES_TYPES_NAMES = [
    ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy'], # Apple
    ['Healthy', 'Powdery Mildew'], # Charry
    ['Gray Leaf Spot', 'Common Rust', 'Healthy', 'Northern Leaf Blight'], # Corn
    ['Black Rot', 'Esca (Black Measles)', 'Healthy', 'Leaf blight (Isariopsis Leaf Spot)'], # Grape
    ['Bacterial Spot', 'Healthy'], # Peach
    ['Bacterial Spot', 'Healthy'], # Pepper
    ['Early Blight', 'Healthy', 'Late Blight'], # Potato
    ['Healthy', 'Leaf Scorch'], # Strawberry
]

def __resize_image(image):
    img = cv2.imread(image)
    img = np.array(cv2.resize(img, (200, 200)))
    img = (img.reshape(-1, 200, 200, 3)) / 255
    return img


def __identify_plant(img):
    model = models.load_model(ML_MODELS_PATH + "identify_plants_inception_v3")
    return np.argmax(model.predict(img))


def identify_plant(image):
    return __identify_plant(__resize_image(image))


def identify_disease(image):
    img = __resize_image(image)
    plant_type = __identify_plant(img)

    if plant_type == 0:
        return [plant_type, __identify_apple_disease(img)]

    if plant_type == 1:
        return [plant_type, __identify_charry_disease(img)]

    if plant_type == 2:
        return [plant_type, __identify_corn_disease(img)]

    if plant_type == 3:
        return [plant_type, __identify_grape_disease(img)]
    
    if plant_type == 4:
        return [plant_type, __identify_peach_disease(img)]
    
    if plant_type == 5:
        return [plant_type, __identify_papper_disease(img)]
    
    if plant_type == 6:
        return [plant_type, __identify_potato_disease(img)]
    
    return [plant_type, __identify_strawberry_disease(img)]


def identify_ripeness(image):
    return __identify_plant(__resize_image(image))


def __identify_apple_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_apple_inception_v3")
    return np.argmax(model.predict(image))


def __identify_charry_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_cherry_inception_v3")
    return np.argmax(model.predict(image))


def __identify_corn_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_corn_inception_v3")
    return np.argmax(model.predict(image))


def __identify_grape_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_grape_inception_v3")
    return np.argmax(model.predict(image))


def __identify_peach_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_peach_inception_v3")
    return np.argmax(model.predict(image))


def __identify_papper_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_pepper_inception_v3")
    return np.argmax(model.predict(image))


def __identify_potato_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_potato_inception_v3")
    return np.argmax(model.predict(image))


def __identify_strawberry_disease(image):
    model = models.load_model(ML_MODELS_PATH + "diseases_strawberry_inception_v3")
    return np.argmax(model.predict(image))
