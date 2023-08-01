import numpy as np 
import pandas as pd
from tensorflow import keras
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image


from keras.models import load_model
model = load_model('model_car_vs_truck.h5')

# Function to predict and display the class label for a given image path
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(128, 128), interpolation='bicubic')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        class_label = 'truck'
    else:
        class_label = 'car'

    plt.imshow(img)
    plt.title("Predicted Class: " + class_label)
    plt.axis('off')
    plt.show()

# Example usage: Provide the path of the image you want to predict
predict_image_class('train/Car/00023.jpeg')