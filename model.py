import numpy as np 
import pandas as pd
from tensorflow import keras
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set the paths for training and validation data
train_data_dir = "train"
valid_data_dir = "valid"

# Initialize CNN
model = Sequential()

# Convolutional layers
model.add(Convolution2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# Dense layers
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and preprocess the data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Training
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)

# Evaluation
test_loss, test_accuracy = model.evaluate(valid_generator, steps=len(valid_generator))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('model_car_vs_truck.h5')

# Load the trained model
from keras.models import load_model
model = load_model('model_car_vs_truck.h5')

# Function to predict and display the class label for a given image path
from keras.preprocessing import image
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
predict_image_class('train/Truck/00013.jpeg')
