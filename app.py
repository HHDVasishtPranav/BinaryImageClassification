from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load the pre-trained model
model = load_model('model_car_vs_truck.h5')

app = Flask(__name__)

# Function to predict the class for a given image
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(128, 128), interpolation='bicubic')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        class_label = 'Truck'
    else:
        class_label = 'Car'

    return class_label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image from the request
        file = request.files['file']

        # Save the uploaded image temporarily
        if file:
            filename = file.filename
            file_path = os.path.join('static', filename)
            file.save(file_path)

            # Predict the class for the uploaded image
            class_label = predict_image_class(file_path)

            # Render the result on the template along with the image
            return render_template('result.html', filename=filename, class_label=class_label)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
