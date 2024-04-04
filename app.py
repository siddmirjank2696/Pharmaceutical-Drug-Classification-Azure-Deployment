# Importing the required libraries
import numpy as np
import pandas as pd
import pickle
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
from flask import Flask, request, app, render_template, send_file


# Creating a Flask application
app = Flask(__name__)

# Telling Flask where to find static files (e.g., images)
app.static_folder = 'static'

# Loading the model
model = load_model('drug_classification_model.h5')

# Loading the label classes
label_classes = pickle.load(open("label_classes.pkl", "rb"))


# Creating a decorator to direct to the home page
@app.route("/")
def home():
    # Returning the home page
    return render_template("home.html")


# Creating a decorator to direct to the prediction page
@app.route("/predict", methods=["POST"])
def predict():
    # Retrieving the data from an HTML form
    img_file = request.files['image']

     # Saving the uploaded image to the static folder
    img_filename = f"static/{img_file.filename}"
    img_file.save(img_filename)

    # Reading the image
    img = cv2.imread(img_filename)

    # Resizing the image to make it compatible with the model
    img = cv2.resize(img, (224, 224))

    # Reshaping the image to include the batch size as a single image
    image = np.reshape(img, (1, 224, 224, 3))

    # Predicting the labels of the test_images
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)

    # Mapping the label
    prediction = label_classes[pred[0]]

    # Pass image filename and prediction to the HTML template
    return render_template("home.html", image_filename=img_filename, prediction_text="The Drug Is : {}".format(prediction))


# Creating a main function
if __name__ == "__main__":
    # Allowing debugging of the app and hosting it on my local host at port 4002
    app.run(debug=True, host='0.0.0.0', port=4002)