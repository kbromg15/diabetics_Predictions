'''
This is a Flask web application that reads a pickle file to load a machine learning model,
accepts user input from a web form, and returns a prediction using the model.

'''

# https://flask.palletsprojects.com/en/2.2.x/quickstart/
from flask import Flask, render_template, request
import pickle
import numpy as np

# create flask app
app = Flask(__name__)

# Load the pickle model file
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except:
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        error_message = "Sorry, an error occurred while loading the model. Please try again later."
        return render_template('index.html', result=error_message)

    # Get the features from the request form and make a prediction
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    result = round(prediction[0], 1)

    # Create a result string to show to the user
    result_string = f"The predicted diabetes outcome is {result}."
    return render_template('index.html', result=result_string)

if __name__ == "__main__":
    app.run(debug=True)
