# Flask API
# Created: 24th May, 2020
"""

@author: Vivek Ghosh
"""
import pickle
from flask import Flask, request
import pandas as pd
import numpy as np

app = Flask(__name__)

pickle_in = open("best_classifier.pkl", 'rb')
classifier = pickle.load(pickle_in)


@app.route("/")
def welcome():
    return "welcome all"


@app.route("/predict")
def predict_note_authentication():

    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")

    prediction = classifier.predict([[
        variance, skewness, curtosis, entropy
    ]])

    return 'The Predicted value is {}'.format(str(prediction))


@app.route("/predict_file", methods=['POST'])
def predict_note_authentication_for_file():

    df = pd.read_csv(request.files.get("file"))
    y_pred = classifier.predict(df)

    return "The predicted values for csv file is: "+str(list(y_pred))


if __name__ == '__main__':
    app.run()
