#!/usr/bin/env python3
from flask import Flask, render_template, url_for, request
from models.utils import load_model, predict
from utils.helper_functions import create_pipeline

import logging


# init flask app instance
app = Flask(__name__)


@app.route("/")
def index():
    print(model.summary())
    return render_template("index.html")


@app.route('/predict_message', methods=["POST"])
def predict_message():
    if request.method == "POST":
        message = request.form["message"]
        try:
            prediction = predict(message, model, pipeline)
            return render_template("index.html", prediction=prediction[0][0])
        except IndexError as e:
            logging.critical(e)
    return render_template("index.html")


if __name__ == '__main__':
    model = load_model("model_weights_conv_drop_false_1_5_new_data.h5",
                       "model_conv_drop_false_1_5_new_data.json")
    pipeline = create_pipeline(key_word_path="data/pickled/key_word_map_new_data_1.pkl")
    app.run(debug=True, port=4000, host="0.0.0.0")
