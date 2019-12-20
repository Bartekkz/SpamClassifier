#!/usr/bin/env python3
from flask import Flask, render_template, url_for, request
from models.utils import load_model, predict
from utils.helper_functions import create_pipeline

import json


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
        except AttributeError as e:
            return json.dumps(e)
    return render_template("index.html")


if __name__ == '__main__':
    model = load_model("./data/models_data/model_conv_drop_false_1_5_new_data.json",
                       "./data/models_data/model_weights_conv_drop_false_1_5_new_data.h5")
    pipeline = create_pipeline(key_word_path="data/key_word_map_new_data.pkl")
    app.run(debug=True, port=4000)
