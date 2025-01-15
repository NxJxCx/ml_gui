import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS

from machine_learning.ml_util import (
    get_all_algo_names,
    get_ml_instance,
    get_trained_history_results,
    remove_ml_instance,
    set_ml_instance,
    set_session_id,
)
from machine_learning.util import generate_session_id

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") if "SECRET_KEY" in os.environ.keys() else generate_session_id()

CORS(app)


@app.route("/api/train_history", methods=["GET"])
def ai_train_history():
    try:
        session_id = set_session_id(session, session.get("session_id", None))
        if not session_id:
            raise Exception("No Session Found")
        training_results = get_trained_history_results(session_id)
        if not training_results:
            raise Exception("No Trained Model instance")
        return jsonify(**training_results)
    except Exception as e:
        return jsonify(error=f"{e}")
    except:
        return jsonify(error=f"Internal Server Error (500)")


@app.route("/api/train", methods=["POST"])
def ai_train():
    try:
        if "training_data" not in request.files:
            raise Exception("No Dataset found")

        file = request.files.get("training_data", None)

        if not file or file.filename == "":
            raise Exception("No selected file")

        data = {**request.form}
        if file:
            # Read the CSV file into a DataFrame
            dataset = pd.read_csv(file)
            # add the dataset to data
            data["dataset"] = dataset

        keys = {"dataset", "algo", "features", "target", "hyperparameters"}
        for key in keys:
            if key not in data.keys():
                raise Exception("Invalid Request (400)")

        session_id = set_session_id(session, session.get("session_id", None))
        remove_ml_instance(session_id)  # remove the previous machine learning class first
        algo = data.get("algo")
        dataset = data.get("dataset")
        column_features = data.get("features")
        column_features = json.loads(column_features)
        column_target = data.get("target")
        column_target = json.loads(column_target)
        set_ml_instance(session_id, algo, dataset=dataset, column_features=column_features, column_target=column_target)
        ml = get_ml_instance(session_id)
        if not ml:
            raise Exception("Algorithm not found")
        hyperparameters = data.get("hyperparameters")
        hyperparameters = json.loads(hyperparameters)
        hyperparameters = {k: val for k, val in hyperparameters.items() if val is not None}
        ml.configure_training(**hyperparameters)
        trained = ml.train_model()
        if not trained:
            remove_ml_instance(session_id)
            raise Exception("Failed to train model. Check your hyperparameters for errors.")
        training_results = get_trained_history_results(session_id)
        if not training_results:
            raise Exception("Failed to train model. Check your hyperparameters for errors.")
        return jsonify(**training_results)
    except Exception as e:
        print("error:", e)
        return jsonify(error=f"{e}")
    except:
        print("ERROR!")
        return jsonify(error=f"Internal Server Error (500)")


@app.route("/api/predict", methods=["POST"])
def ai_predict():
    try:
        session_id = session.get("session_id", None)
        print("session id:", session_id)
        if not session_id:
            raise Exception("No Session Found")
        ml = get_ml_instance(session_id)
        if not ml:
            raise Exception("No Trained Model instance")
        data = {**request.json}
        keys = {"input"}
        for key in keys:
            if key not in data.keys():
                raise Exception("Invalid Request (400)")
        dinput: dict = data.get("input", {})
        dinput = {y: z for y, z in map(lambda x: [x[0], [x[1]]], dinput.items())}
        result = ml.predict(dinput)
        print("result:")
        print(result)
        return jsonify(result=[*result.tolist()])
    except Exception as e:
        return jsonify(error=f"{e}")
    except:
        return jsonify(error=f"Internal Server Error (500)")


@app.route("/", methods=["GET"])
def home():
    session_id = set_session_id(session, session.get("session_id", None))
    algorithms = get_all_algo_names()
    return render_template("index.html", session_id=session_id, algorithms=algorithms)


@app.route("/train", methods=["GET"])
def train_html():
    session_id = set_session_id(session, session.get("session_id", None))
    algorithms = get_all_algo_names()
    algorithm = request.args.get("algorithm")
    category = str(
        next(category for category in algorithms.keys() if algorithms.get(category, {}).get(algorithm, False))
    )
    algorithm_name = algorithms.get(category, {}).get(algorithm, "")
    algorithm_selection = algorithms.get(category)
    has_trained = not not get_ml_instance(session_id)
    return render_template(
        "train.html",
        session_id=session_id,
        algorithms=algorithms,
        algorithm=algorithm,
        category=category,
        algorithm_name=algorithm_name,
        algorithm_selection=algorithm_selection,
        has_trained=has_trained,
    )


def run_development(app):
    try:
        print("Running server on http://localhost:5000/")
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        print("Error occured in flask app: ", e)


if __name__ == "__main__":
    run_development(app)
