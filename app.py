from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)

# A simple endpoint for machine learning predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Example: Accept JSON data { "input": [1.0, 2.0, 3.0] }
    data = request.json
    input_data = np.array(data["input"]).reshape(-1, 1)
    
    # Dummy ML model
    model = LinearRegression()
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([2, 4, 6, 8])
    model.fit(X_train, y_train)

    prediction = model.predict(input_data).tolist()
    return jsonify({"prediction": prediction})

@app.route('/')
def home():
    return render_template('index.html')

# API route with CORS support
@app.route('/api/greet', methods=['POST'])
def greet():
    data = request.json
    name = data.get('name', 'Guest')
    return jsonify(message=f"Hello, {name}!")

if __name__ == '__main__':
    app.run(debug=True)
