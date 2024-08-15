import numpy as np
from flask import Flask, request, jsonify
from simple_linear_regr_utils import generate_data
from simple_linear_regr import SimpleLinearRegression

app = Flask(__name__)

# Load and train the model
X_train, y_train, X_test, y_test = generate_data()
model = SimpleLinearRegression()
model.fit(X_train, y_train)

@app.route('/stream', methods=['POST'])
def stream():
    data = request.json
    X_new = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(X_new)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/batch', methods=['POST'])
def batch():
    data = request.json
    X_new = np.array(data['features'])
    predictions = model.predict(X_new)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)