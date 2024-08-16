import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Mapping from input values to encoded values
encoding = {
    'buying': {'vhigh': 3, 'high': 0, 'med': 2, 'low': 1},
    'maint': {'vhigh': 3, 'high': 0, 'med': 2, 'low': 1},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 2, 'med': 1, 'big': 0},
    'safety': {'low': 1, 'med': 2, 'high': 0}
}

# Reverse mapping from predicted class index to label
class_labels = ['good', 'acc', 'unacc', 'vgood']

@app.route('/predict-car-class', methods=['POST'])
def predict_car_class():
    data = request.get_json(force=True)
    
    # Extract and encode input features
    try:
        buying = encoding['buying'][data['buying']]
        maint = encoding['maint'][data['maint']]
        doors = encoding['doors'][data['doors']]
        persons = encoding['persons'][data['persons']]
        lug_boot = encoding['lug_boot'][data['lug_boot']]
        safety = encoding['safety'][data['safety']]
    except KeyError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    
    # Prepare input array for the model
    input_features = np.array([[buying, maint, doors, persons, lug_boot, safety]])
    
    # Make prediction
    prediction = model.predict(input_features)
    predicted_class = class_labels[prediction[0]]
    
    return jsonify(predicted_class)

# Start the server
if __name__ == '__main__':
    app.run(port=5000, debug=True)
