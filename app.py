from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import predict
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.json
        logging.info('Received data: %s', data)

        # Input validation
        required_fields = ['units_sold', 'unit_price', 'product_category', 'region', 'payment_method']
        for field in required_fields:
            if field not in data:
                logging.error('Missing field: %s', field)
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Convert the input data to DataFrame with correct column names
        df = pd.DataFrame([{
            'Units Sold': data['units_sold'],
            'Unit Price': data['unit_price'],
            'Product Category': data['product_category'],
            'Region': data['region'],
            'Payment Method': data['payment_method']
        }])

        # Predict using the model
        prediction = predict(df)
        logging.info('Prediction made: %s', prediction[0])
        return jsonify({'prediction': round(prediction[0], 2)}), 200

    except KeyError as e:
        logging.error('KeyError: %s', str(e))
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        logging.error('Error occurred: %s', str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
