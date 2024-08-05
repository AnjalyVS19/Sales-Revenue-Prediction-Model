# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():

    try:
        data = request.json
        
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
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
