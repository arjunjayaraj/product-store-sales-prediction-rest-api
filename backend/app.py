# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
product_store_sales_predictor_api = Flask("SuperKart Product Store Sales Predictor")

# Load the trained machine learning model
model = joblib.load("backend_files/product_store_sales_model_v1_0.joblib")

# Define a route for the home page (GET request)
@product_store_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Product Store Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@product_store_sales_predictor_api.post('/v1/sales')
def product_store_sales():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted product sales as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    input_data = {
        'Product_Id': product_data['Product_Id'],
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_Type': product_data['Product_Type'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Id': product_data['Store_Id'],
        'Store_Establishment_Year': product_data['Store_Establishment_Year'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Make prediction
    predicted_product_store_sales = model.predict(input_data_df)[0]

    
    # Convert predicted_price to Python float
    predicted_product_store_sales = round(float(predicted_product_store_sales), 2)
    

    # Return the actual price
    return jsonify({'Predicted Product store sales (in dollars)': predicted_product_store_sales})


# Define an endpoint for batch prediction (POST request)
@product_store_sales_predictor_api.post('/v1/salesbatch')
def product_stor_sales_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing product details for multiple or single outlets,
    and returns the predicted product store sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all products in the DataFrame
    predicted_product_store_sales = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_rounded_sales = [round(float(sales), 2) for sales in predicted_product_store_sales]

    # Create a dictionary of predictions with product IDs as keys
    product_ids = input_data['Product_Id'].tolist()  # Assuming 'id' is the product ID column
    output_dict = dict(zip(product_ids, predicted_rounded_sales))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    product_store_sales_predictor_api.run(debug=True)
