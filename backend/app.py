# Import necessary libraries
import os
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
from datetime import datetime # For Store_Establishment_Year feature engineering

# MRP_BIN_EDGES must be defined before engineer_features if it's used within
mrp_quantiles = np.array([126.2225, 146.585, 167.505]) # Copied from notebook cell 'fet3-MuLqbBL'
MRP_BIN_EDGES = [-np.inf, mrp_quantiles[0], mrp_quantiles[1], mrp_quantiles[2], np.inf]

def engineer_features(df, mrp_bin_edges):
    df = df.copy()
    # Dynamically fetch current year (e.g., 2026, 2027, etc.)
    current_year = datetime.now().year

    # 1. Store Age with fallback
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age'] = current_year - df['Store_Establishment_Year']
    else:
        # Assign NaN so SimpleImputer can handle it cleanly downstream
        df['Store_Age'] = np.nan

    # 2. Product Category with fallback
    if 'Product_Id' in df.columns:
        df['Product_Category'] = df['Product_Id'].astype(str).str[:2]
    else:
        df['Product_Category'] = np.nan

    # 3. MRP_BIN with fallback
    if 'Product_MRP' in df.columns and mrp_bin_edges is not None:
        df['MRP_CAT'] = pd.cut(
            df['Product_MRP'],
            bins=mrp_bin_edges,
            labels=['Low', 'Medium', 'High', 'Premium'],
            include_lowest=True
        )
    else:
        df['MRP_CAT'] = np.nan

    # Drop raw columns if present
    cols_to_drop = ['Product_Id', 'Store_Establishment_Year']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df
# --- End of Feature Engineering components ---

# Initialize the Flask application
product_store_sales_predictor_api = Flask("SuperKart Product Store Sales Predictor")

# Resolve the model path relative to this file so it works both locally and in Docker
MODEL_PATH = os.path.join(os.path.dirname(__file__), "product_store_sales_model_v1_0.joblib")


# Load the trained machine learning model
# This ensures that the FunctionTransformer within the pipeline can find its dependencies.
model = joblib.load(MODEL_PATH)

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
