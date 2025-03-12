import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Check if the model file exists
if not os.path.exists('sales_prediction_model.pkl'):
    st.error("Model file 'sales_prediction_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()  # Stop execution if the model file is missing

# Check if the scaler file exists
if not os.path.exists('scaler.pkl'):
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
    st.stop()  # Stop execution if the scaler file is missing

# Load the model and trained_columns
try:
    with open('sales_prediction_model.pkl', 'rb') as file:
        model, trained_columns = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Define input features (adjust based on your actual features)
st.title('Sales Prediction App')

# Example input widgets (adjust to match your features)
item_weight = st.number_input('Item Weight', value=10.0)
item_visibility = st.number_input('Item Visibility', value=0.05)
item_mrp = st.number_input('Item MRP', value=150.0)
outlet_age = st.number_input('Outlet Age', value=20)

# Example categorical inputs (adjust to match your one-hot encoded columns)
item_fat_content_low_fat = st.selectbox('Item Fat Content Low Fat', [0, 1])
item_type_dairy = st.selectbox('Item Type Dairy', [0, 1])
item_type_softdrinks = st.selectbox('Item Type Soft Drinks', [0, 1])
outlet_size_medium = st.selectbox('Outlet Size Medium', [0, 1])
outlet_location_type_tier1 = st.selectbox('Outlet Location Type Tier 1', [0, 1])
outlet_type_supermarket_type1 = st.selectbox('Outlet Type Supermarket Type1', [0, 1])
item_category_fd = st.selectbox('Item Category FD', [0, 1])

# Then when creating your input data.
input_data = pd.DataFrame({
    'Item_Weight': [item_weight],
    'Item_Visibility': [item_visibility],
    'Item_MRP': [item_mrp],
    'Outlet_Age': [outlet_age],
    'Item_Fat_Content_Low Fat': [item_fat_content_low_fat],
    'Item_Type_Dairy': [item_type_dairy],
    'Item_Type_Soft Drinks': [item_type_softdrinks],
    'Outlet_Size_Medium': [outlet_size_medium],
    'Outlet_Location_Type_Tier 1': [outlet_location_type_tier1],
    'Outlet_Type_Supermarket Type1': [outlet_type_supermarket_type1],
    'Item_Category_FD': [item_category_fd],
    'Item_Fat_Content_Regular': [0], #Add these columns and set to 0 or 1.
    'Item_Type_Breads': [0],
    'Item_Type_Breakfast': [0],
    'Item_Type_Canned': [0],
    'Item_Type_Frozen Foods': [0],
    'Item_Type_Fruits and Vegetables': [0],
    'Item_Type_Hard Drinks': [0],
    'Item_Type_Health and Hygiene': [0],
    'Item_Type_Household': [0],
    'Item_Type_Meat': [0],
    'Item_Type_Others': [0],
    'Item_Type_Seafood': [0],
    'Item_Type_Snack Foods': [0],
    'Item_Type_Starchy Foods': [0],
    'Outlet_Size_Small': [0],
    'Outlet_Location_Type_Tier 2': [0],
    'Outlet_Location_Type_Tier 3': [0],
    'Outlet_Type_Supermarket Type2': [0],
    'Outlet_Type_Supermarket Type3': [0],
    'Item_Category_NC': [0]
})

# Debugging: Print trained_columns and input_data columns
print("Trained Columns:", trained_columns)
print("Input Data Columns:", input_data.columns.tolist())

# Reorder the columns to match the trained model
input_data = input_data[trained_columns]

# Scale numerical features
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make prediction
if st.button('Predict Sales'):
    try:
        prediction = model.predict(input_data)
        st.write(f'Predicted Sales: ${prediction[0]:.2f}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
