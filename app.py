import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and scaler
with open('sales_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

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
item_category_fd = st.selectbox('Item Category FD', [0,1])

# Create a DataFrame from the input values
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
    'Item_Category_FD': [item_category_fd]
})

# Scale numerical features
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make prediction
if st.button('Predict Sales'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Sales: ${prediction[0]:.2f}')
