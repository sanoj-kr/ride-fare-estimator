import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
pipeline = joblib.load('fare_pipeline.pkl')

st.set_page_config(page_title="Dynamic Ride Fare Estimator", layout="centered")
st.title("🚕 Dynamic Ride Fare Estimator")
st.write("Enter ride details to get an estimated fare.")

# Input form
with st.form("fare_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        riders = st.number_input("Number of Riders", min_value=0, value=50)
        drivers = st.number_input("Number of Drivers", min_value=0, value=50)
        duration = st.slider("Expected Ride Duration (mins)", 1, 120, 30)
        past_rides = st.number_input("Number of Past Rides", min_value=0, value=10)
        ratings = st.slider("Average Ratings", 1.0, 5.0, 4.5)

    with col2:
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
        loyalty = st.selectbox("Customer Loyalty Status", ['Regular', 'Silver', 'Gold'])
        time = st.selectbox("Time of Booking", ["Morning", "Afternoon", "Evening", "Night"])
        vehicle = st.selectbox("Vehicle Type", ["Economy", "Premium"])
    
    submit = st.form_submit_button("Predict Fare")

# Preprocess input
def preprocess_input(riders, drivers, duration, past_rides, ratings, location, loyalty, time, vehicle):
    input_df = pd.DataFrame({
        'Number_of_Riders': [riders],
        'Number_of_Drivers': [drivers],
        'Expected_Ride_Duration': [duration],
        'Number_of_Past_Rides': [past_rides],
        'Average_Ratings': [ratings],
        'Location_Category': [location],
        'Customer_Loyalty_Status': [loyalty],
        'Time_of_Booking': [time],
        'Vehicle_Type': [vehicle]
    })

    # TODO: Apply the same encoding as in training (use saved encoders if applicable)
    # For now, use dummy encoding as placeholder

    return input_df

if submit:
    input_df = preprocess_input(riders, drivers, duration, past_rides, ratings, location, loyalty, time, vehicle)
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted Fare: ₹{round(prediction, 2)}")