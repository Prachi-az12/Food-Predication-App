import streamlit as st
import pickle
import numpy as np

# Model load करा (Random Forest Regression pickle फाईल)
rf = pickle.load(open("food_delivery_sample.pkl", "rb"))

st.title("🍔 Food Delivery Time Prediction")

distance = st.number_input("Delivery Distance (km)", 0.1, 40.0, 5.0)
prep_time = st.number_input("Preparation Time (minutes)", 5, 60, 30)
traffic = st.slider("Traffic Level Score", 0.0, 10.0, 5.0)
weather = st.slider("Weather Severity Score", 0.0, 10.0, 5.0)
rating = st.slider("Restaurant Rating", 1.0, 5.0, 4.0)

if st.button("Predict Delivery Time"):
    # Inputs NumPy array मध्ये convert करा
    input_data = np.array([[distance, prep_time, traffic, weather, rating]])
    
    # Direct prediction (scaler नाही वापरला)
    prediction = rf.predict(input_data)[0]
   
    st.success(f"Estimated Delivery Time: {round(prediction, 2)} minutes")
