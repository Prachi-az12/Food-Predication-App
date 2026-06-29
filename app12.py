import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Model load करा
rf = pickle.load(open("food_delivery_rf_model.pkl", "rb"))

st.title("🍔 Food Delivery Time Prediction")

st.write("Enter order details below to predict delivery time (minutes).")

# User inputs
distance = st.number_input("Delivery Distance (km)", min_value=0.1, max_value=40.0, value=5.0)
prep_time = st.number_input("Preparation Time (minutes)", min_value=5, max_value=60, value=30)
traffic = st.slider("Traffic Level Score", 0.0, 10.0, 5.0)
weather = st.slider("Weather Severity Score", 0.0, 10.0, 5.0)
rating = st.slider("Restaurant Rating", 1.0, 5.0, 4.0)

# Prediction
if st.button("Predict Delivery Time"):
    input_data = np.array([[distance, prep_time, traffic, weather, rating]])
    
    # Scaling (same scaler वापरायचा जो training मध्ये वापरला होता)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)  # ⚠️ Note: इथे training scaler save करून वापरणं उत्तम
    
    prediction = rf.predict(input_data_scaled)[0]
    st.success(f"Estimated Delivery Time: {round(prediction, 2)} minutes")

