import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("house_price_model.joblib")

st.title("🏠 House Price Prediction App")

bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
sqft = st.number_input("Square Feet", 300, 10000, 1500)

if st.button("Predict Price"):
    data = np.array([[bedrooms, bathrooms, sqft]])
    price = model.predict(data)

    st.success(f"💰 Estimated House Price: ₹ {price[0]:,.2f}")
