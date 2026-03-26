import streamlit as st
import joblib
import pandas as pd
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = joblib.load(os.path.join(BASE_DIR, "../models/model.pkl"))

# UI Title
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")
st.title("Water Potability Prediction App")

st.markdown("Enter water quality parameters below:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness")
    solids = st.number_input("Solids")
    chloramines = st.number_input("Chloramines")
    sulfate = st.number_input("Sulfate")

with col2:
    conductivity = st.number_input("Conductivity")
    organic_carbon = st.number_input("Organic Carbon")
    trihalomethanes = st.number_input("Trihalomethanes")
    turbidity = st.number_input("Turbidity")

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame(
        {
            "ph": [ph],
            "Hardness": [hardness],
            "Solids": [solids],
            "Chloramines": [chloramines],
            "Sulfate": [sulfate],
            "Conductivity": [conductivity],
            "Organic_carbon": [organic_carbon],
            "Trihalomethanes": [trihalomethanes],
            "Turbidity": [turbidity],
        }
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Water is Potable")
    else:
        st.error("Water is NOT Potable")