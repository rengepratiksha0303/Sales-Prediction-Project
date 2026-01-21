import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Sales Prediction App", layout="centered")
st.title("📊 Sales Prediction App")

# -----------------------------------
# LOAD MODEL (SAFE WAY)
# -----------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "Sales.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------------
# OPTIONAL: LOAD DATA FOR DISPLAY
# -----------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "sales_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

if df is not None:
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

# -----------------------------------
# USER INPUT
# -----------------------------------
st.subheader("🧮 Predict Sales")

quantity = st.number_input("Enter Quantity Sold", min_value=1, value=10)
price = st.number_input("Enter Price per Unit", min_value=1, value=50)

if st.button("Predict Sales"):
    input_data = np.array([[quantity, price]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Sales Amount: {prediction[0]:.2f}")
