import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Sales Prediction App", layout="centered")

st.title("📊 Sales Prediction Using Machine Learning")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    return df

df = load_data()

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Basic EDA
st.subheader("📈 Exploratory Data Analysis")

if st.checkbox("Show Sales Distribution"):
    fig, ax = plt.subplots()
    ax.hist(df["Sales"], bins=10)
    ax.set_title("Sales Distribution")
    st.pyplot(fig)

# Feature Selection
X = df[["Quantity", "Price"]]
y = df["Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")

# User Input for Prediction
st.subheader("🧮 Predict Sales")

quantity = st.number_input("Enter Quantity Sold", min_value=1, value=10)
price = st.number_input("Enter Price per Unit", min_value=1, value=50)

if st.button("Predict Sales"):
    prediction = model.predict([[quantity, price]])
    st.success(f"💰 Predicted Sales Amount: {prediction[0]:.2f}")
