import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Sales Prediction App", layout="centered")
st.title("📊 Sales Prediction App")

# -------------------------------
# LOAD DATA (CORRECT WAY)
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "sales_data.csv")
    df = pd.read_csv(file_path)
    return df

df = load_data()

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# EDA
# -------------------------------
st.subheader("📈 Sales Distribution")
fig, ax = plt.subplots()
ax.hist(df["Sales"], bins=10)
ax.set_xlabel("Sales")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# -------------------------------
# MODEL
# -------------------------------
X = df[["Quantity", "Price"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Evaluation")
st.write(f"**RMSE:** {rmse:.2f}")

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("🧮 Predict Sales")

quantity = st.number_input("Enter Quantity", min_value=1, value=10)
price = st.number_input("Enter Price", min_value=1, value=50)

if st.button("Predict"):
    result = model.predict([[quantity, price]])
    st.success(f"💰 Predicted Sales: {result[0]:.2f}")
