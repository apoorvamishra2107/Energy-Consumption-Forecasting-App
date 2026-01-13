import streamlit as st
import torch
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    layout="wide"
)

# -----------------------------
# CACHED LOADERS (CRITICAL)
# -----------------------------
@st.cache_resource
def load_model():
    model = torch.load("model/lstm_model.pt", map_location="cpu")
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.save")

@st.cache_data
def load_data():
    return pd.read_csv("data/energy_small.csv", sep=";")

model = load_model()
scaler = load_scaler()
data = load_data()

# -----------------------------
# UI
# -----------------------------
st.title("⚡ Energy Consumption Forecasting")

st.sidebar.header("Controls")
SEQ_LEN = st.sidebar.slider("Sequence Length", 12, 48, 24)

st.subheader("Recent Energy Usage")
st.line_chart(
    data["Global_active_power"]
    .astype(float)
    .tail(300)
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Next Consumption"):
    recent = data["Global_active_power"].astype(float).values[-SEQ_LEN:]
    recent = recent.reshape(-1, 1)
    scaled = scaler.transform(recent)

    seq_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(seq_tensor).item()

    prediction = scaler.inverse_transform([[pred_scaled]])[0][0]

    st.success(f"Predicted Next Value: {prediction:.3f} kW")
