import streamlit as st
import joblib

st.set_page_config(
    page_title="Heritage Housing Price Prediction",
    layout="wide"
)

# Load model (test only)
@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.pkl")

rf_model = load_model()

st.title("Heritage Housing Price Prediction App")
st.write("Model has been loaded.")
