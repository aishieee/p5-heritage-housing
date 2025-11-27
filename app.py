import streamlit as st
import joblib

# Label - code mappings
KITCHEN_MAP = {
    "Excellent": "Ex",
    "Good": "Gd",
    "Typical/Average": "TA",
    "Fair": "Fa",
    "Poor": "Po",
}

EXPOSURE_MAP = {
    "No Exposure": "No",
    "Minimum Exposure": "Mn",
    "Average Exposure": "Av",
    "Good Exposure": "Gd",
}

BSMT_FIN_MAP = {
    "Unfinished": "Unf",
    "Low Quality": "LwQ",
    "Rec Room": "Rec",
    "Below Average Living Quarters": "BLQ",
    "Average Living Quarters": "ALQ",
    "Good Living Quarters": "GLQ",
}

GARAGE_FIN_MAP = {
    "Unfinished": "Unf",
    "Rough Finished": "RFn",
    "Finished": "Fin",
}

st.set_page_config(
    page_title="Heritage Housing Price Prediction",
    layout="wide"
)

# Load model (test only)
@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.pkl")

rf_model = load_model()

# -----------------------------------------------------
# Transform sidebar inputs into model-ready features
# -----------------------------------------------------
import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "GarageArea",
    "OverallQual",
    "OverallCond",
    "KitchenQual",
    "BsmtExposure",
    "BsmtFinType1",
    "GarageFinish",
    "GrLivArea_log",
    "TotalBsmtSF_log",
    "LotArea_log",
]

# ---------------------------------------
# Placeholder for sidebar page functions 
# ---------------------------------------
def show_inherited_prediction_page():
    st.title("Inherited Houses & Price Prediction")
    st.write("This page will show the 4 inherited houses and allow the user to predict prices.")

    # --------------- Sidebar inputs ---------------------
    st.sidebar.header("🔧 House Feature Inputs")

    # Size features
    first_flr = st.sidebar.number_input(
        "1st Floor Area (sq ft)", min_value=200, max_value=3000, value=900, step=10
    )
    second_flr = st.sidebar.number_input(
        "2nd Floor Area (sq ft)", min_value=0, max_value=2500, value=0, step=10
    )
    lot_area = st.sidebar.number_input(
        "Lot Area (sq ft)", min_value=1000, max_value=40000, value=10000, step=100
    )
    total_bsmt = st.sidebar.number_input(
        "Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800, step=10
    )
    garage_area = st.sidebar.number_input(
        "Garage Area (sq ft)", min_value=0, max_value=1200, value=400, step=10
    )

    # Quality features
    overall_qual = st.sidebar.slider(
        "Overall Quality (1–10)", min_value=1, max_value=10, value=5
    )
    overall_cond = st.sidebar.slider(
        "Overall Condition (1–10)", min_value=1, max_value=10, value=5
    )

    # Categorical encoded features
    kitchen_qual = st.sidebar.selectbox(
        "Kitchen Quality", ["Excellent", "Good", "Typical/Average", "Fair", "Poor"]
    )
    bsmt_exposure = st.sidebar.selectbox(
        "Basement Exposure", ["No Exposure", "Minimum Exposure", "Average Exposure", "Good Exposure"]
    )
    bsmt_fin_type1 = st.sidebar.selectbox(
        "Basement Finish Type",
        ["Unfinished", "Low Quality", "Rec Room", 
        "Below Average Living Quarters", 
        "Average Living Quarters", 
        "Good Living Quarters"]
    )
    garage_finish = st.sidebar.selectbox(
        "Garage Finish",
        ["Unfinished", "Rough Finished", "Finished"]
    )

    # Show the user inputs as a dictionary (for debugging before prediction)
    st.subheader("📦 Current Inputs (Preview)")
    raw_inputs = {
        "1stFlrSF": first_flr,
        "2ndFlrSF": second_flr,
        "LotArea": lot_area,
        "TotalBsmtSF": total_bsmt,
        "GarageArea": garage_area,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "KitchenQual": KITCHEN_MAP[kitchen_qual],
        "BsmtExposure": EXPOSURE_MAP[bsmt_exposure],
        "BsmtFinType1": BSMT_FIN_MAP[bsmt_fin_type1],
        "GarageFinish": GARAGE_FIN_MAP[garage_finish],
    }
    st.write(raw_inputs)

def show_summary_page():
    st.title("Project Summary")
    st.write("This page will describe the project, dataset, and client requirements.")

def show_feature_insights_page():
    st.title("Feature Insights")
    st.write("This page will show which features are most strongly related to SalePrice.")

def show_hypotheses_page():
    st.title("Project Hypotheses")
    st.write("This page will explain the project hypotheses and how they were tested.")

def show_model_performance_page():
    st.title("Model Performance")
    st.write("This page will show the model metrics and pipeline details.")

def main():
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "Project Summary",
            "Feature Insights",
            "Inherited Houses & Prediction",
            "Hypotheses",
            "Model Performance",
        ],
    )

    # Show the selected page
    if page == "Project Summary":
        show_summary_page()
    elif page == "Feature Insights":
        show_feature_insights_page()
    elif page == "Inherited Houses & Prediction":
        show_inherited_prediction_page()
    elif page == "Hypotheses":
        show_hypotheses_page()
    elif page == "Model Performance":
        show_model_performance_page()
        
if __name__ == "__main__":
    main()
