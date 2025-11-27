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

# ---------------------------------------
# Placeholder for sidebar page functions 
# ---------------------------------------
def show_inherited_prediction_page():
    st.title("Inherited Houses & Price Prediction")
    st.write("This page will show the 4 inherited houses and allow the user to predict prices.")

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
