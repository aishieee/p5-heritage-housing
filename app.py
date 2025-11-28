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

# Load the engineered training data used to fit the model
@st.cache_resource
def load_train_data():
    return pd.read_csv("data/processed/train_engineered.csv")


# -----------------------------------------------------
# Transform sidebar inputs into model-ready features
# -----------------------------------------------------
import numpy as np
import pandas as pd
import altair as alt

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

def transform_inputs(raw_inputs: dict) -> pd.DataFrame:
    """
    Takes the raw sidebar inputs and returns
    a DataFrame with the 10 engineered features
    used by the Random Forest model.
    """

    # Compute total above-ground living area
    gr_liv_area = raw_inputs["1stFlrSF"] + raw_inputs["2ndFlrSF"]

    # Log transforms
    grliv_log = np.log1p(gr_liv_area)
    total_bsmt_log = np.log1p(raw_inputs["TotalBsmtSF"])
    lotarea_log = np.log1p(raw_inputs["LotArea"])

    # Categorical encodings 
    kitchen_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1}
    bsmt_fin_map = {
        "GLQ": 6, "ALQ": 5, "BLQ": 4,
        "Rec": 3, "LwQ": 2, "Unf": 1
    }
    garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1}

    # Apply numeric encodings
    kitchen_enc = kitchen_map[raw_inputs["KitchenQual"]]
    bsmt_exp_enc = bsmt_exposure_map[raw_inputs["BsmtExposure"]]
    bsmt_fin_enc = bsmt_fin_map[raw_inputs["BsmtFinType1"]]
    garage_fin_enc = garage_finish_map[raw_inputs["GarageFinish"]]

    # Build final DataFrame (matches model training order)
    df = pd.DataFrame(
        {
            "GarageArea": [raw_inputs["GarageArea"]],
            "OverallQual": [raw_inputs["OverallQual"]],
            "OverallCond": [raw_inputs["OverallCond"]],
            "KitchenQual": [kitchen_enc],
            "BsmtExposure": [bsmt_exp_enc],
            "BsmtFinType1": [bsmt_fin_enc],
            "GarageFinish": [garage_fin_enc],
            "GrLivArea_log": [grliv_log],
            "TotalBsmtSF_log": [total_bsmt_log],
            "LotArea_log": [lotarea_log],
        }
    )

    # Ensure columns are ordered correctly
    df = df[MODEL_FEATURES]

    return df



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

    st.subheader("📦 Current Inputs (Preview)")
    st.write(raw_inputs)

    # Transform into model-ready features
    transformed_df = transform_inputs(raw_inputs)

    st.subheader("📐 Transformed Features (Model-Ready)")
    st.write(transformed_df)

    # ---------------- Prediction section ------------------------
    st.subheader("💰 Predicted Sale Price")

    if st.button("Predict Price"):
        prediction = rf_model.predict(transformed_df)[0]
        st.success(f"Estimated Sale Price: ${prediction:,.0f}")
        st.caption(
            "Prediction generated using the trained Random Forest model on the Ames housing dataset."
        )
    else:
        st.info("Adjust the inputs in the sidebar and click 'Predict Price' to see an estimate.")

def show_summary_page():
    st.title("🏠 House Price Predictor")
    st.subheader("Project Summary")

    # --- Project Dataset ---
    st.subheader("📂 Project Dataset")
    st.markdown("""
    This project uses the **Ames Housing Dataset**, containing detailed information 
    for over **1,400 residential properties** in Ames, Iowa.  
    The dataset includes structural features, quality ratings, and lot characteristics.
    """)

    # --- Business Requirements ---
    st.subheader("🧾 Business Requirements")
    st.markdown("""
    The client asked for:
    1. **A study of which housing features are most strongly related to SalePrice.**  
    2. **Predicted sale prices for four inherited houses.**  
    3. **An interactive dashboard** to explore data, insights, models, and predictions.
    """)

    # --- Open README ---
    st.subheader("📘 Check out Project README")
    st.markdown("Click below to open the full README file in GitHub:")
    st.link_button("Open README", "https://github.com/aishieee/p5-heritage-housing?tab=readme-ov-file#readme")

    # --- Dataset Guidelines ---
    st.subheader("📑 Dataset Guidelines")
    st.markdown("""
    - Numerical features include areas (sq ft), counts, and quality ratings.  
    - Categorical features (e.g., kitchen quality, garage finish) use ordinal encodings.  
    - Skewed features (LotArea, GrLivArea, TotalBsmtSF) were log-transformed.  
    - Missing values were handled using domain-aware strategies.  
    """)

def show_feature_insights_page():
    st.title("📊 House Sales Price Study")

    # Load data and feature importances
    train_df = load_train_data()

    importances = rf_model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": MODEL_FEATURES, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    # --- Client expectations ---
    st.subheader("🧾 Client Expectations")
    st.markdown(
        """
        The client wants to understand **which housing features are most desirable**
        in terms of increasing sale price. In particular, they asked which 
        variables they should focus on when **renovating or valuing properties**.
        """
    )

    # Inspect House Price Data
    with st.expander("🔍 Inspect a sample of the training data"):
        st.write(train_df.head(10))
    
    # --- Feature importance ---
    st.subheader("⭐ Most Influential Features for Sale Price")

    st.markdown(
        """
        The chart below shows the **Random Forest feature importance scores**.  
        Higher values mean that the feature plays a bigger role in the model's 
        price predictions.
        """
    )

    st.bar_chart(importance_df.set_index("Feature"))

    top3 = importance_df["Feature"].head(3).tolist()

    st.markdown(
        f"""
        The three most influential features in this model are:

        - **{top3[0]}**  
        - **{top3[1]}**  
        - **{top3[2]}**

        This confirms that **overall property quality** and **size of the living/basement
        areas** are key drivers of sale price in the Ames housing market.
        """
    )

    # --- Plots: relationships with SalePrice ---
    st.subheader("📈 View Feature vs. SalePrice Plots")

    left_col, right_col = st.columns(2)

    # Scatter: GrLivArea_log vs SalePrice
    with left_col:
        st.caption("GrLivArea_log vs SalePrice")
        chart1 = (
            alt.Chart(train_df)
            .mark_circle(size=40, opacity=0.5)
            .encode(
                x="GrLivArea_log",
                y="SalePrice",
                tooltip=["GrLivArea_log", "SalePrice"],
            )
            .interactive()
        )
        st.altair_chart(chart1, use_container_width=True)

    # Scatter: TotalBsmtSF_log vs SalePrice
    with right_col:
        st.caption("TotalBsmtSF_log vs SalePrice")
        chart2 = (
            alt.Chart(train_df)
            .mark_circle(size=40, opacity=0.5)
            .encode(
                x="TotalBsmtSF_log",
                y="SalePrice",
                tooltip=["TotalBsmtSF_log", "SalePrice"],
            )
            .interactive()
        )
        st.altair_chart(chart2, use_container_width=True)

    # Boxplot: OverallQual vs median SalePrice
    st.caption("OverallQual vs Median SalePrice")
    qual_df = (
        train_df.groupby("OverallQual")["SalePrice"]
        .median()
        .reset_index()
        .rename(columns={"SalePrice": "MedianSalePrice"})
    )

    chart3 = (
        alt.Chart(qual_df)
        .mark_bar()
        .encode(
            x="OverallQual:O",
            y="MedianSalePrice:Q",
            tooltip=["OverallQual", "MedianSalePrice"],
        )
    )
    st.altair_chart(chart3, use_container_width=True)

    st.markdown(
        """
        These plots reinforce the earlier finding:

        - Houses with **larger living areas** and **larger basements** tend to sell for more.  
        - As **OverallQual** increases, the **median SalePrice** rises sharply.

        This provides the client with clear visual evidence of which improvements
        are most likely to increase a property's market value.
        """
    )


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
