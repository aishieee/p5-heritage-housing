import streamlit as st
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

@st.cache_resource
def load_test_data():
    return pd.read_csv("data/processed/test_engineered.csv")

# Load the 4 inherited houses with their predicted sale prices.
@st.cache_resource
def load_inherited_predictions():
    return pd.read_csv("data/processed/inherited_predictions.csv")

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
    # Hero Image
    st.image(
        "images/iowa-neighbourhood.avif", 
        width=600
    )
    st.title("Inherited Houses & Price Prediction")
    st.write(
        """
        This page has two parts:

        1. A summary of the **4 inherited houses** and their predicted sale prices.  
        2. A **real-time price predictor** where the user can try out new house configurations.
        """
    )

    # Part 1: Inherited houses summary
    st.subheader("🏡 Predicted Prices for the 4 Inherited Houses")

    inherited_df = load_inherited_predictions()

    # Show the table
    st.dataframe(inherited_df)

    # --- Individual predicted prices  ---
    st.subheader("💵 Predicted Sale Price per House")

    if "PredictedSalePrice" in inherited_df.columns:
        for idx, row in inherited_df.iterrows():
            house_num = idx + 1
            price = row["PredictedSalePrice"]
            st.markdown(f"- **House {house_num}: ${price:,.0f}**")
    else:
        st.warning("Prediction column not found in inherited houses file.")

    # --- Total value ---
    if "PredictedSalePrice" in inherited_df.columns:
        total_value = inherited_df["PredictedSalePrice"].sum()
        st.markdown(
            f"""
            **Total estimated value for all 4 inherited houses:**  
            👉 **${total_value:,.0f}**
            """
        )

    st.markdown("---")  

    # Part 2: Real-time prediction form 
    st.subheader("Check Your Own Predicted House Price")
    st.write(
        " 👈 Use the sidebar on the left to adjust the house features and generate a new prediction."
    )

    # --------------- Sidebar inputs ---------------------
    st.sidebar.image("images/house-for-sale-sign.webp", use_container_width=True)
    
    # Sidebar image
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
    st.title("🧪 Project Hypotheses & Validation")

    st.markdown(
        """
        This page summarises the main hypotheses defined after the initial
        exploration of the Ames Housing dataset and explains **how they were
        tested and supported** using exploratory data analysis (EDA) and
        machine learning (ML) models.
        """
    )

  
    # 1. --- Initial hypotheses --- 
    st.subheader("📌 Initial Hypotheses")

    st.markdown(
        """
        The following hypotheses were defined **before** data cleaning and modelling:

        1. **Larger homes sell for higher prices.**  
           Homes with greater living area (`GrLivArea`, `1stFlrSF`, `TotalBsmtSF`) 
           and a larger overall footprint should achieve higher sale prices.

        2. **Higher construction quality increases sale price.**  
           `OverallQual` (materials and finish quality) is expected to be one of 
           the strongest predictors of price.

        3. **Homes with larger garages sell for higher prices.**  
           Homes with bigger `GarageArea` and more garage spaces (`GarageCars`) 
           are expected to sell for more.

        4. **Newer or recently renovated homes are worth more.**  
           `YearBuilt` and `YearRemodAdd` should be positively associated with 
           sale price, as newer properties typically require less maintenance.

        5. **Smaller features have limited influence.**  
           Variables such as `BedroomAbvGr`, `EnclosedPorch` and `OverallCond` 
           were expected to have weaker relationships with `SalePrice`.
        """
    )

    # 2. --- How / Why --- 
    st.subheader("🔍 How the Hypotheses Were Tested and Why")

    st.markdown(
        """
        To test these hypotheses in a structured way, the project used a combination of:

        **1. Correlation analysis (Pearson correlation matrix).**  
        This was used to quickly identify **linear relationships** between numerical
        variables and `SalePrice`. It is appropriate here because most of the main
        predictors (size, quality scores, years) are numeric and reasonably continuous.
        The correlation heatmap in the EDA notebook showed that:

        * `OverallQual` and `GrLivArea` had some of the **strongest positive correlations**
          with `SalePrice`.
        * Basement area and garage size also showed positive, but slightly weaker,
          correlations.
        * Features such as `BedroomAbvGr` and `EnclosedPorch` had noticeably lower
          correlations with price.

        **2. Visual explorations (scatterplots and grouped summaries).**  
        Scatterplots of `GrLivArea`, `TotalBsmtSF` and `GarageArea` against `SalePrice`
        were used to visually confirm the trends seen in the correlation matrix.  
        Grouped summaries (e.g. median `SalePrice` by `OverallQual`) showed a clear,
        almost stepwise increase in price as quality improved.

        **3. Feature engineering and transformations.**  
        Several features (`GrLivArea`, `LotArea`, `TotalBsmtSF`) were log-transformed
        to reduce skew and make the relationships with `SalePrice` more linear.  
        This choice was important for Linear Regression, which assumes a more linear
        relationship, and also helped stabilise the scale for the Random Forest model.

        **4. Two complementary ML models.**  
        A **Linear Regression** model was used as a simple baseline to test whether a
        mostly linear relationship could already explain a large portion of the
        variation in house prices.  

        A **Random Forest Regressor** was then trained as the main model, because it can:
        * capture **non-linear relationships** (e.g. quality thresholds),
        * handle interactions between features (e.g. large size *and* high quality),
        * and is robust to outliers and different feature scales.

        Comparing these two models helps validate whether the relationships suggested
        in the hypotheses are strong enough to be learned by both simple and more
        flexible algorithms.
        """
    )

    # 3. --- Evidence from model performance --- 
    st.subheader("📈 Evidence from Model Performance")

    st.markdown(
        """
        The models achieved the following performance on the training/test sets:

        * **Linear Regression (scaled features):**  
          • Train R² ≈ **0.79**  
          • Test R² ≈ **0.80**  
          • Test RMSE ≈ **39,600** and Test MAE ≈ **24,300**

          This shows that a simple linear model can already explain a large portion
          of the variation in sale prices, which supports the idea that size and
          quality have strong, mostly monotonic relationships with price.

        * **Random Forest Regressor (unscaled features):**  
          • Train R² ≈ **0.97**  
          • Test R² ≈ **0.89**  
          • Test RMSE ≈ **28,700** and Test MAE ≈ **17,800**

          The Random Forest significantly reduces the error compared to Linear
          Regression, especially MAE, indicating that non-linear effects and
          interactions between size, quality, basement and garage features help
          explain additional variation in `SalePrice`. This gives stronger support
          to the hypotheses about these features being key drivers of price.
        """
    )

    # 4. --- Evidence from feature importance ---
    st.subheader("Evidence from Feature Importance")

    importances = rf_model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": MODEL_FEATURES, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    top_rows = importance_df.head(5)

    st.markdown(
        """
        The Random Forest feature importance values provide direct evidence of how
        much each engineered feature contributed to the final predictions.
        """
    )

    st.dataframe(top_rows)

    st.markdown(
        """
        In this project:

        * **`OverallQual`** has the highest importance (around **0.58**), confirming
          that construction and finish quality is the most influential single factor.
        * **`GrLivArea_log`** (above-ground living area) has the second-highest
          importance (around **0.18**), matching the hypothesis that larger homes
          sell for more.
        * **`TotalBsmtSF_log`**, **`GarageArea`** and **`LotArea_log`** follow next,
          supporting the idea that basement size, garage size and lot size all add
          value, although to a lesser degree than overall quality and main living area.
        * Variables such as `BsmtFinType1`, `KitchenQual`, `GarageFinish` and
          `BsmtExposure` have smaller, but still meaningful, contributions and help
          refine the price estimate.
        """
    )

    # 5. --- Final Summary ---
    st.subheader("✅ Summary of Findings")

    st.markdown(
        """
        * **H1 – Larger homes sell for higher prices → Supported.**  
          Log-transformed living area and basement size show strong positive
          relationships with `SalePrice` in both the EDA plots and the Random
          Forest feature importance ranking.

        * **H2 – Higher construction quality increases sale price → Strongly supported.**  
          `OverallQual` is both highly correlated with `SalePrice` and by far the
          most important feature in the Random Forest model.

        * **H3 – Homes with larger garages sell for higher prices → Supported.**  
          `GarageArea` has a clear positive trend with `SalePrice` and appears among
          the more important features, although it is less influential than overall
          quality and main living area.

        * **H4 – Newer or recently renovated homes are worth more → Partially supported.**  
          `YearBuilt` and `YearRemodAdd` showed the expected positive trends in the
          EDA, but their importance in the final model was lower than that of size
          and quality features. Age still matters, but not as strongly as originally
          expected.

        * **H5 – Smaller features have limited influence → Supported.**  
          Features such as `BedroomAbvGr`, `EnclosedPorch` and `OverallCond`
          showed weaker correlations with `SalePrice` and did not appear as key
          drivers in the model, confirming that they play a secondary role.

        Overall, the combination of EDA, correlation analysis, model performance
        and feature importance strongly supports the conclusion that **overall
        quality and property size are the primary drivers of house prices in Ames**,
        with garage size, basement size and age acting as secondary factors.
        """
    )

def show_model_performance_page():
    st.title("Model Performance & Evaluation")

    st.markdown(
        """
        This page summarises how well the models perform on unseen data and 
        explains why the Random Forest Regressor was chosen as the final model.
        """
    )

    # 1. --- Linear Regression summary ---
    st.subheader("📉 Linear Regression (baseline model)")

    st.markdown(
        """
        A Linear Regression model was trained on the **scaled** engineered features.  
        It provides a simple baseline assuming mostly linear relationships
        between the predictors and `SalePrice`.
        """
    )

    lr_metrics = pd.DataFrame(
        {
            "Dataset": ["Train", "Test"],
            "R²": [0.786, 0.796],
            "RMSE": [35724, 39582],
            "MAE": [23172, 24307],
        }
    )

    st.table(lr_metrics.style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}"}))

    st.markdown(
        """
        These results show that a basic linear model can already explain close to 
        **80% of the variance** in house prices, but the error values leave room 
        for improvement, especially for more complex properties.
        """
    )

    # 2. --- Random Forest evaluation --- 
    st.subheader("🌲 Random Forest Regressor (final model)")

    test_df = load_test_data()
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["SalePrice"]

    # Predictions
    y_pred_rf = rf_model.predict(X_test)

    # Metrics
    r2_test_rf = r2_score(y_test, y_pred_rf)
    rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_test_rf = mean_absolute_error(y_test, y_pred_rf)

    rf_metrics = pd.DataFrame(
        {
            "Dataset": ["Test"],
            "R²": [r2_test_rf],
            "RMSE": [rmse_test_rf],
            "MAE": [mae_test_rf],
        }
    )

    st.table(rf_metrics.style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R²": "{:.3f}"}))

    st.markdown(
        """
        Compared with Linear Regression, the Random Forest:

        * Achieves a **higher R²** on the test set (closer to 0.90).  
        * Reduces the **RMSE and MAE**, meaning predictions are closer to the
          true sale prices on average.
        * Can capture **non-linear effects** and interactions between features
          (for example, high quality *and* large living area).
        """
    )

    # 3. --- Actual vs Predicted plot --- 
    st.subheader("📈 Actual vs Predicted SalePrice (Random Forest)")

    plot_df = pd.DataFrame(
        {
            "ActualSalePrice": y_test,
            "PredictedSalePrice": y_pred_rf,
        }
    )

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=40, opacity=0.5)
        .encode(
            x="ActualSalePrice",
            y="PredictedSalePrice",
            tooltip=["ActualSalePrice", "PredictedSalePrice"],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown(
        """
        The points are clustered close to the diagonal, which indicates that the
        model predictions follow the actual sale prices closely.  
        Larger deviations from the diagonal correspond to properties where the
        model under- or over-estimates the price.

        Overall, the evaluation confirms that the **Random Forest Regressor** is a
        suitable final model for this problem: it generalises well to unseen data
        and captures the main drivers of house prices in the Ames dataset.
        """
    )

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
