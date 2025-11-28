import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

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
    layout="wide",
)


@st.cache_resource
def load_model():
    """
    Load the trained Random Forest model.
    """
    return joblib.load("models/rf_model.pkl")


rf_model = load_model()


@st.cache_resource
def load_train_data():
    """
    Load the engineered training data.
    """
    return pd.read_csv("data/processed/train_engineered.csv")


@st.cache_resource
def load_test_data():
    """
    Load the engineered test data.
    """
    return pd.read_csv("data/processed/test_engineered.csv")


@st.cache_resource
def load_inherited_predictions():
    """
    Load the 4 inherited houses with predicted sale prices.
    """
    return pd.read_csv("data/processed/inherited_predictions.csv")


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
    Convert raw sidebar inputs into the engineered features used
    by the Random Forest model.
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
        "GLQ": 6,
        "ALQ": 5,
        "BLQ": 4,
        "Rec": 3,
        "LwQ": 2,
        "Unf": 1,
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
    return df[MODEL_FEATURES]


def show_inherited_prediction_page():
    """
    Page: Inherited houses summary and interactive prediction.
    """
    # Hero Image
    st.image(
        "images/iowa-neighbourhood.avif",
        width=600,
    )

    st.title("Inherited Houses & Price Prediction")
    st.write(
        """
        This page has two parts:

        1. A summary of the 4 inherited houses and their predicted sale prices.
        2. A real-time price predictor to try out new house configurations.
        """
    )

    # Part 1: Inherited houses summary
    st.subheader("🏡 Predicted Prices for the 4 Inherited Houses")

    inherited_df = load_inherited_predictions()
    st.dataframe(inherited_df)

    # Individual predicted prices
    st.subheader("💵 Predicted Sale Price per House")

    if "PredictedSalePrice" in inherited_df.columns:
        for idx, row in inherited_df.iterrows():
            house_num = idx + 1
            price = row["PredictedSalePrice"]
            st.markdown(f"- **House {house_num}: ${price:,.0f}**")
    else:
        st.warning("Prediction column not found in inherited houses file.")

    # Total value
    if "PredictedSalePrice" in inherited_df.columns:
        total_value = inherited_df["PredictedSalePrice"].sum()
        st.markdown(
            """
        **Total estimated value for all 4 inherited houses:**
        👉 **${:,.0f}**
            """.format(total_value)
        )

    st.markdown("---")

    # Part 2: Real-time prediction form
    st.subheader("Check Your Own Predicted House Price")
    st.write(
        """
        👈 Use the sidebar on the left to adjust the house features
        and generate a new prediction.
        """
    )

    # Sidebar content
    st.sidebar.image(
        "images/house-for-sale-sign.webp",
        use_container_width=True,
    )
    st.sidebar.header("🔧 House Feature Inputs")

    # Size features
    first_flr = st.sidebar.number_input(
        "1st Floor Area (sq ft)",
        min_value=200,
        max_value=3000,
        value=900,
        step=10,
    )
    second_flr = st.sidebar.number_input(
        "2nd Floor Area (sq ft)",
        min_value=0,
        max_value=2500,
        value=0,
        step=10,
    )
    lot_area = st.sidebar.number_input(
        "Lot Area (sq ft)",
        min_value=1000,
        max_value=40000,
        value=10000,
        step=100,
    )
    total_bsmt = st.sidebar.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        max_value=3000,
        value=800,
        step=10,
    )
    garage_area = st.sidebar.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1200,
        value=400,
        step=10,
    )

    # Quality features
    overall_qual = st.sidebar.slider(
        "Overall Quality (1–10)",
        min_value=1,
        max_value=10,
        value=5,
    )
    overall_cond = st.sidebar.slider(
        "Overall Condition (1–10)",
        min_value=1,
        max_value=10,
        value=5,
    )

    # Categorical encoded features
    kitchen_qual = st.sidebar.selectbox(
        "Kitchen Quality",
        [
            "Excellent",
            "Good",
            "Typical/Average",
            "Fair",
            "Poor",
        ],
    )
    bsmt_exposure = st.sidebar.selectbox(
        "Basement Exposure",
        [
            "No Exposure",
            "Minimum Exposure",
            "Average Exposure",
            "Good Exposure",
        ],
    )
    bsmt_fin_type1 = st.sidebar.selectbox(
        "Basement Finish Type",
        [
            "Unfinished",
            "Low Quality",
            "Rec Room",
            "Below Average Living Quarters",
            "Average Living Quarters",
            "Good Living Quarters",
        ],
    )
    garage_finish = st.sidebar.selectbox(
        "Garage Finish",
        [
            "Unfinished",
            "Rough Finished",
            "Finished",
        ],
    )

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

    # Prediction
    st.subheader("💰 Predicted Sale Price")

    if st.button("Predict Price"):
        prediction = rf_model.predict(transformed_df)[0]
        st.success(f"Estimated Sale Price: ${prediction:,.0f}")
        st.caption(
            """
            Prediction generated using the trained Random Forest model
            on the Ames housing dataset.
            """
        )
    else:
        st.info(
            """
            Adjust the inputs in the sidebar and click 'Predict Price'
            to see an estimate.
            """
        )


def show_summary_page():
    """
    Page: Project summary.
    """
    st.title("🏠 House Price Predictor")
    st.subheader("Project Summary")

    # Project Dataset
    st.subheader("📂 Project Dataset")
    st.markdown(
        """
        This project uses the Ames Housing Dataset, containing information
        for over 1,400 residential properties in Ames, Iowa. The dataset
        includes structural features, quality ratings and lot characteristics.
        """
    )

    # Business Requirements
    st.subheader("🧾 Business Requirements")
    st.markdown(
        """
        The client asked for:

        1. A study of which housing features are most strongly related
           to SalePrice.
        2. Predicted sale prices for four inherited houses.
        3. An interactive dashboard to explore data, insights,
           models and predictions.
        """
    )

    # README link
    st.subheader("📘 Check out Project README")
    st.markdown("Click below to open the full README file in GitHub:")
    st.link_button(
        "Open README",
        "https://github.com/aishieee/p5-heritage-housing"
        "?tab=readme-ov-file#readme",
    )

    # Dataset Guidelines
    st.subheader("📑 Dataset Guidelines")
    st.markdown("""
    - Numerical features include areas (sq ft), counts and quality ratings.
    - Categorical features (such as kitchen quality) use ordinal encodings.
    - Skewed features (LotArea, GrLivArea, TotalBsmtSF) were log-transformed.
    - Missing values were handled using domain-aware strategies.
    """)


def show_feature_insights_page():
    """
    Page: Feature importance and EDA plots.
    """
    st.title("📊 House Sales Price Study")

    train_df = load_train_data()
    importances = rf_model.feature_importances_

    importance_df = (
        pd.DataFrame(
            {
                "Feature": MODEL_FEATURES,
                "Importance": importances,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Client expectations
    st.subheader("🧾 Client Expectations")
    st.markdown(
        """
        The client wants to understand which housing features are most
        desirable in terms of increasing sale price. They are particularly
        interested in which variables to focus on when renovating or
        valuing properties.
        """
    )

    # Inspect training data
    with st.expander("🔍 Inspect a sample of the training data"):
        st.write(train_df.head(10))

    # Feature importance
    st.subheader("⭐ Most Influential Features for Sale Price")
    st.markdown(
        """
        The chart below shows the Random Forest feature importance scores.
        Higher values mean that the feature plays a bigger role in the
        model's price predictions.
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

        This confirms that overall property quality and the size of the
        living and basement areas are key drivers of sale price in the
        Ames housing market.
        """
    )

    # Plots: relationships with SalePrice
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

    # Bar: OverallQual vs median SalePrice
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

        - Houses with larger living areas and larger basements
          tend to sell for more.
        - As OverallQual increases, the median SalePrice rises sharply.

        This provides the client with clear visual evidence of which
        improvements are most likely to increase a property's market value.
        """
    )


def show_hypotheses_page():
    """
    Page: Project hypotheses and validation.
    """
    st.title("🧪 Project Hypotheses & Validation")

    st.markdown(
        """
        This page summarises the main hypotheses defined after the initial
        exploration of the Ames Housing dataset and explains how they were
        tested using exploratory data analysis (EDA) and machine learning.
        """
    )

    # Initial hypotheses
    st.subheader("📌 Initial Hypotheses")
    st.markdown(
        """
        1. Larger homes sell for higher prices. Homes with greater living area
           and larger overall footprint should achieve higher sale prices.

        2. Higher construction quality increases sale price. OverallQual is
           expected to be one of the strongest predictors of price.

        3. Homes with larger garages sell for higher prices. Bigger GarageArea
           and more garage spaces are expected to increase value.

        4. Newer or recently renovated homes are worth more. YearBuilt and
           YearRemodAdd should be positively associated with sale price.

        5. Smaller features have limited influence. Variables such as
           BedroomAbvGr, EnclosedPorch and OverallCond were expected to have
           weaker relationships with SalePrice.
        """
    )

    # How and why
    st.subheader("🔍 How the Hypotheses Were Tested and Why")
    st.markdown(
        """
        The project used several steps to test the hypotheses:

        1. Correlation analysis to identify linear relationships between
           numerical variables and SalePrice.

        2. Visual exploration with scatterplots and grouped summaries,
           such as median SalePrice by OverallQual.

        3. Feature engineering and log transformations for skewed features
           (GrLivArea, TotalBsmtSF, LotArea) to stabilise relationships.

        4. Two models: a Linear Regression baseline and a Random Forest
           Regressor as the main model to capture non-linear patterns.
        """
    )

    # Evidence from model performance
    st.subheader("📈 Evidence from Model Performance")
    st.markdown(
        """
        The models achieved the following performance (approximate values):

        - Linear Regression (scaled features):
          Train R² ≈ 0.79, Test R² ≈ 0.80,
          Test RMSE ≈ 39,600 and Test MAE ≈ 24,300.

        - Random Forest Regressor (unscaled features):
          Train R² ≈ 0.97, Test R² ≈ 0.89,
          Test RMSE ≈ 28,700 and Test MAE ≈ 17,800.

        The Random Forest reduces the errors compared with Linear Regression,
        suggesting that non-linear effects and interactions are important.
        """
    )

    # Evidence from feature importance
    st.subheader("📊 Evidence from Feature Importance")

    importances = rf_model.feature_importances_
    importance_df = (
        pd.DataFrame(
            {
                "Feature": MODEL_FEATURES,
                "Importance": importances,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    top_rows = importance_df.head(5)
    st.dataframe(top_rows)

    st.markdown(
        """
        OverallQual has the highest importance, confirming that construction
        and finish quality is the most influential factor.

        GrLivArea_log is the second most important feature, supporting the
        hypothesis that larger homes sell for more.

        TotalBsmtSF_log, GarageArea and LotArea_log also contribute, showing
        that basement size, garage size and lot size add value.
        """
    )

    # Final summary
    st.subheader("✅ Summary of Findings")
    st.markdown(
        """
    - H1: Larger homes sell for higher prices → Supported.
    - H2: Higher construction quality increases price → Strongly supported.
    - H3: Larger garages increase value → Supported.
    - H4: Newer or renovated homes are worth more → Partially supported.
    - H5: Smaller features have limited influence → Supported.

        Overall, the analysis shows that overall quality and property size
        are the primary drivers of house prices in Ames, with garage size,
        basement size and age acting as secondary factors.
        """
    )


def show_model_performance_page():
    """
    Page: Model performance and evaluation.
    """
    st.title("Model Performance & Evaluation")

    st.markdown(
        """
        This page summarises how well the models perform on unseen data and
        explains why the Random Forest Regressor was chosen as the final model.
        """
    )

    # Linear Regression summary (hard-coded metrics)
    st.subheader("📉 Linear Regression (baseline model)")
    st.markdown(
        """
        A Linear Regression model was trained on scaled engineered features.
        It provides a simple baseline assuming mostly linear relationships
        between the predictors and SalePrice.
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
        These results show that a basic linear model can already explain close
        to 80% of the variance in house prices, but the error values leave room
        for improvement, especially for more complex properties.
        """
    )

    # Random Forest evaluation
    st.subheader("🌲 Random Forest Regressor (final model)")

    test_df = load_test_data()
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["SalePrice"]

    y_pred_rf = rf_model.predict(X_test)

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

    st.table(
        rf_metrics.style.format(
            {
                "RMSE": "{:,.0f}",
                "MAE": "{:,.0f}",
                "R²": "{:.3f}",
            }
        )
    )

    st.markdown(
        """
        Compared with Linear Regression, the Random Forest:

    - Achieves a higher R² on the test set (close to 0.90).
    - Reduces RMSE and MAE, meaning predictions are closer to the
    true sale prices on average.
    - Captures non-linear effects and interactions between features.
        """
    )

    # Actual vs Predicted plot
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
        The points are clustered close to the diagonal, which indicates that
        the model predictions follow the actual sale prices closely. Larger
        deviations correspond to properties where the model under- or
        over-estimates the price.

        Overall, the evaluation confirms that the Random Forest Regressor is
        a suitable final model for this problem.
        """
    )


def main():
    """
    Main app entry point with sidebar navigation.
    """
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
