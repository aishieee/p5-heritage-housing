
## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In your Cloud IDE, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with *Regenerate API Key*.

## Introduction

This project is the final requirement for the Code Institute Diploma in Full Stack Software Development (Predictive Analytics).  
Its purpose is to support a client who wants to understand what makes a house valuable and to predict the sale prices of four homes they have inherited in Ames, Iowa.

The project uses the Ames Housing dataset from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) and applies data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning (ML) to build an accurate house price prediction model.  
The results are presented in an interactive Streamlit dashboard that allows the client to:

- explore key factors that influence house prices,
- view model performance,
- see predicted prices for their inherited properties, and
- generate real-time predictions for any house configuration.

A link to the deployed dashboard...

## Business Requirements

The client, Lydia Doe, recently inherited four residential properties in Ames, Iowa, from her great-grandfather.  
Although Lydia is familiar with property valuation in her home country of Belgium, she is unsure whether the same factors that influence property prices there also apply to the Iowan housing market. Because inaccurate pricing could lead to financial loss, she has asked for data-driven support.

Lydia located a publicly available dataset containing historical house prices for Ames, Iowa, and has provided this data for analysis. She requires a solution that can help her:

1. **Understand which house attributes are most strongly correlated with SalePrice.**  
   Lydia wants to identify what makes a house desirable and valuable in the Ames market.  
   She expects clear **data visualisations** demonstrating how key variables relate to the sale price, so she can understand which features contribute most to valuation.

2. **Predict the sale price of her four inherited houses, as well as any other house in Ames, Iowa.**  
   Lydia needs accurate price estimates for her inherited properties in order to maximise their sale value.  
   Additionally, she wants the ability to generate predictions for any future property she might consider buying or selling in Ames.

To meet these requirements, a ML regression model is developed and integrated into an interactive Streamlit dashboard that presents the insights and predictions in an accessible, user-friendly format.

## Dataset Content

This project uses the **Ames Housing Dataset**, originally compiled by Dean De Cock and made publicly available on [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).  
The dataset contains detailed structural, qualitative, and locational attributes for residential properties in Ames, Iowa, and is commonly used for house price prediction tasks.

### Dataset Overview

- **Rows:** ~1,460 observations  
- **Columns:** 79 variables  
- **Time span:** Houses built between **1872 and 2010**  
- **Target variable:** `SalePrice` (the final sale price of the property)

Each row represents a single house and describes features such as lot size, living area, basement finish type, kitchen quality, number of rooms, and garage characteristics.

### Key Variables Used in This Project

Although the full dataset contains 79 variables, this project focuses on the variables most relevant to the client’s business requirements.  
Below is a summary of the main features used in the modelling and dashboard analysis:

| Variable | Description | Typical Range / Categories |
|---------|-------------|---------------------------|
| **1stFlrSF** | First floor living area (sq ft) | 334 – 4692 |
| **2ndFlrSF** | Second floor living area (sq ft) | 0 – 2065 |
| **BedroomAbvGr** | Bedrooms above ground | 0 – 8 |
| **BsmtExposure** | Basement exposure / walkout type | Gd, Av, Mn, No |
| **BsmtFinType1** | Basement finish quality/type | GLQ, ALQ, BLQ, Rec, LwQ, Unf |
| **BsmtFinSF1** | Finished basement area (sq ft) | 0 – 5644 |
| **BsmtUnfSF** | Unfinished basement area (sq ft) | 0 – 2336 |
| **TotalBsmtSF** | Total basement area (sq ft) | 0 – 6110 |
| **GarageArea** | Garage size (sq ft) | 0 – 1418 |
| **GarageFinish** | Interior finish of garage | Fin, RFn, Unf |
| **GrLivArea** | Above-ground living area (sq ft) | 334 – 5642 |
| **KitchenQual** | Kitchen quality | Ex, Gd, TA, Fa, Po |
| **LotArea** | Lot size (sq ft) | 1,300 – 215,245 |
| **LotFrontage** | Street-facing linear feet | 21 – 313 |
| **MasVnrArea** | Masonry veneer area (sq ft) | 0 – 1600 |
| **EnclosedPorch** | Enclosed porch area (sq ft) | 0 – 286 |
| **OpenPorchSF** | Open porch area (sq ft) | 0 – 547 |
| **OverallCond** | Overall condition of house (1–10) | 1 – 10 |
| **OverallQual** | Overall material/finish quality (1–10) | 1 – 10 |
| **WoodDeckSF** | Wood deck area (sq ft) | 0 – 736 |
| **YearBuilt** | Original construction year | 1872 – 2010 |
| **YearRemodAdd** | Year remodeled (or year built if unchanged) | 1950 – 2010 |
| **SalePrice** | Final sale price (USD) | $34,900 – $755,000 |

### Why This Dataset Was Chosen

The Ames Housing dataset is suitable for this project because:

- It directly reflects the **Iowa property market**, aligning with the client’s needs.
- It contains enough historical data to train a reliable predictive model.
- It includes the types of variables the client is curious about (quality, size, basement, garage, living area).
- It is already publicly available and ethically safe to use.

### Variables Used for Modelling

After data cleaning and feature engineering, the following **10 engineered features** were used in the final Random Forest model:

- `GarageArea`  
- `OverallQual`  
- `OverallCond`  
- `KitchenQual` (ordinal encoded)  
- `BsmtExposure` (ordinal encoded)  
- `BsmtFinType1` (ordinal encoded)  
- `GarageFinish` (ordinal encoded)  
- `GrLivArea_log` (log-transformed)  
- `TotalBsmtSF_log` (log-transformed)  
- `LotArea_log` (log-transformed)

These features were selected based on correlation analysis, predictive power, and alignment with the client’s business questions.

## Hypothesis

The following hypothese were developed after inspecting and exploring the Ames Housing dataset (EDA) but before cleaning or modelling: 

* **Larger homes sell for higher prices.** - Homes with greater living area (`GrLivArea`, `1stFlrSF`, `TotalBsmtSF`) and larger overall footprint should achieve higher sale prices

* **Higher construction quality increases sale price.** - `OverallQual` (materials and finish quality) is expected to be one of the strongest predictors of price.

* **Homes with larger garages sell for higher prices.** - Homes with bigger `GarageArea` and `GarageCars` are expected to sell for more.

* **Newer or recently renovated homes are worth more.** - `YearBuilt` and `YearRemodAdd` should be positively associated with price as newer homes often require less maintenance.

* **Smaller features have limited influence** - Variables such as `BedroomAbvGr`, `EnclosedPorch` and `OverallCond` were expected to have weak correlations with price.

### How the Hypotheses will be validated:

Each hypothesis will be validated using a combination of exploratory data analysis (EDA), visualisations, and machine learning (ML) modelling.

1. **Correlation Analysis**
Pearson correlation coefficients and the correlation heatmap are used to test if key features (e.g., `GrLivArea`, `OverallQual`, `TotalBsmtSF`, `GarageArea`) show strong relationships with `SalePrice`.
This directly validates hypotheses about size, quality, and age.

2. **Visual Explorations**
Scatterplots, boxplots, and distribution plots will be used to visually confirm the strength of the relationships between important predictors and the sale price.
This helps validate whether trends expected in the hypotheses appear in the data.

3. **Feature Importance from the ML Model**
Once the regression model is trained, its feature importance values (e.g., coefficients for Linear Regression or feature importances from Random Forest) will indicate which features the model relies on most.
This validates whether the predicted “top features” are indeed useful in predicting sale price.

4. **Model Performance (R² Score)**
A model achieving R² ≥ 0.75 on both training and test data suggests that the chosen features and hypotheses align with real market behaviour.
If removing weak features improves the score, the hypotheses about feature relevance are further validated.

All these steps ensures conclusions are backed by both EDA insights and predictive modelling performance.


## The rationale to map the business requirements to the Data Visualisations and ML tasks

**Business Requirement 1** - Understand how house attributes correlate with `SalePrice`

To answer this requirement, the project uses conventional EDA techniques:
    - Correlation analysis to identify the strongest relationships between features and `SalePrice`.
    - Heatmaps to provide a visual overview of feature interactions.
    - Scatterplots & boxplots to show clear trends and validate assumptions.
    - Summary statistics & distributions to detect skewness, outliers, and variable behavior.

These visualisations help the client understand which attributes influence a property's value the most and satisfies their first requirement by showing the most relevant variables correlated to `SalePrice`.

**Business Requirement 2** - Predict the sale prices of the 4 inherited houses (and any other house in Ames)

To answer this requirement, the project implements a ML regression model:
    - Data cleaning and preprocessing ensure the model receives complete, reliable inputs.
    - Feature selection is guided by the EDA results from requirement 1.
    - Regression modelling (Linear Regression and Random Forest) is used to map feature relationships to `SalePrice`.
    - Model evaluation ensures performance meets the agreed metric of **R² ≥ 0.75** on both train and test sets.
    - Model inference allows prediction of `SalePrice` for the inherited houses and other inputs.

This satisfies the client’s second requirement to accurately predict house prices based on their attributes.

**Business Requirement 3** - Provide insights and predictions through an interactive dashboard

The project includes a Streamlit dashboard that allows the client to:
    - View EDA visualisations 
    - Explore relationships between features and `SalePrice`
    - Enter custom house features and generate predictions 
    - View predicted sale prices for all four inherited houses
    - Navigate results in a user-friendly interface

This ensures the project outcome is intuitive, transparent, and actionable for the client.

## CRISP-DM

The above business requirements map directly onto the CRISP-DM workflow:

| **CRISP-DM Phase**        | **How It Maps to the Project**                                   |
|---------------------------|------------------------------------------------------------------|
| Business Understanding    |     Identify client   goals: correlations + predictions          |
|     Data Understanding    |     EDA, correlation   analysis, visualisations                  |
|     Data Preparation      |     Cleaning, handling   missing values, encoding                |
|     Modelling             |     Regression model   training + optimisation                   |
|     Evaluation            |     R² ≥ 0.75   performance requirement                          |
|     Deployment            |     Streamlit   dashboard delivering insights and predictions    |


## ML Business Case

The objective of this project is to build a supervised machine learning system that predicts house sale prices in Ames, Iowa. The client will use this system to determine the combined value of 4 inherited properties and to evaluate any future properties.

#### Type of ML Task:

This is a supervised regression task because the **target variable `SalePrice`** is continuous and the model learns patterns from labelled hitorical data. 

#### Variables

**Dependent variable (Target):** 

- The variable we aim to predict is `SalePrice` of a house in USD. 
- This is the value the model outputs 

**Independent Variables (Features):**

- These are the inputs the model uses to learn and make predictions which are based on EDA findings and correlation analysis, the following features will be used:
   * Size-related features: `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `TotalBsmtSF` and `LotArea`.
   * Quality and condition features: `OverallQual` and `OverallCond`
   * Garage features: `GarageArea` and `GarageYrBlt`
   * Age features: `YearBuilt` and `YearRemodAdd`
   * Exterior/amenity features: `MasVnrArea`, `WoodDeckSF`, `OpenPorchSF` and `EnclosedPorch`.
   * Basement features: `BsmtFinSF1` and `BsmtUnfSF`
   * Other numerical features included after cleaning and feature selection.

These features provide the information the model needs to understand how house characteristics influence sale price.

**Output**

A single numerical value = the predicted house sale price 

## Business Value 

Building this regression model benefits the client by enabling:

- Accurate sale price estimates for the four inherited houses
- Evidence-based Understanding of which features influence property value the most
- Better decision-making when selling or investing in properties
- A tool to assess the value of any additional property through the dashboard

This ensures the client has a reliable, data-driven tool rather than relying on guesswork.

## Model Requirements

### The client defined success as:
    - R² score ≥ 0.75 on both the training and test sets
    - Accurate predictions for all 4 inherited houses
    - Confidence that the model generalises to unseen data
    - A deployed dashboard the client can use independently

### The model is considered unsuccessful if:
    - R² < 0.75
    - Predictions are unstable
    - The model is wrong by more than ~30% over a longer period
    - Insights do not match real market trends

## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the .python-version Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

