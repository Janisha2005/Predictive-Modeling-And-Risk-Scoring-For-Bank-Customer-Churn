import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# Page settings
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# Get the folder where app.py is located
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "final_model.pkl"
RISK_FILE = BASE_DIR / "risk_predictions.csv"


# Load trained model
try:
    final_model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Unable to load final_model.pkl")
    st.exception(e)
    st.stop()


# Feature engineering
def create_features(data):

    data = data.copy()

    data["BalanceSalaryRatio"] = (
        data["Balance"] /
        (data["EstimatedSalary"] + 1)
    )

    data["ProductDensity"] = (
        data["NumOfProducts"] /
        (data["Tenure"] + 1)
    )

    data["EngagementProductInteraction"] = (
        data["IsActiveMember"] *
        data["NumOfProducts"]
    )

    data["AgeTenureInteraction"] = (
        data["Age"] *
        data["Tenure"]
    )

    return data


# Risk category
def get_risk_category(score):

    if score < 30:
        return "Low"

    elif score < 60:
        return "Moderate"

    elif score < 80:
        return "High"

    else:
        return "Very High"


# Sidebar
st.sidebar.title("🏦 Bank Churn Prediction")

page = st.sidebar.radio(
    "Select Page",
    [
        "Risk Calculator",
        "Risk Distribution",
        "Feature Information",
        "About Project"
    ]
)


# Main title
st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Predict the probability of customer churn and calculate "
    "an individual customer risk score."
)


# ==========================================================
# 1. RISK CALCULATOR
# ==========================================================

if page == "Risk Calculator":

    st.header("Customer Risk Calculator")

    st.write(
        "Enter the customer information and click the button "
        "to calculate the predicted churn risk."
    )

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            step=1
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=20,
            value=5,
            step=1
        )

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col2:

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        has_card = st.selectbox(
            "Has Credit Card",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        active_member = st.selectbox(
            "Is Active Member",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    st.divider()

    predict = st.button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

    if predict:

        customer = pd.DataFrame({
            "Year": [year],
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_card],
            "IsActiveMember": [active_member],
            "EstimatedSalary": [salary]
        })

        # Create the same engineered features used during training
        customer = create_features(customer)

        try:

            probability = final_model.predict_proba(
                customer
            )[0][1]

            risk_score = probability * 100

            risk_category = get_risk_category(
                risk_score
            )

            st.subheader("Prediction Result")

            result1, result2, result3 = st.columns(3)

            with result1:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with result2:

                st.metric(
                    "Risk Score",
                    f"{risk_score:.2f}"
                )

            with result3:

                st.metric(
                    "Risk Category",
                    risk_category
                )

            if risk_category == "Low":

                st.success(
                    "The predicted churn probability is relatively low."
                )

            elif risk_category == "Moderate":

                st.info(
                    "The customer has a moderate predicted churn risk."
                )

            elif risk_category == "High":

                st.warning(
                    "The customer has a high predicted churn risk."
                )

            else:

                st.error(
                    "The customer has a very high predicted churn risk."
                )

            # Risk score graph
            st.subheader("Risk Score")

            fig, ax = plt.subplots(figsize=(8, 2))

            ax.barh(
                ["Customer"],
                [risk_score]
            )

            ax.set_xlim(0, 100)
            ax.set_xlabel("Risk Score")
            ax.set_title("Customer Churn Risk Score")

            st.pyplot(fig)

            # Customer details
            st.subheader("Customer Information")

            customer_display = customer[
                [
                    "Year",
                    "CreditScore",
                    "Geography",
                    "Gender",
                    "Age",
                    "Tenure",
                    "Balance",
                    "NumOfProducts",
                    "HasCrCard",
                    "IsActiveMember",
                    "EstimatedSalary"
                ]
            ]

            st.dataframe(
                customer_display,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.exception(e)


# ==========================================================
# 2. RISK DISTRIBUTION
# ==========================================================

elif page == "Risk Distribution":

    st.header("Customer Risk Distribution")

    if not RISK_FILE.exists():

        st.warning(
            "risk_predictions.csv was not found."
        )

    else:

        risk_data = pd.read_csv(RISK_FILE)

        risk_order = [
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]

        risk_counts = (
            risk_data["RiskCategory"]
            .value_counts()
            .reindex(
                risk_order,
                fill_value=0
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Risk Category Counts")

            risk_table = pd.DataFrame({
                "Risk Category": risk_counts.index,
                "Customers": risk_counts.values
            })

            st.dataframe(
                risk_table,
                use_container_width=True,
                hide_index=True
            )

        with col2:

            st.subheader("Risk Distribution")

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            risk_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Risk Category")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Risk Distribution")
            ax.tick_params(
                axis="x",
                rotation=0
            )

            st.pyplot(fig)

        # Actual churn rate
        if "ActualExited" in risk_data.columns:

            st.divider()

            st.subheader(
                "Actual Churn Rate by Risk Category"
            )

            churn_rate = (
                risk_data
                .groupby("RiskCategory")["ActualExited"]
                .mean()
                .reindex(risk_order)
                * 100
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            churn_rate.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Risk Category")
            ax.set_ylabel("Actual Churn Rate (%)")
            ax.set_title(
                "Actual Churn Rate by Risk Category"
            )

            ax.tick_params(
                axis="x",
                rotation=0
            )

            st.pyplot(fig)

            churn_table = pd.DataFrame({
                "Risk Category": churn_rate.index,
                "Actual Churn Rate (%)":
                    churn_rate.values
            })

            st.dataframe(
                churn_table,
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# 3. FEATURE INFORMATION
# ==========================================================

elif page == "Feature Information":

    st.header("Features Used by the Model")

    features = pd.DataFrame({
        "Feature": [
            "Year",
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "BalanceSalaryRatio",
            "ProductDensity",
            "EngagementProductInteraction",
            "AgeTenureInteraction"
        ],

        "Description": [
            "Year of the customer record.",
            "Customer credit score.",
            "Customer geographical location.",
            "Customer gender.",
            "Customer age.",
            "Number of years with the bank.",
            "Customer account balance.",
            "Number of bank products used.",
            "Whether the customer has a credit card.",
            "Whether the customer is an active member.",
            "Estimated customer salary.",
            "Balance divided by estimated salary.",
            "Number of products relative to tenure.",
            "Interaction between active membership and products.",
            "Interaction between age and tenure."
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Risk Categories")

    risk_categories = pd.DataFrame({
        "Category": [
            "Low",
            "Moderate",
            "High",
            "Very High"
        ],

        "Risk Score": [
            "0 - 29",
            "30 - 59",
            "60 - 79",
            "80 - 100"
        ]
    })

    st.dataframe(
        risk_categories,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The risk thresholds are operational thresholds defined "
        "for this academic project."
    )


# ==========================================================
# 4. ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.header("About the Project")

    st.write(
        """
        This project focuses on predictive modeling and risk scoring
        for bank customer churn.

        The objective is to identify customers with a higher predicted
        probability of leaving the bank.
        """
    )

    st.subheader("Machine Learning Models")

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ]

    for model in models:
        st.write("• " + model)

    st.subheader("Evaluation Metrics")

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for metric in metrics:
        st.write("• " + metric)

    st.subheader("Explainability Methods")

    st.write(
        """
        The project uses feature importance, SHAP and Partial
        Dependence Plots to understand the factors associated with
        model predictions.
        """
    )

    st.subheader("Risk Scoring")

    st.write(
        """
        The predicted churn probability is converted into a score
        from 0 to 100.

        Low: 0-29

        Moderate: 30-59

        High: 60-79

        Very High: 80-100
        """
    )

    st.info(
        "Model predictions are based on historical data patterns "
        "and should be treated as decision-support information."
    )

    st.write(
        "Academic Project - Bank Customer Churn Prediction"
    )