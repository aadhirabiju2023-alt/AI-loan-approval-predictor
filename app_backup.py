import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")
# Feature importance from the trained Random Forest
feature_importance = pd.DataFrame({
    "Feature": [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ],
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)


# Page configuration
st.set_page_config(
    page_title="AI Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


# Title
st.title("🏦 AI Loan Approval Predictor")

st.write(
    "Enter the applicant's details below to get an ML-based "
    "loan approval prediction."
)

st.info(
    "This is an educational machine-learning prototype and "
    "should not be used as the sole basis for real lending decisions."
)


# -----------------------------
# USER INPUTS
# -----------------------------

st.header("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=2
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=5000000,
        step=100000
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=10000000,
        step=100000
    )

    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        max_value=30,
        value=10
    )


with col2:
    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=700
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value (₹)",
        min_value=0,
        value=10000000,
        step=100000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value (₹)",
        min_value=0,
        value=5000000,
        step=100000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value (₹)",
        min_value=0,
        value=10000000,
        step=100000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value (₹)",
        min_value=0,
        value=5000000,
        step=100000
    )


# Convert text inputs to numbers
education_value = 1 if education == "Graduate" else 0
self_employed_value = 1 if self_employed == "Yes" else 0


# Create input dataframe
input_data = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": education_value,
    "self_employed": self_employed_value,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value
}])


# -----------------------------
# PREDICTION
# -----------------------------

st.divider()

if st.button("🔮 Predict Loan Approval", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    approval_probability = probabilities[1] * 100

    st.header("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Status: APPROVED")
    else:
        st.error("❌ Loan Status: REJECTED")

    st.metric(
        "Approval Probability",
        f"{approval_probability:.2f}%"
    )

    st.progress(int(approval_probability))
        # -----------------------------
    # EXPLAINABLE AI
    # -----------------------------

    st.divider()

    st.header("🔍 Why does the model use these factors?")

    st.write(
        "The chart below shows the relative importance of each "
        "feature in the Random Forest model."
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "Feature importance shows how much the trained model relied "
        "on each feature. It does not mean that a feature causes "
        "loan approval or rejection."
    )
       