import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/random_forest_model.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title("🏦 AI Loan Approval Predictor")

st.write(
    "An interactive machine-learning system for "
    "loan approval analysis."
)

st.info(
    "This is an educational machine-learning prototype. "
    "The model prediction should not be used as the sole "
    "basis for real lending decisions."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🔍 Analysis Mode")

analysis_mode = st.sidebar.radio(
    "Choose an option:",
    [
        "Single Loan",
        "Multiple Loans"
    ]
)


# ==========================================
# SINGLE LOAN ANALYSIS
# ==========================================

if analysis_mode == "Single Loan":

    st.header("👤 Single Loan Analysis")

    st.write(
        "Enter the applicant's financial and personal "
        "details to generate an ML-based loan prediction."
    )

    st.divider()

    # ======================================
    # APPLICANT INFORMATION
    # ======================================

    st.subheader("📋 Applicant Information")

    col1, col2 = st.columns(2)

    # ======================================
    # LEFT COLUMN
    # ======================================

    with col1:

        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=2
        )

        education = st.selectbox(
            "Education",
            [
                "Graduate",
                "Not Graduate"
            ]
        )

        self_employed = st.selectbox(
            "Self Employed",
            [
                "Yes",
                "No"
            ]
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

    # ======================================
    # RIGHT COLUMN
    # ======================================

    with col2:

        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700,
            step=10
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

    # ======================================
    # CONVERT TEXT VALUES TO NUMBERS
    # ======================================

    education_value = 1 if education == "Graduate" else 0

    self_employed_value = 1 if self_employed == "Yes" else 0

    # ======================================
    # CREATE MODEL INPUT
    # ======================================

    input_data = pd.DataFrame(
        [
            {
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
            }
        ]
    )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    approval_probability = probabilities[1] * 100

    rejection_probability = probabilities[0] * 100

    # ======================================
    # PREDICTION RESULT
    # ======================================

    st.divider()

    st.subheader("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    # ======================================
    # RESULT
    # ======================================

    with result_col1:

        if prediction == 1:

            st.success("✅ APPROVED")

        else:

            st.error("❌ REJECTED")

    # ======================================
    # APPROVAL PROBABILITY
    # ======================================

    with result_col2:

        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )

    # ======================================
    # REJECTION PROBABILITY
    # ======================================

    with result_col3:

        st.metric(
            "Rejection Probability",
            f"{rejection_probability:.2f}%"
        )

    # ======================================
    # APPROVAL CONFIDENCE
    # ======================================

    st.write("### 📈 Approval Confidence")

    st.progress(
        int(approval_probability)
    )

    st.caption(
        "The percentage shown is the model-estimated "
        "probability for this input. It is not a guarantee "
        "of actual loan approval."
    )

    # ======================================
    # WHAT-IF CIBIL SIMULATOR
    # ======================================

    st.divider()

    st.header(" What-If CIBIL Simulator")

    st.write(
        "Change the CIBIL score below to see how the "
        "model's approval probability changes."
    )

    what_if_cibil = st.slider(
        "Adjust CIBIL Score",
        min_value=300,
        max_value=900,
        value=int(cibil_score),
        step=10
    )

    # Create a copy of the original input
    what_if_data = input_data.copy()

    # Change only CIBIL score
    what_if_data["cibil_score"] = what_if_cibil

    # Calculate new probability
    what_if_probabilities = model.predict_proba(
        what_if_data
    )[0]

    what_if_approval = (
        what_if_probabilities[1] * 100
    )

    what_if_rejection = (
        what_if_probabilities[0] * 100
    )

    # Calculate probability change
    probability_change = (
        what_if_approval - approval_probability
    )

    # ======================================
    # WHAT-IF RESULTS
    # ======================================

    st.subheader("📊 What-If Result")

    what_if_col1, what_if_col2, what_if_col3 = st.columns(3)

    # ======================================
    # ORIGINAL CIBIL
    # ======================================

    with what_if_col1:

        st.metric(
            "Original CIBIL",
            int(cibil_score)
        )

    # ======================================
    # NEW CIBIL
    # ======================================

    with what_if_col2:

        st.metric(
            "What-If CIBIL",
            what_if_cibil
        )

    # ======================================
    # NEW APPROVAL PROBABILITY
    # ======================================

    with what_if_col3:

        st.metric(
            "New Approval Probability",
            f"{what_if_approval:.2f}%",
            f"{probability_change:+.2f}%"
        )

    # ======================================
    # WHAT-IF STATUS
    # ======================================

    if what_if_probabilities[1] >= 0.5:

        st.success(
            "✅ Model prediction: APPROVED"
        )

    else:

        st.error(
            "❌ Model prediction: REJECTED"
        )

    # ======================================
    # WHAT-IF APPROVAL BAR
    # ======================================

    st.write("###  What-If Approval Confidence")

    st.progress(
        int(what_if_approval)
    )

    st.caption(
        f"At a CIBIL score of {what_if_cibil}, "
        f"the model estimates an approval probability "
        f"of {what_if_approval:.2f}%."
    )

    # ======================================
    # EXPLAINABLE AI
    # ======================================

    st.divider()

    st.header("🔍 Model Feature Importance")

    st.write(
        "This chart shows the relative importance of "
        "each input feature in the Random Forest model."
    )

    # ======================================
    # FEATURE IMPORTANCE DATA
    # ======================================

    feature_importance = pd.DataFrame(
        {
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
        }
    )

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=True
    )

    # ======================================
    # FEATURE IMPORTANCE CHART
    # ======================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "Feature importance shows how much the trained "
        "model relied on each feature. It does not mean "
        "that a feature causes loan approval or rejection."
    )


# ==========================================
# MULTIPLE LOANS
# ==========================================

elif analysis_mode == "Multiple Loans":

    st.header("Multiple Loan Analysis")

    st.write(
        "Upload a CSV file containing multiple loan "
        "applications to generate predictions for all applicants."
    )

    st.divider()

    st.subheader("Upload Loan Applications")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV containing the required loan application columns."
    )

    if uploaded_file is not None:

        try:

            # ======================================
            # READ CSV
            # ======================================

            bulk_data = pd.read_csv(uploaded_file)

            bulk_data.columns = bulk_data.columns.str.strip()

            st.success(
                f"File uploaded successfully — "
                f"{len(bulk_data)} applications found."
            )

            # ======================================
            # REQUIRED COLUMNS
            # ======================================

            required_columns = [
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
            ]

            # ======================================
            # CHECK MISSING COLUMNS
            # ======================================

            missing_columns = [
                column
                for column in required_columns
                if column not in bulk_data.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing these columns:"
                )

                st.write(missing_columns)

                st.info(
                    "Please upload a CSV containing all required columns."
                )

            else:

                st.success(
                    "CSV structure is valid."
                )

                # ==================================
                # DATA PREVIEW
                # ==================================

                st.subheader("Data Preview")

                st.dataframe(
                    bulk_data.head(10),
                    use_container_width=True
                )

                st.caption(
                    "Showing the first 10 applications."
                )

                # ==================================
                # ENCODE CATEGORICAL COLUMNS
                # ==================================

                bulk_data["education"] = (
                    bulk_data["education"]
                    .astype(str)
                    .str.strip()
                    .map({
                        "Graduate": 1,
                        "Not Graduate": 0
                    })
                )

                bulk_data["self_employed"] = (
                    bulk_data["self_employed"]
                    .astype(str)
                    .str.strip()
                    .map({
                        "Yes": 1,
                        "No": 0
                    })
                )

                # ==================================
                # CHECK INVALID VALUES
                # ==================================

                if bulk_data[required_columns].isnull().any().any():

                    st.error(
                        "Some values are missing or invalid."
                    )

                    st.info(
                        "Check Education values "
                        "(Graduate / Not Graduate) and "
                        "Self Employed values (Yes / No)."
                    )

                else:

                    # ==================================
                    # MODEL INPUT
                    # ==================================

                    model_input = bulk_data[
                        required_columns
                    ].copy()

                    # ==================================
                    # GENERATE PREDICTIONS
                    # ==================================

                    predictions = model.predict(
                        model_input
                    )

                    probabilities = model.predict_proba(
                        model_input
                    )

                    approval_probabilities = (
                        probabilities[:, 1] * 100
                    )

                    # ==================================
                    # ADD RESULTS
                    # ==================================

                    bulk_data["Prediction"] = [
                        "Approved"
                        if prediction == 1
                        else "Rejected"
                        for prediction in predictions
                    ]

                    bulk_data["Approval Probability (%)"] = (
                        approval_probabilities.round(2)
                    )

                    # ==================================
                    # PREDICTION RESULTS
                    # ==================================

                    st.divider()

                    st.subheader("Prediction Results")

                    st.dataframe(
                        bulk_data,
                        use_container_width=True,
                        height=500
                    )

                    # ==================================
                    # SUMMARY
                    # ==================================

                    st.divider()

                    st.subheader("Analysis Summary")

                    total_applications = len(
                        bulk_data
                    )

                    approved_count = (
                        bulk_data["Prediction"] == "Approved"
                    ).sum()

                    rejected_count = (
                        bulk_data["Prediction"] == "Rejected"
                    ).sum()

                    approval_rate = (
                        approved_count
                        / total_applications
                    ) * 100

                    average_probability = (
                        bulk_data[
                            "Approval Probability (%)"
                        ].mean()
                    )

                    # ==================================
                    # SUMMARY CARDS
                    # ==================================

                    summary_col1, summary_col2, summary_col3, summary_col4 = (
                        st.columns(4)
                    )

                    with summary_col1:

                        st.metric(
                            "Total Applications",
                            total_applications
                        )

                    with summary_col2:

                        st.metric(
                            "Approved",
                            approved_count
                        )

                    with summary_col3:

                        st.metric(
                            "Rejected",
                            rejected_count
                        )

                    with summary_col4:

                        st.metric(
                            "Approval Rate",
                            f"{approval_rate:.2f}%"
                        )

                    st.metric(
                        "Average Approval Probability",
                        f"{average_probability:.2f}%"
                    )

                    # ==================================
                    # APPROVAL VS REJECTION CHART
                    # ==================================

                    st.divider()

                    st.subheader(
                        "Approval vs Rejection"
                    )

                    chart_data = pd.DataFrame(
                        {
                            "Applications": [
                                approved_count,
                                rejected_count
                            ]
                        },
                        index=[
                            "Approved",
                            "Rejected"
                        ]
                    )

                    st.bar_chart(
                        chart_data,
                        height=350
                    )

                    # ==================================
                    # PROBABILITY DISTRIBUTION
                    # ==================================

                    st.subheader(
                        "Approval Probability Distribution"
                    )

                    fig, ax = plt.subplots(
                        figsize=(10, 4)
                    )

                    ax.hist(
                        bulk_data[
                            "Approval Probability (%)"
                        ],
                        bins=10
                    )

                    ax.set_xlabel(
                        "Approval Probability (%)"
                    )

                    ax.set_ylabel(
                        "Number of Applications"
                    )

                    ax.set_title(
                        "Distribution of Model-Estimated Approval Probabilities"
                    )

                    ax.set_xlim(
                        0,
                        100
                    )

                    plt.tight_layout()

                    st.pyplot(fig)

                    # ==================================
                    # DOWNLOAD RESULTS
                    # ==================================

                    st.divider()

                    st.subheader(
                        "Download Prediction Report"
                    )

                    csv_data = bulk_data.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="loan_prediction_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Unable to process the CSV file: {e}"
            )