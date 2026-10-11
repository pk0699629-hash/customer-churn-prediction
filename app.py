
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROJECT FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")

DATA_PATH = os.path.join(
    BASE_DIR,
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #183153;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #475569;
    }

    .info-box {
        padding: 16px;
        border-radius: 10px;
        background-color: #eef4ff;
        border-left: 5px solid #2563eb;
        margin-bottom: 15px;
    }

    .success-box {
        padding: 16px;
        border-radius: 10px;
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        margin-bottom: 15px;
    }

    .danger-box {
        padding: 16px;
        border-radius: 10px;
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. EXPECTED CUSTOMER FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# =========================================================
# 5. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


# =========================================================
# 6. LOAD DATASET
# =========================================================

@st.cache_data
def read_csv_file(path):
    return pd.read_csv(path)


def prepare_dataset(data):
    """Clean the Telco customer churn dataset."""

    data = data.copy()

    # Remove spaces around column names
    data.columns = data.columns.str.strip()

    # Remove customer ID because it is not a useful predictor
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # Convert TotalCharges into numeric values
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce",
        )

    # Convert churn labels to numeric values for charts
    if "Churn" in data.columns:
        data["Churn"] = (
            data["Churn"]
            .astype(str)
            .str.strip()
            .replace({"Yes": 1, "No": 0})
        )

        data["Churn"] = pd.to_numeric(
            data["Churn"],
            errors="coerce",
        )

    return data


# =========================================================
# 7. LOAD MODEL AND DATASET
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file nahi mila. "
        "Please make sure churn_model.pkl "
        "app.py ke saath project folder mein hai."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Model load nahi ho saka: {error}")
    st.stop()


df = None

# First try loading the CSV from the project folder
if os.path.exists(DATA_PATH):
    try:
        df = prepare_dataset(read_csv_file(DATA_PATH))
    except Exception as error:
        st.sidebar.warning(
            f"Default CSV load nahi ho saka: {error}"
        )

# Optional fallback: upload CSV only if the bundled file is missing
if df is None:
    st.sidebar.warning(
        "Project folder mein default CSV nahi mili. "
        "Aap CSV upload kar sakte hain."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Telco CSV dataset",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = prepare_dataset(
                pd.read_csv(uploaded_file)
            )
            st.sidebar.success("CSV successfully loaded.")
        except Exception as error:
            st.sidebar.error(
                f"CSV read nahi ho saki: {error}"
            )


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Customer Churn AI")
st.sidebar.caption("Customer Retention Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Predict Churn",
        "Feature Importance",
        "Model Evaluation",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "This application uses a trained machine learning "
    "model to estimate customer churn."
)


# =========================================================
# 9. HELPER FUNCTIONS
# =========================================================

def get_churn_labels(data):
    """Return Yes/No churn labels for display."""

    labels = data["Churn"].map(
        {1: "Yes", 0: "No"}
    )

    return labels.fillna("Unknown")


def get_prediction_probability(estimator, row):
    """Return churn probability when predict_proba is available."""

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(row)

        if probabilities.ndim == 2:
            classes = getattr(
                estimator,
                "classes_",
                None,
            )

            if classes is not None:
                for index, class_value in enumerate(classes):
                    if str(class_value).strip().lower() in (
                        "1",
                        "yes",
                        "true",
                    ):
                        return float(probabilities[0][index])

            # Binary classifiers commonly put the positive class
            # in the second column.
            if probabilities.shape[1] >= 2:
                return float(probabilities[0][1])

    return None


def get_feature_importance(estimator):
    """
    Try to get feature importance from a model or pipeline.
    Returns a DataFrame or None.
    """

    model_object = estimator
    preprocessor = None

    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.items())

        if len(steps) > 0:
            model_object = steps[-1][1]

        for _, step in steps[:-1]:
            if hasattr(step, "get_feature_names_out"):
                preprocessor = step

    importance_values = None

    if hasattr(model_object, "feature_importances_"):
        importance_values = model_object.feature_importances_

    elif hasattr(model_object, "coef_"):
        coefficients = np.asarray(model_object.coef_)

        if coefficients.ndim == 2:
            importance_values = np.abs(
                coefficients[0]
            )
        else:
            importance_values = np.abs(coefficients)

    if importance_values is None:
        return None

    feature_names = None

    if preprocessor is not None:
        try:
            feature_names = (
                preprocessor.get_feature_names_out()
            )
        except Exception:
            feature_names = None

    if feature_names is None:
        feature_names = getattr(
            estimator,
            "feature_names_in_",
            None,
        )

    if feature_names is None:
        feature_names = [
            f"Feature {index + 1}"
            for index in range(len(importance_values))
        ]

    if len(feature_names) != len(importance_values):
        feature_names = [
            f"Feature {index + 1}"
            for index in range(len(importance_values))
        ]

    result = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    )

    return result.sort_values(
        "Importance",
        ascending=False,
    )


def get_evaluation_data(data):
    """Prepare features and target for basic model evaluation."""

    if data is None or "Churn" not in data.columns:
        return None, None

    available_features = [
        column
        for column in FEATURE_COLUMNS
        if column in data.columns
    ]

    if not available_features:
        return None, None

    clean_data = data.copy()
    clean_data = clean_data.dropna(
        subset=["Churn"]
    )

    y = clean_data["Churn"].astype(int)

    X = clean_data[available_features].copy()

    # Evaluation uses the dataset's available predictor columns.
    # The saved model must accept the same feature format.
    return X, y


# =========================================================
# 10. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        '📊 Customer Churn Dashboard'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Understand customer behaviour and churn patterns.'
        '</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning(
            "Dashboard dikhane ke liye dataset load nahi hua. "
            "WA_Fn-UseC_-Telco-Customer-Churn.csv ko "
            "app.py ke saath project folder mein rakhein "
            "aur GitHub par upload karein."
        )

    elif "Churn" not in df.columns:
        st.error(
            "Dataset mein 'Churn' column nahi mila."
        )

    else:
        total_customers = len(df)

        churn_count = int(
            (df["Churn"] == 1).sum()
        )

        retained_count = int(
            (df["Churn"] == 0).sum()
        )

        churn_rate = (
            churn_count / total_customers * 100
            if total_customers > 0
            else 0
        )

        average_monthly = (
            df["MonthlyCharges"].mean()
            if "MonthlyCharges" in df.columns
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            f"{total_customers:,}",
        )

        col2.metric(
            "Churned Customers",
            f"{churn_count:,}",
        )

        col3.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%",
        )

        col4.metric(
            "Avg. Monthly Charges",
            f"${average_monthly:,.2f}",
        )

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Customer Churn Distribution")

            churn_counts = (
                get_churn_labels(df)
                .value_counts()
                .reindex(
                    ["No", "Yes"],
                    fill_value=0,
                )
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.pie(
                churn_counts.values,
                labels=churn_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )

            ax.set_title("Churn vs Retained Customers")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Churn by Contract Type")

            if "Contract" in df.columns:
                contract_data = df.copy()

                contract_data["ChurnLabel"] = (
                    get_churn_labels(contract_data)
                )

                contract_table = pd.crosstab(
                    contract_data["Contract"],
                    contract_data["ChurnLabel"],
                )

                fig, ax = plt.subplots(figsize=(7, 4))

                contract_table.plot(
                    kind="bar",
                    ax=ax,
                )

                ax.set_xlabel("Contract Type")
                ax.set_ylabel("Number of Customers")
                ax.set_title("Contract and Churn")
                ax.tick_params(axis="x", rotation=20)
                ax.legend(title="Churn")

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Contract column available nahi hai.")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Monthly Charges Distribution")

            if "MonthlyCharges" in df.columns:
                fig, ax = plt.subplots(figsize=(7, 4))

                sns.histplot(
                    data=df,
                    x="MonthlyCharges",
                    hue="Churn" if "Churn" in df.columns else None,
                    kde=True,
                    ax=ax,
                )

                ax.set_title("Monthly Charges")
                st.pyplot(fig)
                plt.close(fig)

        with right:
            st.subheader("Customer Tenure")

            if "tenure" in df.columns:
                fig, ax = plt.subplots(figsize=(7, 4))

                sns.histplot(
                    data=df,
                    x="tenure",
                    hue="Churn" if "Churn" in df.columns else None,
                    bins=20,
                    kde=True,
                    ax=ax,
                )

                ax.set_title("Tenure Distribution")
                ax.set_xlabel("Tenure (months)")
                st.pyplot(fig)
                plt.close(fig)

        st.divider()

        st.subheader("Dataset Preview")
        st.dataframe(
            df.head(20),
            use_container_width=True,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Dataset CSV",
            data=csv_data,
            file_name="customer_churn_data.csv",
            mime="text/csv",
        )


# =========================================================
# 11. PREDICT CHURN PAGE
# =========================================================

elif page == "Predict Churn":

    st.markdown(
        '<div class="main-title">🔮 Predict Customer Churn</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Enter customer details to estimate churn risk.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Prediction saved model ke basis par hogi. "
        "Ye estimate hai, customer ke future behaviour ki guarantee nahi."
    )

    with st.form("churn_prediction_form"):

        st.subheader("Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

        with col2:
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda value: (
                    "Yes" if value == 1 else "No"
                ),
            )

        with col3:
            partner = st.selectbox(
                "Has Partner?",
                ["Yes", "No"],
            )

        col1, col2 = st.columns(2)

        with col1:
            dependents = st.selectbox(
                "Has Dependents?",
                ["Yes", "No"],
            )

        with col2:
            tenure = st.slider(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
            )

        st.divider()
        st.subheader("Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
            )

        with col2:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"],
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"],
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
            )

        with col3:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"],
            )

        st.divider()
        st.subheader("Billing and Contract")

        col1, col2, col3 = st.columns(3)

        with col1:
            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=1000.0,
                value=70.0,
                step=1.0,
            )

        with col3:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(monthly_charges * tenure),
                step=10.0,
            )

        submitted = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True,
        )

    if submitted:

        customer = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        try:
            # Use the same raw feature names expected by the model.
            if hasattr(model, "feature_names_in_"):
                expected_columns = list(
                    model.feature_names_in_
                )

                missing_columns = [
                    column
                    for column in expected_columns
                    if column not in customer.columns
                ]

                if missing_columns:
                    st.error(
                        "Model ko kuch extra features chahiye: "
                        + ", ".join(missing_columns)
                    )
                    st.stop()

                customer = customer[expected_columns]

            prediction = model.predict(customer)[0]

            probability = get_prediction_probability(
                model,
                customer,
            )

            is_churn = str(prediction).strip().lower() in (
                "1",
                "yes",
                "true",
            )

            if is_churn:
                st.markdown(
                    '<div class="danger-box">'
                    '<h3>⚠️ High Churn Concern</h3>'
                    'Model predicts that this customer may churn.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="success-box">'
                    '<h3>✅ Customer Likely to Stay</h3>'
                    'Model predicts that this customer may stay.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            if probability is not None:
                st.metric(
                    "Estimated Churn Probability",
                    f"{probability * 100:.2f}%",
                )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

            st.subheader("Customer Details")
            st.dataframe(
                customer,
                use_container_width=True,
            )

            if is_churn:
                st.subheader("Suggested Retention Actions")

                st.write(
                    "- Offer a suitable contract or retention plan."
                )
                st.write(
                    "- Review monthly charges and available discounts."
                )
                st.write(
                    "- Improve customer support and follow up."
                )
                st.write(
                    "- Check whether the customer needs additional services."
                )

        except Exception as error:
            st.error(
                "Prediction nahi ho saki. Check karein ki "
                "churn_model.pkl isi feature format ke saath train hua tha."
            )
            st.code(str(error))


# =========================================================
# 12. FEATURE IMPORTANCE PAGE
# =========================================================

elif page == "Feature Importance":

    st.markdown(
        '<div class="main-title">🎯 Feature Importance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Explore which model features contribute most to predictions.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        importance_df = get_feature_importance(model)

        if importance_df is None:
            st.warning(
                "Is saved model se direct feature importance "
                "available nahi hai. Agar model mein tree-based "
                "estimator ya linear coefficients hain, to yahan "
                "importance dikh sakti hai."
            )
        else:
            st.subheader("Most Important Features")

            top_n = st.slider(
                "Number of features",
                min_value=5,
                max_value=max(
                    5,
                    min(30, len(importance_df)),
                ),
                value=min(10, len(importance_df)),
            )

            top_features = importance_df.head(top_n)

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(
                data=top_features,
                x="Importance",
                y="Feature",
                ax=ax,
            )

            ax.set_title("Top Feature Importance")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Complete Importance Table")
            st.dataframe(
                importance_df,
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Download Feature Importance",
                data=importance_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="feature_importance.csv",
                mime="text/csv",
            )

    except Exception as error:
        st.error(
            f"Feature importance display nahi ho saki: {error}"
        )


# =========================================================
# 13. MODEL EVALUATION PAGE
# =========================================================

elif page == "Model Evaluation":

    st.markdown(
        '<div class="main-title">📈 Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Review performance metrics for the saved model.'
        '</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning(
            "Model evaluation ke liye dataset load nahi hua."
        )

    elif "Churn" not in df.columns:
        st.error("Dataset mein Churn target column nahi mila.")

    else:
        try:
            X, y = get_evaluation_data(df)

            if X is None or y is None:
                st.error(
                    "Dataset mein evaluation ke liye required "
                    "feature columns nahi mile."
                )
            else:
                # A separate holdout split gives a useful evaluation
                # only when the saved model accepts these raw columns
                # and was trained using compatible preprocessing.
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y if y.nunique() > 1 else None,
                )

                st.info(
                    "Metrics neeche dataset ke 20% holdout subset par "
                    "calculate ki jaati hain. Model ka input format "
                    "training ke format se match hona chahiye."
                )

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(
                    y_test,
                    y_pred,
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Holdout Accuracy",
                    f"{accuracy * 100:.2f}%",
                )

                probability = None

                if hasattr(model, "predict_proba"):
                    try:
                        probability = model.predict_proba(
                            X_test
                        )

                        if (
                            probability.ndim == 2
                            and probability.shape[1] >= 2
                            and y_test.nunique() == 2
                        ):
                            auc = roc_auc_score(
                                y_test,
                                probability[:, 1],
                            )

                            col2.metric(
                                "ROC-AUC",
                                f"{auc:.3f}",
                            )
                    except Exception:
                        col2.metric(
                            "ROC-AUC",
                            "Not available",
                        )
                else:
                    col2.metric(
                        "ROC-AUC",
                        "Not available",
                    )

                st.divider()
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Predicted No", "Predicted Yes"],
                    yticklabels=["Actual No", "Actual Yes"],
                    ax=ax,
                )

                ax.set_xlabel("Prediction")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")

                st.pyplot(fig)
                plt.close(fig)

                st.divider()
                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                    target_names=["No Churn", "Churn"],
                    output_dict=True,
                    zero_division=0,
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df.round(3),
                    use_container_width=True,
                )

                st.caption(
                    "Note: evaluation metrics meaningful tabhi hain "
                    "jab saved model ko training jaisa compatible "
                    "input data diya ja raha ho."
                )

        except Exception as error:
            st.error(
                "Evaluation run nahi ho saki. Saved model ki "
                "preprocessing aur dataset features verify karein."
            )
            st.code(str(error))


# =========================================================
# 14. ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About Project</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ## Customer Churn Prediction

        Customer Churn Prediction ek machine learning project hai
        jo customer ke available details ke basis par estimate karta
        hai ki customer service chhod sakta hai ya nahi.

        ### Project Objectives

        - Customer churn ko samajhna.
        - Customer data ko analyse karna.
        - Churn risk ka prediction karna.
        - Customer retention strategies mein help karna.

        ### Technologies Used

        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Matplotlib and Seaborn
        - Streamlit
        - Joblib

        ### Application Pages

        1. Dashboard
        2. Predict Churn
        3. Feature Importance
        4. Model Evaluation
        5. About Project

        ### Dataset

        Telco Customer Churn dataset.

        ### Important Note

        Predictions model ke output par based estimates hain.
        Inhe guaranteed outcomes ya professional business advice
        nahi samajhna chahiye.
        """
    )

    st.divider()

    st.subheader("Project Files")

    st.write(
        "- `app.py` — Streamlit application"
    )
    st.write(
        "- `churn_model.pkl` — trained model"
    )
    st.write(
        "- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset"
    )
    st.write(
        "- `requirements.txt` — Python dependencies"
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Customer Churn Prediction | Built with Python & Streamlit"
)
