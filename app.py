
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.inspection import permutation_importance

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. PROFESSIONAL UI
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #172554;
}
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stSidebar"] {
    background-color: #eaf0fb;
}
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. FILE PATHS AND LOADERS
# ==================================================

MODEL_PATH = "churn_model.pkl"

DATA_PATH = (
    r"C:\Users\Dell\Downloads\archive"
    r"\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def read_csv_file(path):
    return pd.read_csv(path)


@st.cache_data
def prepare_dataset(data):
    data = data.copy()

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    data = data.dropna(subset=["Churn"])

    return data


# ==================================================
# 4. SIDEBAR AND NAVIGATION
# ==================================================

st.sidebar.title("📊 Customer Churn AI")
st.sidebar.caption("Machine Learning Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Churn",
        "Feature Importance",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Predict customer churn, explore important features "
    "and evaluate machine learning models."
)

# ==================================================
# 5. LOAD MODEL AND DATA
# ==================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "churn_model.pkl file nahi mili. "
        "Project folder mein model file rakhein aur "
        "pehle python train_model.py run karein."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

df = None

if os.path.exists(DATA_PATH):
    try:
        df = prepare_dataset(read_csv_file(DATA_PATH))
    except Exception as e:
        st.sidebar.warning(f"Dataset load error: {e}")

if df is None:
    st.sidebar.warning("Default CSV path par dataset nahi mila.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Telco CSV dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            df = prepare_dataset(uploaded_df)
        except Exception as e:
            st.error(f"CSV read nahi hui: {e}")

# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

TARGET = "Churn"
ID_COLUMN = "customerID"


def make_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


def prepare_training_data(data):
    required = {"Churn", "customerID"}

    if not required.issubset(data.columns):
        raise ValueError(
            "CSV mein Churn aur customerID columns hone chahiye."
        )

    work = data.copy()
    work["TotalCharges"] = pd.to_numeric(
        work["TotalCharges"], errors="coerce"
    )

    y = work["Churn"].map({"Yes": 1, "No": 0})

    valid = y.notna()
    work = work.loc[valid].copy()
    y = y.loc[valid].astype(int)

    X = work.drop(columns=["Churn", "customerID"])

    # Match the feature structure expected by the saved model.
    return X, y


@st.cache_data
def calculate_model_evaluation(data):
    X, y = prepare_training_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    }

    results = {}
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Confusion Matrix": confusion_matrix(
                y_test, predictions
            ),
            "Predictions": predictions,
            "Probabilities": probabilities
        }

        fitted_models[name] = pipeline

    return (
        X_train, X_test, y_train, y_test,
        results, fitted_models
    )


def get_churn_probability(pipeline, customer):
    prediction = int(pipeline.predict(customer)[0])

    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(
            pipeline.predict_proba(customer)[0][1]
        )

    return prediction, probability


# ==================================================
# 7. DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.title("📊 Customer Churn Dashboard")
    st.write(
        "Customer retention aur churn patterns ka overview."
    )

    if df is None:
        st.warning(
            "Dashboard ke liye Telco CSV dataset upload karein "
            "sidebar se."
        )
    else:
        total_customers = len(df)
        churned = int(
            (df["Churn"].astype(str).str.strip() == "Yes").sum()
        )
        retained = total_customers - churned

        churn_rate = (
            churned / total_customers * 100
            if total_customers else 0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Churned Customers", f"{churned:,}")
        c3.metric("Retained Customers", f"{retained:,}")
        c4.metric("Churn Rate", f"{churn_rate:.1f}%")

        st.markdown("## Customer Analytics")

        left, right = st.columns(2)

        with left:
            st.subheader("Churn Distribution")

            counts = df["Churn"].value_counts().reindex(
                ["No", "Yes"], fill_value=0
            )

            fig, ax = plt.subplots()
            ax.bar(
                ["Retained", "Churned"],
                counts.values
            )
            ax.set_ylabel("Number of Customers")
            ax.set_title("Churn vs Retention")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Contract Type")

            if "Contract" in df.columns:
                contract_churn = pd.crosstab(
                    df["Contract"],
                    df["Churn"].astype(str).str.strip()
                )

                st.bar_chart(contract_churn)
            else:
                st.info("Contract column available nahi hai.")

        st.subheader("Monthly Charges by Churn")

        if "MonthlyCharges" in df.columns:
            charge_summary = df.groupby("Churn")[
                "MonthlyCharges"
            ].mean()

            st.bar_chart(charge_summary)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)


# ==================================================
# 8. PREDICT CHURN PAGE
# ==================================================

elif page == "Predict Churn":

    st.title("🔮 Predict Customer Churn")
    st.write(
        "Customer details enter karein aur prediction hasil karein."
    )

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with c2:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )
            protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with c3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0
            )
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn", use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:
            prediction, probability = get_churn_probability(
                model, customer
            )

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Model predicts that the customer may churn.")
            else:
                st.success("✅ Model predicts that the customer may stay.")

            if probability is not None:
                st.metric(
                    "Estimated Churn Probability",
                    f"{probability * 100:.1f}%"
                )
                st.progress(probability)

            st.caption(
                "Prediction estimate hai, guaranteed outcome nahi."
            )

        except Exception as e:
            st.error(
                f"Prediction error: {e}. "
                "Check karein ki model isi feature structure se train hua tha."
            )


# ==================================================
# 9. FEATURE IMPORTANCE PAGE
# ==================================================

elif page == "Feature Importance":

    st.title("🔍 Feature Importance")
    st.write(
        "Yeh analysis batata hai ki kisi feature ko shuffle karne se "
        "model ki ROC-AUC performance kitni badalti hai."
    )

    if df is None:
        st.warning(
            "Feature Importance ke liye sidebar se dataset CSV upload karein."
        )
    else:
        try:
            with st.spinner("Feature importance calculate ho rahi hai..."):

                (
                    X_train, X_test, y_train, y_test,
                    results, fitted_models
                ) = calculate_model_evaluation(df)

                selected_name = st.selectbox(
                    "Model select karein",
                    list(fitted_models.keys())
                )

                selected_model = fitted_models[selected_name]

                importance = permutation_importance(
                    selected_model,
                    X_test,
                    y_test,
                    scoring="roc_auc",
                    n_repeats=3,
                    random_state=42,
                    n_jobs=1
                )

                importance_df = pd.DataFrame({
                    "Feature": X_test.columns,
                    "Importance": importance.importances_mean
                })

                importance_df = importance_df.sort_values(
                    "Importance", ascending=False
                )

            st.subheader("Most Influential Features")

            top_n = st.slider(
                "Number of features",
                min_value=5,
                max_value=min(19, len(importance_df)),
                value=min(10, len(importance_df))
            )

            top_features = importance_df.head(top_n).sort_values(
                "Importance"
            )

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(
                top_features["Feature"],
                top_features["Importance"]
            )
            ax.set_xlabel("Decrease in ROC-AUC after shuffling")
            ax.set_title(f"{selected_name} - Permutation Importance")
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                importance_df.reset_index(drop=True),
                use_container_width=True
            )

            st.info(
                "Positive importance ka matlab hai ki feature shuffle "
                "karne par model performance ghati. Near-zero ya negative "
                "values ka matlab feature ka measured contribution kam "
                "ya unstable ho sakta hai. Yeh causation prove nahi karta."
            )

        except Exception as e:
            st.error(f"Feature importance error: {e}")


# ==================================================
# 10. MODEL EVALUATION PAGE
# ==================================================

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation Dashboard")
    st.write(
        "Dono machine learning models ko same test split par compare karein."
    )

    if df is None:
        st.warning(
            "Model Evaluation ke liye sidebar se dataset CSV upload karein."
        )
    else:
        try:
            with st.spinner("Models train aur evaluate ho rahe hain..."):

                (
                    X_train, X_test, y_train, y_test,
                    results, fitted_models
                ) = calculate_model_evaluation(df)

            summary = pd.DataFrame(results).T[
                ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
            ]

            summary_display = summary.style.format("{:.3f}")

            st.subheader("Model Comparison")
            st.dataframe(
                summary_display,
                use_container_width=True
            )

            chosen = st.selectbox(
                "Model ke detailed results dekhein",
                list(results.keys())
            )

            metrics = results[chosen]

            a, b, c, d, e = st.columns(5)
            a.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
            b.metric("Precision", f"{metrics['Precision']:.1%}")
            c.metric("Recall", f"{metrics['Recall']:.1%}")
            d.metric("F1-score", f"{metrics['F1-score']:.3f}")
            e.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

            st.subheader("Confusion Matrix")

            cm = metrics["Confusion Matrix"]

            fig, ax = plt.subplots()
            image = ax.imshow(cm)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Predicted Stay", "Predicted Churn"])
            ax.set_yticklabels(["Actual Stay", "Actual Churn"])
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Actual")
            ax.set_title(chosen)

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center"
                    )

            fig.colorbar(image, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("ROC Curve")

            probabilities = metrics["Probabilities"]
            fpr, tpr, _ = roc_curve(y_test, probabilities)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {metrics['ROC-AUC']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {chosen}")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                metrics["Predictions"],
                target_names=["Stay (0)", "Churn (1)"],
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).T.round(3),
                use_container_width=True
            )

            st.caption(
                "Evaluation page models ko same split aur training setup "
                "par dobara fit karta hai. Yeh saved model ki direct "
                "evaluation nahi, balki reproducible comparison hai."
            )

        except Exception as e:
            st.error(f"Model evaluation error: {e}")


# ==================================================
# 11. ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.title("ℹ️ About Customer Churn AI")

    st.markdown("""
    ### Project Objective
    Telecom customers ke churn risk ko Machine Learning se predict karna.

    ### Models
    - Logistic Regression
    - Random Forest

    ### Technologies
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit

    ### Features
    - Customer churn prediction
    - Customer statistics dashboard
    - Permutation-based feature importance
    - Model evaluation and comparison
    - Confusion matrix and ROC curve

    **Important:** Feature importance model behaviour ka measure hai;
    yeh apne aap causal relationship prove nahi karta.
    """)

    st.markdown("---")
    st.caption("Customer Churn AI | Machine Learning Project")
