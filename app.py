import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Predictive Healthcare System",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    font-size: 34px;
    font-weight: bold;
    color: #17365D;
}
.subtitle {
    font-size: 16px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "readmission_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feature_importance():
    if os.path.exists("feature_importance.csv"):
        return pd.read_csv("feature_importance.csv")
    return pd.DataFrame()

try:
    bundle = load_model()
    metrics = load_metrics()
    importance = load_feature_importance()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = bundle["threshold"]

    features = bundle["features"]
    categorical_columns = bundle["categorical_columns"]
    numerical_columns = bundle["numerical_columns"]

except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


# ---------------- DEFAULT INPUT VALUES ----------------

def get_default_values():

    defaults = {}

    try:
        num_pipeline = preprocessor.named_transformers_["num"]
        cat_pipeline = preprocessor.named_transformers_["cat"]

        num_imputer = num_pipeline.named_steps["imputer"]
        cat_imputer = cat_pipeline.named_steps["imputer"]

        for col, value in zip(
            numerical_columns,
            num_imputer.statistics_
        ):
            defaults[col] = float(value)

        for col, value in zip(
            categorical_columns,
            cat_imputer.statistics_
        ):
            defaults[col] = value

    except Exception:
        for col in features:
            defaults[col] = 0

    return defaults


defaults = get_default_values()


# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">🏥 Predictive Healthcare System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning-Based Hospital Readmission Prediction</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "This academic prototype estimates the probability of a hospital "
    "readmission within 30 days using historical patient encounter data."
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("About the Project")

st.sidebar.write(
    "This system uses a trained machine learning model "
    "to estimate 30-day hospital readmission risk."
)

st.sidebar.markdown("### Prediction Details")

st.sidebar.write("Dataset: UCI Diabetes 130-US Hospitals")
st.sidebar.write("Target: Readmission within 30 days")
st.sidebar.write(f"Selected model: {bundle.get('model_name', 'Trained ML Model')}")


# ---------------- PATIENT INPUT FORM ----------------

st.header("Patient Information")

st.write(
    "Enter the patient encounter details below. "
    "Default values are based on the training data."
)

user_input = {}

with st.form("patient_form"):

    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(features):

        column = col1 if index % 2 == 0 else col2

        default = defaults.get(feature, 0)

        with column:

            label = feature.replace("_", " ").title()

            if feature in categorical_columns:

                value = st.text_input(
                    label,
                    value=str(default),
                    key=feature
                )

            else:

                value = st.number_input(
                    label,
                    value=float(default),
                    key=feature
                )

            user_input[feature] = value

    submitted = st.form_submit_button(
        "Predict Readmission Risk",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submitted:

    try:

        input_df = pd.DataFrame(
            [user_input],
            columns=features
        )

        # Convert numerical fields
        for col in numerical_columns:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

        # Convert categorical fields
        for col in categorical_columns:
            input_df[col] = input_df[col].astype(str)

        # Apply preprocessing
        processed_input = preprocessor.transform(input_df)

        # Predict probability
        probability = model.predict_proba(processed_input)[0][1]

        prediction = int(probability >= threshold)

        st.divider()
        st.header("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Estimated Readmission Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Model Decision Threshold",
                f"{threshold * 100:.2f}%"
            )

        st.progress(float(probability))

        if prediction == 1:

            st.error(
                "The model predicts a higher likelihood of "
                "readmission within 30 days."
            )

        else:

            st.success(
                "The model predicts that readmission within "
                "30 days is less likely."
            )

        st.caption(
            "This is a model-generated estimate, not a clinical diagnosis "
            "or a validated medical risk assessment."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ---------------- MODEL PERFORMANCE ----------------

st.divider()

st.header("Model Performance")

if metrics:

    st.json(metrics)

else:

    st.warning("Model evaluation metrics are not available.")

# ---------------- FEATURE IMPORTANCE ----------------

if not importance.empty:

    st.header("Important Prediction Features")

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    st.bar_chart(
        importance.set_index("Feature")["Importance"]
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Academic Project | Predictive Healthcare System "
    "| UCI Diabetes 130-US Hospitals Dataset"
)
