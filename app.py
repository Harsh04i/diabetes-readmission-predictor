import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MediPredict | Healthcare Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background-color: #101D35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
}

.hero {
    background: linear-gradient(120deg, #102D50, #176B87);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #D8EAF5;
    font-size: 15px;
}

.section-title {
    color: #183153;
    font-size: 25px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #728096;
    font-size: 14px;
    margin-bottom: 22px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E5EAF2;
    box-shadow: 0 3px 12px rgba(20,40,70,0.04);
    min-height: 120px;
}

.metric-label {
    color: #75849A;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #183153;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.info-card {
    background: white;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #E5EAF2;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #183153;
    font-size: 18px;
}

.info-card p {
    color: #728096;
    font-size: 14px;
}

.result-card {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.result-number {
    font-size: 52px;
    font-weight: 800;
    color: #176B87;
}

.success-box {
    background: #E6F6EF;
    border-left: 5px solid #159568;
    padding: 20px;
    border-radius: 10px;
    color: #176B50;
}

.warning-box {
    background: #FFF3E5;
    border-left: 5px solid #E99A35;
    padding: 20px;
    border-radius: 10px;
    color: #965A15;
}

.danger-box {
    background: #FCEAEA;
    border-left: 5px solid #D94A4A;
    padding: 20px;
    border-radius: 10px;
    color: #9E3030;
}

.stButton > button {
    background: #176B87;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #124D69;
    color: white;
}

div[data-testid="stFormSubmitButton"] button {
    background: #176B87;
    color: white;
    font-weight: 700;
    border-radius: 9px;
    min-height: 48px;
}

.footer {
    text-align: center;
    color: #8A96A8;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND DATA
# =====================================================

@st.cache_resource
def load_model():

    bundle = joblib.load("readmission_model.pkl")

    return bundle


@st.cache_data
def load_json(path):

    if os.path.exists(path):

        with open(path, "r") as f:
            return json.load(f)

    return {}


@st.cache_data
def load_csv(path):

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


try:

    bundle = load_model()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    threshold = float(bundle["threshold"])

    features = bundle["features"]
    categorical_columns = bundle["categorical_columns"]
    numerical_columns = bundle["numerical_columns"]

    metrics = load_json("model_metrics.json")
    dataset_info = load_json("dataset_info.json")

    feature_importance = load_csv("feature_importance.csv")
    model_comparison = load_csv("model_comparison.csv")

except Exception as e:

    st.error(f"Unable to initialize application: {e}")
    st.stop()


# =====================================================
# DEFAULT VALUES
# =====================================================

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

            defaults[col] = str(value)

    except Exception:

        for col in numerical_columns:
            defaults[col] = 0.0

        for col in categorical_columns:
            defaults[col] = ""

    return defaults


defaults = get_default_values()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def flatten_metrics(obj, prefix=""):

    result = {}

    if isinstance(obj, dict):

        for key, value in obj.items():

            new_key = f"{prefix} {key}".strip()

            if isinstance(value, dict):
                result.update(flatten_metrics(value, new_key))

            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result[new_key] = value

    return result


def get_metric_value(flat_metrics, keywords):

    for key, value in flat_metrics.items():

        normalized = key.lower().replace("_", " ")

        if any(word in normalized for word in keywords):

            if 0 <= float(value) <= 1:
                return f"{float(value) * 100:.2f}%"

            return f"{float(value):.3f}"

    return "N/A"


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("# 🏥 MediPredict")

    st.caption("HEALTHCARE INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Patient Assessment",
            "Model Analytics",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.write(
        bundle.get("model_name", "Trained ML Model")
    )

    st.caption("UCI Diabetes 130-US Hospitals Dataset")

    st.divider()

    st.caption("Academic Project | 2026")


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">

        <h1>Predictive Healthcare Intelligence</h1>

        <p>
        An intelligent machine learning system designed to estimate
        the likelihood of hospital readmission within 30 days.
        </p>

        <p>
        Transforming historical patient encounter data into
        meaningful predictive insights.
        </p>

    </div>
    """, unsafe_allow_html=True)

    flat = flatten_metrics(metrics)

    st.markdown(
        '<div class="section-title">Model Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Performance of the trained prediction model</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Accuracy",
            get_metric_value(flat, ["accuracy"])
        )

    with c2:
        metric_card(
            "Precision",
            get_metric_value(flat, ["precision"])
        )

    with c3:
        metric_card(
            "Recall",
            get_metric_value(flat, ["recall"])
        )

    with c4:
        metric_card(
            "F1 Score",
            get_metric_value(flat, ["f1"])
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown(
            '<div class="info-card">'
            '<h3>🎯 Project Objective</h3>'
            '<p>Predict whether a patient encounter is likely to '
            'result in hospital readmission within 30 days. '
            'The system uses historical encounter details, '
            'previous visits, medications and clinical information.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-card">'
            '<h3>⚙️ Prediction Workflow</h3>'
            '<p>01. Enter patient encounter information</p>'
            '<p>02. Process input using the trained preprocessing pipeline</p>'
            '<p>03. Generate readmission probability</p>'
            '<p>04. Compare probability against the selected threshold</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="info-card">'
            '<h3>📊 Dataset Information</h3>'
            '<p><b>Dataset:</b> UCI Diabetes 130-US Hospitals</p>'
            '<p><b>Records:</b> 101,766 encounters originally</p>'
            '<p><b>Time period:</b> 1999–2008</p>'
            '<p><b>Target:</b> 30-day hospital readmission</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-card">'
            '<h3>🔬 Prediction Model</h3>'
            f'<p><b>Model:</b> {bundle.get("model_name", "Trained ML Model")}</p>'
            f'<p><b>Decision threshold:</b> {threshold:.3f}</p>'
            f'<p><b>Input features:</b> {len(features)}</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.info(
        "Select Patient Assessment from the sidebar to generate "
        "a new readmission prediction."
    )


# =====================================================
# PATIENT ASSESSMENT
# =====================================================

elif page == "Patient Assessment":

    st.markdown(
        '<div class="section-title">Patient Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter the encounter details to estimate 30-day readmission risk.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Default values are derived from the training data's "
        "imputation statistics. Replace them with meaningful "
        "patient encounter values before interpreting a prediction."
    )

    user_input = {}

    # Group fields by their names for easier navigation

    def classify_feature(feature):

        name = feature.lower()

        if any(x in name for x in [
            "race", "gender", "age", "weight"
        ]):
            return "Demographics"

        if any(x in name for x in [
            "admission", "discharge", "source", "payer",
            "encounter", "time_in_hospital"
        ]):
            return "Admission Details"

        if any(x in name for x in [
            "diag", "procedure", "lab", "glucose",
            "a1c", "number_diagnoses"
        ]):
            return "Clinical Information"

        return "Medical History & Medication"


    groups = {}

    for feature in features:

        category = classify_feature(feature)

        groups.setdefault(category, []).append(feature)


    with st.form("assessment_form"):

        for group_name, group_features in groups.items():

            st.markdown(f"### {group_name}")

            cols = st.columns(2)

            for index, feature in enumerate(group_features):

                col = cols[index % 2]

                default = defaults.get(feature, "")

                label = feature.replace("_", " ").title()

                with col:

                    if feature in categorical_columns:

                        value = st.text_input(
                            label,
                            value=str(default),
                            key=f"input_{feature}"
                        )

                    else:

                        value = st.number_input(
                            label,
                            value=float(default),
                            key=f"input_{feature}"
                        )

                    user_input[feature] = value

            st.divider()


        submitted = st.form_submit_button(
            "🔍 Generate Readmission Prediction",
            use_container_width=True
        )


    if submitted:

        try:

            input_df = pd.DataFrame(
                [user_input],
                columns=features
            )

            for col in numerical_columns:

                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce"
                )

            for col in categorical_columns:

                input_df[col] = input_df[col].astype(str)

            processed_input = preprocessor.transform(input_df)

            probability = model.predict_proba(
                processed_input
            )[0][1]

            prediction = int(probability >= threshold)

            st.session_state["prediction"] = probability
            st.session_state["decision"] = prediction
            st.session_state["patient_input"] = input_df

        except Exception as e:

            st.error(f"Prediction failed: {e}")


    # Display prediction

    if "prediction" in st.session_state:

        probability = st.session_state["prediction"]
        prediction = st.session_state["decision"]

        st.divider()

        st.markdown("## Prediction Results")

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                f"""
                <div class="result-card">

                    <p style="color:#728096;">
                    ESTIMATED READMISSION PROBABILITY
                    </p>

                    <div class="result-number">
                    {probability * 100:.2f}%
                    </div>

                    <p style="color:#728096;">
                    Probability generated by the trained model
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="result-card">

                    <p style="color:#728096;">
                    MODEL DECISION
                    </p>

                    <h2 style="color:#183153;">
                    {"Higher likelihood" if prediction else "Lower likelihood"}
                    </h2>

                    <p style="color:#728096;">
                    Threshold: {threshold:.3f}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(float(probability))

        if prediction:

            st.markdown(
                """
                <div class="warning-box">

                <b>Higher likelihood of readmission</b>

                The model estimates that this encounter has a
                probability at or above the selected decision threshold
                for readmission within 30 days.

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="success-box">

                <b>Lower likelihood of readmission</b>

                The model estimates that this encounter has a
                probability below the selected decision threshold
                for readmission within 30 days.

                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "This is an experimental academic prediction. "
            "It is not a clinical diagnosis or a validated "
            "medical risk assessment."
        )


# =====================================================
# MODEL ANALYTICS
# =====================================================

elif page == "Model Analytics":

    st.markdown(
        '<div class="section-title">Model Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Evaluation results and feature-level insights from the trained model.'
        '</div>',
        unsafe_allow_html=True
    )

    flat = flatten_metrics(metrics)

    st.markdown("### Final Model Evaluation")

    cols = st.columns(5)

    metric_list = [
        ("Accuracy", ["accuracy"]),
        ("Precision", ["precision"]),
        ("Recall", ["recall"]),
        ("F1 Score", ["f1"]),
        ("ROC-AUC", ["roc_auc", "roc auc"])
    ]

    for col, (label, keywords) in zip(cols, metric_list):

        with col:

            metric_card(
                label,
                get_metric_value(flat, keywords)
            )


    st.divider()

    if not model_comparison.empty:

        st.markdown("### Model Comparison")

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = model_comparison.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) > 0:

            st.markdown("### Performance Comparison")

            st.bar_chart(
                model_comparison.set_index(
                    model_comparison.columns[0]
                )[numeric_columns]
            )


    if not feature_importance.empty:

        st.divider()

        st.markdown("### Feature Importance")

        importance = feature_importance.copy()

        importance.columns = [
            str(c).strip() for c in importance.columns
        ]

        if "Feature" in importance.columns and "Importance" in importance.columns:

            importance = importance.sort_values(
                "Importance",
                ascending=False
            ).head(15)

            st.bar_chart(
                importance.set_index("Feature")["Importance"]
            )

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "Feature importance file has an unexpected column structure."
            )

    if not metrics:

        st.warning(
            "Model evaluation metrics could not be loaded."
        )


# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understanding the dataset, methodology and technology behind MediPredict.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <h3>Project Overview</h3>

        <p>
        MediPredict is an academic machine learning project that
        estimates the likelihood of hospital readmission within
        30 days using historical patient encounter data.
        </p>

        <p>
        The system uses patient demographics, admission details,
        previous hospital visits, laboratory information and
        medication-related attributes.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Dataset")

    st.write(
        "**UCI Diabetes 130-US Hospitals for Years 1999–2008**"
    )

    st.write(
        "The dataset contains historical hospital encounters "
        "involving patients with diabetes across 130 US hospitals."
    )

    st.write(
        "Original dataset size: approximately 101,766 encounters."
    )

    st.markdown(
        "[View the original UCI dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)"
    )

    st.divider()

    st.markdown("### Technology Stack")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="info-card">
            <h3>Python</h3>
            <p>Data processing and machine learning.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
            <h3>Scikit-learn</h3>
            <p>Preprocessing, model training and evaluation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">
            <h3>Streamlit</h3>
            <p>Interactive web application and deployment.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### Important Limitations")

    st.warning(
        "This project uses historical data from 1999–2008. "
        "Its predictions have not been clinically validated and "
        "must not be used to make real patient-care decisions. "
        "The output represents an experimental model estimate only."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">

    MEDIPREDICT | PREDICTIVE HEALTHCARE SYSTEM

    <br>

    Academic Machine Learning Project | 2026

    </div>
    """,
    unsafe_allow_html=True
)
