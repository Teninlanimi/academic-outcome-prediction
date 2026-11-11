import streamlit as st
import joblib
import pandas as pd

# ==========================
# Load the complete pipeline and feature list
# ==========================
@st.cache_resource
def load_pipeline():
    model = joblib.load("multiclass_classification_model.pkl")
    # Load feature list from file to ensure exact column order
    with open("final_feature_list.txt", "r") as f:
        feature_columns = [line.strip() for line in f if line.strip()]
    return model, feature_columns

model, feature_columns = load_pipeline()

# ==========================
# Category codes (from the UCI "Predict Students' Dropout and Academic Success" dataset)
# ==========================
MARITAL_STATUS = {
    "Single": 1,
    "Married": 2,
    "Widower": 3,
    "Divorced": 4,
    "Facto union": 5,
    "Legally separated": 6,
}

COURSES = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (evening attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (evening attendance)": 9991,
}

NATIONALITIES = {
    "Portuguese": 1,
    "German": 2,
    "Spanish": 6,
    "Italian": 11,
    "Dutch": 13,
    "English": 14,
    "Lithuanian": 17,
    "Angolan": 21,
    "Cape Verdean": 22,
    "Guinean": 24,
    "Mozambican": 25,
    "Santomean": 26,
    "Turkish": 32,
    "Brazilian": 41,
    "Romanian": 62,
    "Moldovan": 100,
    "Mexican": 101,
    "Ukrainian": 103,
    "Russian": 105,
    "Cuban": 108,
    "Colombian": 109,
}

YES_NO = {"No": 0, "Yes": 1}
GENDER = {"Female": 0, "Male": 1}          # dataset encoding: 1 = male, 0 = female
ATTENDANCE = {"Daytime": 1, "Evening": 0}  # dataset encoding: 1 = daytime, 0 = evening

# Training encoding of the target: Enrolled = 0, Dropout = 1, Graduate = 2
LABEL_MAP = {0: "Enrolled", 1: "Dropout", 2: "Graduate"}


def select(label, options, default=None):
    """Show a dropdown of readable names and return the dataset code."""
    names = list(options.keys())
    index = names.index(default) if default in names else 0
    return options[st.selectbox(label, names, index=index)]


# ==========================
# Streamlit layout
# ==========================
st.set_page_config(page_title="Predict Academic Outcome", layout="centered")
st.title("Predict Academic Outcome")
st.markdown(
    "This app predicts whether a student is likely to **Graduate**, **Drop out**, "
    "or **Remain Enrolled**, based on academic, personal and economic information."
)

with st.form("academic_outcome_form"):
    user_inputs = {}
    for feature in feature_columns:
        f = feature.lower()
        if f == "marital status":
            user_inputs[feature] = select(feature, MARITAL_STATUS, "Single")
        elif f == "course":
            user_inputs[feature] = select(feature, COURSES, "Nursing")
        elif f == "daytime/evening attendance":
            user_inputs[feature] = select(feature, ATTENDANCE, "Daytime")
        elif f == "nacionality":
            user_inputs[feature] = select("Nationality", NATIONALITIES, "Portuguese")
        elif f in ["displaced", "educational special needs", "tuition fees up to date", "scholarship holder"]:
            user_inputs[feature] = select(feature, YES_NO)
        elif f == "gender":
            user_inputs[feature] = select(feature, GENDER)
        elif "enrolled" in f or "approved" in f:
            user_inputs[feature] = st.number_input(feature, min_value=0, max_value=30, value=6, step=1)
        elif "grade" in f:
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        elif f == "age at enrollment":
            user_inputs[feature] = st.number_input(feature, min_value=16, max_value=90, value=20, step=1)
        elif f == "unemployment rate":
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.8, step=0.1)
        elif f == "inflation rate":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=10.0, value=1.4, step=0.1)
        elif f == "gdp":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=10.0, value=1.74, step=0.01)
        else:
            user_inputs[feature] = st.number_input(feature, value=0.0)
    submitted = st.form_submit_button("Predict Outcome")

# ==========================
# Prediction
# ==========================
if submitted:
    # Keep the exact column order the model was trained on
    input_data = pd.DataFrame([user_inputs])[feature_columns]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    predicted_label = LABEL_MAP.get(int(prediction), str(prediction))
    class_names = [LABEL_MAP.get(int(c), str(c)) for c in model.classes_]

    st.success(f"Predicted Academic Outcome: **{predicted_label}**")

    st.markdown("### Prediction Probabilities")
    prob_df = pd.DataFrame({"Outcome": class_names, "Probability": probabilities})
    st.bar_chart(prob_df.set_index("Outcome"))
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================
# Load the complete pipeline and feature list
# ==========================
@st.cache_resource
def load_pipeline():
    # All files are in the same folder, so use the direct filename
    model = joblib.load("multiclass_classification_model.pkl")
    # Load feature list from file to ensure exact match
    with open('final_feature_list.txt', 'r') as f:
        feature_columns = [line.strip() for line in f if line.strip()]
    return model, feature_columns

model, feature_columns = load_pipeline()

# ==========================
# Streamlit Layout
# ==========================
st.set_page_config(page_title=" Predict Academic Outcome", layout="centered")
st.title(" Predict Academic Outcome")
st.markdown("""
This app predicts whether a student is likely to **Graduate**, **Dropout**, or **Remain Enrolled** 
based on key academic and personal information.
""")

# ==========================
# User Input Form
# ==========================

# Dynamically generate input fields matching training data encoding
with st.form("Academic Outcome Prediction Form"):
    user_inputs = {}
    for feature in feature_columns:
        feature_lower = feature.lower()
        # Match input types to training data encoding (numerical values)
        if feature_lower == "marital status":
            marital_options = {"Single": 1, "Married": 2, "Divorced": 3, "Widower": 4, "Divorced": 5, "Facto union": 6}
            marital_choice = st.selectbox(feature, list(marital_options.keys()))
            user_inputs[feature] = marital_options[marital_choice]
        elif feature_lower == "course":
            # Use number input for course ID (training data has course IDs like 171, 9254, etc.)
            user_inputs[feature] = st.number_input(feature + " (Course ID)", min_value=1, value=9254, step=1)
        elif feature_lower == "daytime/evening attendance":
            attendance_options = {"Daytime": 1, "Evening": 0}
            attendance_choice = st.selectbox(feature, list(attendance_options.keys()))
            user_inputs[feature] = attendance_options[attendance_choice]
        elif feature_lower == "nacionality":
            # Nationality is encoded numerically in training data
            user_inputs[feature] = st.number_input(feature + " (1=Nigerian, other=0)", min_value=1, value=1, step=1)
        elif feature_lower in ["displaced", "educational special needs", "tuition fees up to date", "scholarship holder"]:
            binary_options = {"No": 0, "Yes": 1}
            binary_choice = st.selectbox(feature, list(binary_options.keys()))
            user_inputs[feature] = binary_options[binary_choice]
        elif feature_lower == "gender":
            gender_options = {"Male": 0, "Female": 1}
            gender_choice = st.selectbox(feature, list(gender_options.keys()))
            user_inputs[feature] = gender_options[gender_choice]
        elif "enrolled" in feature_lower or "approved" in feature_lower:
            user_inputs[feature] = st.number_input(feature, min_value=0, max_value=30, value=0, step=1)
        elif "grade" in feature_lower:
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        elif feature_lower == "age at enrollment":
            user_inputs[feature] = st.number_input(feature, min_value=16, max_value=90, value=20, step=1)
        elif feature_lower == "unemployment rate":
            user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=50.0, value=10.8, step=0.1)
        elif feature_lower == "inflation rate":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=10.0, value=1.4, step=0.1)
        elif feature_lower == "gdp":
            user_inputs[feature] = st.number_input(feature, min_value=-10.0, max_value=10.0, value=1.74, step=0.01)
        else:
            # Default to number input for unknown numeric features
            user_inputs[feature] = st.number_input(feature, value=0.0)
    submitted = st.form_submit_button("Predict Outcome")

# ==========================
# Prediction Logic
# ==========================
if submitted:
    # Data is already in numerical format matching training data
    input_data = pd.DataFrame([user_inputs])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Map prediction to label based on training encoding
    # Training: Graduate=2, Dropout=1, Enrolled=0
    label_map = {0: "Enrolled", 1: "Dropout", 2: "Graduate"}
    predicted_label = label_map.get(prediction, str(prediction))
    
    # Get class names for display
    if hasattr(model, 'classes_'):
        class_names = [label_map.get(cls, str(cls)) for cls in model.classes_]
    else:
        class_names = ["Enrolled", "Dropout", "Graduate"]

    st.success(f" Predicted Academic Outcome: **{predicted_label}**")

    st.markdown("### Prediction Probabilities:")
    prob_df = pd.DataFrame({
        "Outcome": class_names,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Outcome"))
