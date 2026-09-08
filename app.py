import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's details below. The model (Random Forest) will predict "
    "whether the patient is likely to have heart disease."
)

# ------------------------------------------------------------
# Load model + column order
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type", 
            options=[("Typical angina", 0), ("Atypical angina", 1),
                     ("Non-anginal pain", 2), ("Asymptomatic", 3)],
            format_func=lambda x: x[0]
        )[1]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)],
            format_func=lambda x: x[0]
        )[1]

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise-Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0]
        )[1]
        ca = st.selectbox("Number of Major Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia",
            options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)],
            format_func=lambda x: x[0]
        )[1]

    submitted = st.form_submit_button("Predict")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if submitted:
    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]  # ensure correct column order

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease  \nModel confidence: **{probability*100:.1f}%**")
    else:
        st.success(f"✅ Low risk of heart disease  \nModel confidence: **{(1-probability)*100:.1f}%**")

    with st.expander("See input summary"):
        st.dataframe(input_df)

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")
