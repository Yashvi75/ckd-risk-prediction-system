import streamlit as st
import pickle
import numpy as np
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'models/model_optimized.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models/scaler_optimized.pkl'), 'rb'))
features = pickle.load(open(os.path.join(BASE_DIR, 'models/features_optimized.pkl'), 'rb'))

# Feature mapping
feature_mapping = {
    'specific gravity': 'sg',
    'hemoglobin': 'hemo',
    'hypertension': 'htn',
    'packed cell volume': 'pcv',
    'appetite': 'appet',
    'diabetes mellitus': 'dm',
    'albumin': 'al',
    'blood glucose random': 'bgr',
    'red blood cell count': 'rc',
    'blood urea': 'bu'
}

# Defaults
feature_defaults = {
    'specific gravity': 1.02,
    'hemoglobin': 13,
    'hypertension': 0,
    'packed cell volume': 40,
    'appetite': 1,
    'diabetes mellitus': 0,
    'albumin': 0,
    'blood glucose random': 100,
    'red blood cell count': 5,
    'blood urea': 40
}

st.title("CKD Risk Prediction")

# Inputs
data = {}

data['hemo'] = st.number_input("Hemoglobin", value=13.0)
data['sg'] = st.number_input("Specific Gravity", value=1.02)
data['htn'] = st.selectbox("Hypertension", [0,1])
data['pcv'] = st.number_input("Packed Cell Volume", value=40)

data['appet'] = st.selectbox("Appetite", [0,1])
data['dm'] = st.selectbox("Diabetes", [0,1])
data['al'] = st.number_input("Albumin", value=0)
data['bgr'] = st.number_input("Blood Glucose", value=100)
data['rc'] = st.number_input("RBC Count", value=5.0)
data['bu'] = st.number_input("Blood Urea", value=40)

if st.button("Predict"):
    input_data = []

    for feature in features:
        key = feature_mapping[feature]
        value = data.get(key, feature_defaults[feature])
        input_data.append(float(value))

    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"CKD Detected (Risk: {prob*100:.2f}%)")
    else:
        st.success(f"No CKD (Risk: {prob*100:.2f}%)")
