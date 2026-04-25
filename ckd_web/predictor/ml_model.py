import pickle
import os
import numpy as np

# -----------------------------
# Load artifacts
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(BASE_DIR, '../models/model_optimized.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, '../models/scaler_optimized.pkl'), 'rb'))
features = pickle.load(open(os.path.join(BASE_DIR, '../models/features_optimized.pkl'), 'rb'))

# -----------------------------
# Mapping: Model feature → Form field
# -----------------------------
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

# -----------------------------
# Sensible defaults (IMPORTANT)
# -----------------------------
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

# -----------------------------
# Preprocess input
# -----------------------------
def preprocess_input(data_dict):
    processed = []

    for feature in features:
        form_key = feature_mapping[feature]

        value = data_dict.get(form_key)

        # Handle empty or missing input
        if value is None or value == "":
            value = feature_defaults[feature]

        try:
            value = float(value)
        except:
            value = feature_defaults[feature]

        processed.append(value)

    return processed

# -----------------------------
# Prediction function
# -----------------------------
def predict(data_dict):
    input_data = preprocess_input(data_dict)

    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Debug logs (you can remove later)
    print("FEATURE ORDER:", features)
    print("INPUT VECTOR:", input_data)
    print("SCALED INPUT:", input_scaled)
    print("PRED:", prediction, "PROB:", probability)

    return prediction, probability


# -----------------------------
# Debug test (optional)
# -----------------------------
def debug_static_test():
    test_input = {
        'sg': 1.02,
        'hemo': 14,
        'htn': 0,
        'pcv': 45,
        'appet': 1,
        'dm': 0,
        'al': 0,
        'bgr': 100,
        'rc': 5,
        'bu': 40
    }

    prediction, prob = predict(test_input)
    print("STATIC TEST →", prediction, prob)