import pickle
import numpy as np
import os

# Get correct path (important for Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

# Debug (optional – remove later)
print("Files in directory:", os.listdir(BASE_DIR))

# Load model safely
model = pickle.load(open(model_path, "rb"))

# -------- TEXT ANALYSIS --------
def analyze_text(text):
    keywords = ["stress", "anxiety", "depressed", "tired", "overwhelmed"]
    score = sum([1 for k in keywords if k in text.lower()])

    if score >= 2:
        return "HIGH"
    elif score == 1:
        return "MEDIUM"
    return "LOW"

# -------- ML PREDICTION --------
def predict_from_form(age, gender, family_history, work_interfere):
    gender = 1 if gender == "Male" else 0
    family_history = 1 if family_history == "Yes" else 0

    mapping = {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3
    }
    work_interfere = mapping.get(work_interfere, 0)

    data = np.array([[age, gender, family_history, work_interfere]])

    prediction = model.predict(data)[0]

    return "Needs Treatment" if prediction == 1 else "Low Risk"
