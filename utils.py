import pickle
import numpy as np
import os

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

# ---------- DEBUG (IMPORTANT) ----------
print("📂 Current directory:", BASE_DIR)
print("📁 Files available:", os.listdir(BASE_DIR))
print("📌 Model path:", model_path)

# ---------- LOAD MODEL SAFELY ----------
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None  # prevent crash


# ---------- TEXT ANALYSIS ----------
def analyze_text(text):
    keywords = ["stress", "anxiety", "depressed", "tired", "overwhelmed"]
    score = sum([1 for k in keywords if k in text.lower()])

    if score >= 2:
        return "HIGH"
    elif score == 1:
        return "MEDIUM"
    return "LOW"


# ---------- ML PREDICTION ----------
def predict_from_form(age, gender, family_history, work_interfere):
    if model is None:
        return "Model Not Loaded"

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

    try:
        prediction = model.predict(data)[0]
        return "Needs Treatment" if prediction == 1 else "Low Risk"
    except Exception as e:
        print("❌ Prediction error:", e)
        return "Prediction Error"
