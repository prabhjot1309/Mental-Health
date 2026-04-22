import streamlit as st
from utils import analyze_text, predict_from_form

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

st.title("🧠 Mental Health AI Assistant")

# ---------------- CHAT ----------------
st.subheader("💬 Chat")

user_input = st.text_input("Talk about how you're feeling:")

if user_input:
    risk = analyze_text(user_input)

    if risk == "HIGH":
        st.error("🔴 High Stress Detected")
        st.write("You may be experiencing high stress. Consider talking to a professional.")
    elif risk == "MEDIUM":
        st.warning("🟡 Moderate Stress")
        st.write("You might be under stress. Try relaxation techniques and take breaks.")
    else:
        st.success("🟢 Low Stress")
        st.write("You seem okay. Keep maintaining a healthy routine.")

    st.info(f"Risk Level: {risk}")

# ---------------- FORM ----------------
st.subheader("📊 Mental Health Prediction")

age = st.slider("Age", 18, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox(
    "Work Interference",
    ["Never", "Rarely", "Sometimes", "Often"]
)

if st.button("Predict"):
    result = predict_from_form(age, gender, family_history, work_interfere)

    if result == "Needs Treatment":
        st.error("⚠️ Needs Mental Health Support")
    elif result == "Model Not Loaded":
        st.warning("⚠️ Model not loaded properly")
    else:
        st.success("✅ Low Risk")
