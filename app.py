import streamlit as st
import os
from utils import analyze_text, predict_from_form

# ---- LLM ----
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ================= API KEY =================
# 🔴 REPLACE THIS WITH YOUR KEY (for local testing)
groq_api_key = "YOUR_GROQ_API_KEY"

# For deployment use:
# groq_api_key = os.getenv("GROQ_API_KEY")

# ================= UI =================
st.set_page_config(page_title="Mental Health AI", page_icon="🧠")
st.title("🧠 AI Mental Health Assistant")

st.write("API KEY FOUND:", bool(groq_api_key))  # debug

# ================= CHAT =================
user_input = st.text_input("Talk about how you're feeling:")

if user_input:

    # ---- LLM RESPONSE ----
    llm_response = ""

    if groq_api_key:
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama3-8b-8192"
            )

            prompt = ChatPromptTemplate.from_template(
                "You are a calm, supportive mental health assistant. Respond empathetically.\nUser: {input}"
            )

            chain = prompt | llm
            llm_response = chain.invoke({"input": user_input}).content

        except Exception as e:
            llm_response = f"❌ LLM Error: {e}"

    else:
        llm_response = "❌ API key not found"

    # ---- RISK ANALYSIS ----
    risk = analyze_text(user_input)

    # ---- OUTPUT ----
    st.subheader("💬 AI Response")
    st.write(llm_response)

    st.subheader("📊 Risk Analysis")
    if risk == "HIGH":
        st.error("🔴 High Risk")
    elif risk == "MEDIUM":
        st.warning("🟡 Moderate Risk")
    else:
        st.success("🟢 Low Risk")

# ================= FORM =================
st.subheader("📊 Mental Health Prediction")

age = st.slider("Age", 18, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox(
    "Work Interference",
    ["Never", "Rarely", "Sometimes", "Often"]
)

if st.button("Predict Risk"):
    result = predict_from_form(age, gender, family_history, work_interfere)

    if result == "Needs Treatment":
        st.error("⚠️ High Risk Detected")
    elif result == "Model Not Loaded":
        st.warning("⚠️ Model not loaded properly")
    else:
        st.success("✅ Low Risk")
