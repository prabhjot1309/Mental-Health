### 🧠 MindCare – Mental Health Chatbot & Risk Predictor


🚀 Overview

MindCare Chatbot is an AI-powered web application that analyzes user input and predicts mental health conditions using machine learning. It also provides conversational support through an interactive chatbot interface.

This project combines ML-based prediction + chatbot interaction into a single system.

### 🎯 Key Features
🧠 Mental health prediction using trained ML model
💬 Chatbot-based interaction system
📊 Input-based prediction via form & text analysis
⚡ Fast inference using pre-trained .pkl model
🌐 Web interface using Flask + HTML templates
📁 Dataset-driven insights (survey.csv)

### 🏗️ Tech Stack
Frontend: HTML 
Backend: Python (Flask)
Machine Learning: Scikit-learn , Pandas , NumPy , TF-IDF / text preprocessing

### 📂 Project Structure (REAL – based on your repo)
Mindcare-Chatbot/
│── app.py                         # Main Flask app
│── utils.py                       # Prediction + processing logic
│── model.pkl                      # Trained ML model
│── survey.csv                     # Dataset
│── mental_health_prediction.ipynb # Model training notebook
│── templates/
│   └── index.html                # Frontend UI
│── .streamlit/                   # Streamlit config (optional/unused mix)
│── .devcontainer/                # Dev environment setup
│── requirements.txt
│── README.md
