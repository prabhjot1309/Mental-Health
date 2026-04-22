import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from utils import (
    analyze_sentiment, detect_crisis_keywords,
    calculate_risk_score, generate_counseling_response
)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
load_dotenv()  # loads GROQ_API_KEY from .env locally

app = Flask(
    __name__,
    template_folder="templates",   # index.html lives here
    static_folder="static"         # style.css / script.js live here
)

# ─────────────────────────────────────────────
# INIT LLM CHAIN (once at startup)
# ─────────────────────────────────────────────
def init_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file or environment.")

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=400
    )

    prompt = ChatPromptTemplate.from_template("""
You are MindWell AI, a compassionate and professionally trained mental health support companion.

CORE PRINCIPLES:
- Validate the user's feelings before offering any advice ("I hear you", "That sounds really hard")
- Use empathy-first language; never minimize emotions
- Apply CBT techniques gently — challenge negative thoughts, not the person
- NEVER diagnose, prescribe, or claim to replace therapy
- If the user mentions suicide, self-harm, or is in danger -> immediately urge them to call a helpline
- End every response with one small, actionable step the user can take right now

User message: {input}

Respond with warmth and care (150-250 words):
""")

    return prompt | llm | StrOutputParser()


chain = init_chain()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receive a message from index.html, run NLP + LLM, return JSON.
    Expected request body: { "question": "user message here" }
    Response:             { "answer": "bot reply here" }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("question", "").strip():
        return jsonify({"answer": "I didn't catch that — could you say a little more?"}), 400

    user_input = data["question"].strip()

    # NLP analysis
    sentiment   = analyze_sentiment(user_input)
    crisis_flag = detect_crisis_keywords(user_input)
    risk_score  = calculate_risk_score(user_input)

    # Generate response
    response = generate_counseling_response(chain, user_input, sentiment, risk_score)

    return jsonify({
        "answer":   response,
        "sentiment": sentiment,
        "risk":      risk_score,
        "crisis":    crisis_flag
    })


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
