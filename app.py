import streamlit as st
import os
from datetime import datetime

# Local utils
from utils import (
    analyze_sentiment, detect_crisis_keywords, calculate_risk_score,
    generate_counseling_response
)

# LLM Setup
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindWell AI – Mental Health Companion",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Crisis banner */
    .crisis-banner {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 6px solid #c0392b;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px !important;
        border: 2px solid #d1d5db !important;
        padding: 12px 20px !important;
        font-size: 1rem !important;
    }
    .stButton > button {
        border-radius: 25px !important;
        height: 50px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }

    /* Risk badge */
    .risk-caption {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Disclaimer footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #9ca3af;
        font-size: 0.85rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIT LLM (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def init_llm():
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None
    try:
        llm = ChatGroq(
            api_key=api_key,
            model="llama3-8b-8192",
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
- If the user mentions suicide, self-harm, or is in danger → immediately provide crisis resources and urge them to call a helpline
- End every response with one small, actionable step the user can take right now

User message: {input}

Respond with warmth and care (150–250 words):
""")
        return prompt | llm | StrOutputParser()
    except Exception:
        return None


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = init_llm()
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False





# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
if not LLM_AVAILABLE:
    st.error("❌ Missing dependencies. Run: `pip install -r requirements.txt`")
    st.stop()

if not st.session_state.llm:
    st.error("❌ **GROQ_API_KEY not found.** Set it with: `export GROQ_API_KEY=your_key_here`")
    st.info("Get a free key at https://console.groq.com")
    st.stop()

st.markdown('<h1 class="main-header">🧠 MindWell AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A compassionate space to share how you\'re feeling</p>', unsafe_allow_html=True)

st.markdown("---")

# ── Chat history (shown ABOVE the input box) ──
chat_area = st.container()
with chat_area:
    if not st.session_state.messages:
        st.info("👋 Hi! I'm MindWell AI. Feel free to share what's on your mind — I'm here to listen.")
    else:
        for msg in st.session_state.messages[-20:]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
                    risk = msg.get("risk", 0)
                    risk_icon = "🔴" if risk > 0.7 else "🟡" if risk > 0.4 else "🟢"
                    st.markdown(
                        f'<p class="risk-caption">💭 Sentiment: {msg.get("sentiment","—")} &nbsp;|&nbsp; '
                        f'Wellbeing signal: {risk_icon} {risk:.0%} &nbsp;|&nbsp; {msg["timestamp"]}</p>',
                        unsafe_allow_html=True
                    )

            else:  # assistant
                with st.chat_message("assistant"):
                    # Show crisis banner only once per crisis message
                    if msg.get("crisis"):
                        st.markdown(
                            '<div class="crisis-banner">🚨 I noticed some concerning language. '
                            'Please reach out to a crisis helpline immediately — you are not alone.</div>',
                            unsafe_allow_html=True
                        )
                    st.write(msg["content"])
                    st.caption(msg["timestamp"])

# ── Input area (pinned below chat) ──
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    # Use a key that changes after clear to reset the widget
    input_key = "user_input_0" if not st.session_state.clear_input else "user_input_1"
    user_input = st.text_input(
        "Your message",
        key=input_key,
        placeholder="Share how you're feeling...",
        label_visibility="collapsed"
    )
with col2:
    send_btn = st.button("💬 Send", use_container_width=True)

# ── Process message ──
if send_btn and user_input.strip():
    text = user_input.strip()

    with st.spinner("MindWell is listening..."):
        sentiment      = analyze_sentiment(text)
        crisis_flag    = detect_crisis_keywords(text)
        risk_score     = calculate_risk_score(text)
        llm_response   = generate_counseling_response(
            st.session_state.llm, text, sentiment, risk_score
        )

    # Store user message
    st.session_state.messages.append({
        "role":      "user",
        "content":   text,
        "sentiment": sentiment,
        "risk":      risk_score,
        "crisis":    crisis_flag,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    # Store assistant message
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   llm_response,
        "crisis":    crisis_flag,
        "risk":      risk_score,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    # Toggle key to clear input field
    st.session_state.clear_input = not st.session_state.clear_input
    st.rerun()


# ── Clear chat ──
if st.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ⚠️ <strong>Disclaimer:</strong> MindWell AI is not a substitute for professional mental health care.<br>
    For emergencies, please call your local crisis hotline immediately.<br><br>
    Made with ❤️ for mental wellness awareness
</div>
""", unsafe_allow_html=True)
