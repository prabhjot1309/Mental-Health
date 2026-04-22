import streamlit as st
import os
from datetime import datetime

from utils import (
    analyze_sentiment, detect_crisis_keywords,
    calculate_risk_score, generate_counseling_response
)

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
    page_title="MindCare – Mental Health Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# CSS — matches index.html exactly
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Fraunces:wght@700;900&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; max-width: 860px !important; }

    /* Page */
    .stApp { background-color: #0d1117; font-family: 'DM Sans', sans-serif; }

    /* ── Header ── */
    .mc-header {
        text-align: center;
        padding: 10px 0 6px;
    }
    .mc-header h1 {
        font-family: 'Fraunces', serif;
        font-size: 2.4rem;
        font-weight: 900;
        color: #6ee7b7;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .mc-header p {
        color: #6b7280;
        font-size: 0.92rem;
        margin: 4px 0 0;
    }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22 !important;
        border-radius: 14px 14px 0 0 !important;
        border-bottom: 1px solid #2d3748 !important;
        gap: 0 !important;
        padding: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #6b7280 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        padding: 14px 32px !important;
        border-radius: 0 !important;
        border: none !important;
        flex: 1;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        color: #6ee7b7 !important;
        border-bottom: 2px solid #6ee7b7 !important;
        background: #1f2937 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #161b22 !important;
        border-radius: 0 0 14px 14px !important;
        border: 1px solid #2d3748 !important;
        border-top: none !important;
        padding: 0 !important;
    }

    /* ── Chat messages area ── */
    .chat-scroll {
        min-height: 380px;
        max-height: 420px;
        overflow-y: auto;
        padding: 20px 20px 8px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .chat-scroll::-webkit-scrollbar { width: 4px; }
    .chat-scroll::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }

    /* ── Bubbles ── */
    .bubble {
        max-width: 78%;
        padding: 12px 16px;
        border-radius: 14px;
        font-size: 0.92rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .bubble .sender {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 5px;
        opacity: 0.65;
    }
    .bubble.bot {
        background: #1a2e1f;
        color: #bbf7d0;
        border-bottom-left-radius: 4px;
        align-self: flex-start;
    }
    .bubble.user {
        background: #1e3a5f;
        color: #bfdbfe;
        border-bottom-right-radius: 4px;
        align-self: flex-end;
        margin-left: auto;
    }
    .bubble.crisis {
        background: #450a0a;
        border: 1px solid #f87171;
        color: #fecaca;
    }
    .risk-meta {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 5px;
    }

    /* ── Input row ── */
    .input-row {
        display: flex;
        gap: 10px;
        padding: 14px 20px;
        border-top: 1px solid #2d3748;
        background: #161b22;
        border-radius: 0 0 14px 14px;
    }
    .stTextInput > div > div > input {
        background: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 25px !important;
        color: #e2e8f0 !important;
        padding: 11px 18px !important;
        font-size: 0.93rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6ee7b7 !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: #4b5563 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #6ee7b7 !important;
        color: #0d1117 !important;
        border: none !important;
        border-radius: 25px !important;
        height: 48px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.82 !important; }

    /* ── Risk form ── */
    .risk-wrap { padding: 24px 20px; }
    .disclaimer-box {
        background: #1c1a0e;
        border: 1px solid #78350f;
        border-radius: 10px;
        padding: 12px 16px;
        color: #fbbf24;
        font-size: 0.85rem;
        margin-bottom: 20px;
    }
    .stSlider > div { padding: 4px 0 !important; }
    label { color: #d1d5db !important; font-size: 0.9rem !important; }

    /* Risk result */
    .risk-low    { background:#0f291a; border:1px solid #166534; color:#bbf7d0; border-radius:12px; padding:16px 20px; margin-top:12px; }
    .risk-medium { background:#1c1a0e; border:1px solid #854d0e; color:#fde68a; border-radius:12px; padding:16px 20px; margin-top:12px; }
    .risk-high   { background:#450a0a; border:1px solid #f87171; color:#fecaca; border-radius:12px; padding:16px 20px; margin-top:12px; }
    .risk-low h3, .risk-medium h3, .risk-high h3 {
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        margin-bottom: 6px;
    }

    /* Divider */
    hr { border-color: #2d3748 !important; margin: 8px 0 !important; }

    /* Footer */
    .mc-footer {
        text-align: center;
        color: #374151;
        font-size: 0.8rem;
        padding: 20px 0 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIT LLM
# ─────────────────────────────────────────────
@st.cache_resource
def init_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            return None
    try:
        llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant",
                       temperature=0.3, max_tokens=400)
        prompt = ChatPromptTemplate.from_template("""
You are MindCare AI, a compassionate mental health support companion.
- Always validate feelings first ("I hear you", "That sounds really hard")
- Use empathy-first language; never minimize emotions
- Apply CBT techniques gently
- NEVER diagnose or prescribe
- If crisis/self-harm mentioned → urge helpline immediately
- End with one small actionable step

User message: {input}
Respond warmly (150–250 words):
""")
        return prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
        return None


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = init_llm()
if "flip" not in st.session_state:
    st.session_state.flip = False

if not LLM_AVAILABLE:
    st.error("❌ Run: pip install -r requirements.txt")
    st.stop()
if not st.session_state.llm:
    st.error("❌ GROQ_API_KEY missing — add it to Streamlit secrets.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="mc-header">
    <h1>🧠 MindCare</h1>
    <p>Mental Health Chatbot &amp; Risk Predictor</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_chat, tab_risk = st.tabs(["💬 Chat", "⚠️ Risk Assessment"])


# ══════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════
with tab_chat:

    # Build chat HTML
    bubbles_html = ""
    if not st.session_state.messages:
        bubbles_html = """
        <div class="bubble bot">
            <div class="sender">MindCare</div>
            Hello! I'm here to listen and support you. How are you feeling today? 💙
        </div>"""
    else:
        for msg in st.session_state.messages[-20:]:
            if msg["role"] == "user":
                bubbles_html += f"""
                <div class="bubble user">
                    <div class="sender">You</div>
                    {msg["content"]}
                    <div class="risk-meta">
                        {msg.get("sentiment","—")} &nbsp;·&nbsp;
                        {"🔴" if msg.get("risk",0)>0.7 else "🟡" if msg.get("risk",0)>0.4 else "🟢"}
                        {msg.get("risk",0):.0%} &nbsp;·&nbsp; {msg["timestamp"]}
                    </div>
                </div>"""
            else:
                crisis_html = ""
                if msg.get("crisis"):
                    crisis_html = """<div style="font-size:0.82rem;margin-bottom:8px;
                        padding:8px 12px;background:#7f1d1d;border-radius:8px;color:#fecaca;">
                        🚨 Please reach out to a crisis helpline — you are not alone.</div>"""
                bubbles_html += f"""
                <div class="bubble bot {"crisis" if msg.get("crisis") else ""}">
                    <div class="sender">MindCare</div>
                    {crisis_html}{msg["content"]}
                    <div class="risk-meta">{msg["timestamp"]}</div>
                </div>"""

    st.markdown(f'<div class="chat-scroll" id="chat-end">{bubbles_html}</div>',
                unsafe_allow_html=True)

    # Auto-scroll to bottom
    st.markdown("""
    <script>
        const el = document.getElementById('chat-end');
        if(el) el.scrollTop = el.scrollHeight;
    </script>""", unsafe_allow_html=True)

    # Input row
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        key = f"ci_{st.session_state.flip}"
        user_input = st.text_input("msg", key=key,
                                   placeholder="Share how you're feeling...",
                                   label_visibility="collapsed")
    with col2:
        send = st.button("Send", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Process send
    if send and user_input.strip():
        text = user_input.strip()
        with st.spinner("MindCare is listening..."):
            sentiment   = analyze_sentiment(text)
            crisis_flag = detect_crisis_keywords(text)
            risk_score  = calculate_risk_score(text)
            response    = generate_counseling_response(
                st.session_state.llm, text, sentiment, risk_score)

        st.session_state.messages.append({
            "role": "user", "content": text,
            "sentiment": sentiment, "risk": risk_score,
            "crisis": crisis_flag,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        st.session_state.messages.append({
            "role": "assistant", "content": response,
            "crisis": crisis_flag, "risk": risk_score,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        st.session_state.flip = not st.session_state.flip
        st.rerun()


# ══════════════════════════════════════════════
# TAB 2 — RISK ASSESSMENT
# ══════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="risk-wrap">', unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer-box">
        ⚠️ <strong>Important:</strong> This is NOT a clinical diagnosis.
        Please consult a healthcare professional for proper assessment.
    </div>""", unsafe_allow_html=True)

    sadness  = st.slider("1. Sadness / Depression (past 2 weeks)", 0, 10, 0)
    anxiety  = st.slider("2. Anxiety / Worry",                     0, 10, 0)
    sleep    = st.slider("3. Sleep Quality (0=terrible, 10=great)", 0, 10, 5)
    energy   = st.slider("4. Energy Levels (0=exhausted, 10=high)", 0, 10, 5)
    selfharm = st.slider("5. Thoughts of self-harm (0=none, 10=constant)", 0, 10, 0)

    if st.button("🔍 Analyze My Risk", use_container_width=True):
        total = sadness + anxiety + (10 - sleep) + (10 - energy) + selfharm
        level = "high" if total > 25 else "medium" if total > 15 else "low"

        if level == "high":
            st.markdown(f"""<div class="risk-high">
                <h3>🚨 High Concern Level</h3>
                <p>Score: {total}/50 — Your responses suggest significant distress.</p>
                <p>Please speak with a mental health professional as soon as possible.</p>
                <p>🇮🇳 India: 9152987821 &nbsp;|&nbsp; 🇺🇸 USA: 988 &nbsp;|&nbsp; 🇬🇧 UK: 116 123</p>
            </div>""", unsafe_allow_html=True)
        elif level == "medium":
            st.markdown(f"""<div class="risk-medium">
                <h3>⚠️ Moderate Concern</h3>
                <p>Score: {total}/50 — Consider speaking to a professional soon.</p>
                <p>Try breathing exercises, journaling, or talking to someone you trust.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
                <h3>✅ Low Concern</h3>
                <p>Score: {total}/50 — You seem to be managing well.</p>
                <p>Keep up your self-care habits — small daily steps make a big difference.</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="mc-footer">
    ⚠️ MindCare is not a substitute for professional mental health care.<br>
    Made with ❤️ for mental wellness awareness
</div>
""", unsafe_allow_html=True)
