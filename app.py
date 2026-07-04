import streamlit as st
import os
from datetime import datetime
from utils import (
    analyze_sentiment, detect_crisis_keywords,
    calculate_risk_score, generate_counseling_response
)
import database as db
import auth

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
    layout="wide"
)

db.init_db()

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Fraunces:wght@700;900&display=swap');
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }
.stApp { background-color: #0d1117; font-family: 'DM Sans', sans-serif; }
section[data-testid="stSidebar"] { background-color: #10151c; border-right: 1px solid #2d3748; }
.mc-header { text-align: center; padding: 6px 0 6px; }
.mc-header h1 {
    font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 900;
    color: #6ee7b7; margin: 0; letter-spacing: -0.5px;
}
.mc-header p { color: #6b7280; font-size: 0.9rem; margin: 4px 0 0; }
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important; border-radius: 14px 14px 0 0 !important;
    border-bottom: 1px solid #2d3748 !important; gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6b7280 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.92rem !important;
    font-weight: 500 !important; padding: 14px 32px !important; border-radius: 0 !important;
    border: none !important; flex: 1; text-align: center;
}
.stTabs [aria-selected="true"] {
    color: #6ee7b7 !important; border-bottom: 2px solid #6ee7b7 !important; background: #1f2937 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #161b22 !important; border-radius: 0 0 14px 14px !important;
    border: 1px solid #2d3748 !important; border-top: none !important; padding: 0 !important;
}
.chat-scroll {
    min-height: 420px; max-height: 480px; overflow-y: auto; padding: 20px 20px 8px;
    display: flex; flex-direction: column; gap: 12px;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }
.bubble {
    max-width: 78%; padding: 12px 16px; border-radius: 14px;
    font-size: 0.92rem; line-height: 1.6; word-wrap: break-word;
}
.bubble .sender {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 5px; opacity: 0.65;
}
.bubble.bot { background: #1a2e1f; color: #bbf7d0; border-bottom-left-radius: 4px; align-self: flex-start; }
.bubble.user {
    background: #1e3a5f; color: #bfdbfe; border-bottom-right-radius: 4px;
    align-self: flex-end; margin-left: auto;
}
.bubble.crisis { background: #450a0a; border: 1px solid #f87171; color: #fecaca; }
.risk-meta { font-size: 0.72rem; color: #6b7280; margin-top: 5px; }
.stTextInput > div > div > input {
    background: #1f2937 !important; border: 1px solid #374151 !important; border-radius: 25px !important;
    color: #e2e8f0 !important; padding: 11px 18px !important; font-size: 0.93rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus { border-color: #6ee7b7 !important; box-shadow: none !important; }
.stTextInput > div > div > input::placeholder { color: #4b5563 !important; }
.stButton > button {
    background: #6ee7b7 !important; color: #0d1117 !important; border: none !important;
    border-radius: 25px !important; height: 48px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; font-family: 'DM Sans', sans-serif !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }
.risk-wrap { padding: 24px 20px; }
.disclaimer-box {
    background: #1c1a0e; border: 1px solid #78350f; border-radius: 10px;
    padding: 12px 16px; color: #fbbf24; font-size: 0.85rem; margin-bottom: 20px;
}
label { color: #d1d5db !important; font-size: 0.9rem !important; }
.risk-low { background:#0f291a; border:1px solid #166534; color:#bbf7d0; border-radius:12px; padding:16px 20px; margin-top:12px; }
.risk-medium { background:#1c1a0e; border:1px solid #854d0e; color:#fde68a; border-radius:12px; padding:16px 20px; margin-top:12px; }
.risk-high { background:#450a0a; border:1px solid #f87171; color:#fecaca; border-radius:12px; padding:16px 20px; margin-top:12px; }
.risk-low h3, .risk-medium h3, .risk-high h3 { font-family: 'Fraunces', serif; font-size: 1.15rem; margin-bottom: 6px; }
hr { border-color: #2d3748 !important; margin: 8px 0 !important; }
.mc-footer { text-align: center; color: #374151; font-size: 0.8rem; padding: 20px 0 10px; }
.auth-box {
    background: #161b22; border: 1px solid #2d3748; border-radius: 14px;
    padding: 28px 24px; max-width: 420px; margin: 40px auto 0;
}
.history-item {
    background: #1f2937; border-radius: 10px; padding: 10px 14px;
    margin-bottom: 8px; font-size: 0.82rem; color: #9ca3af;
}
.convo-item {
    padding: 9px 10px; border-radius: 8px; margin-bottom: 4px;
    font-size: 0.85rem; color: #d1d5db; cursor: pointer;
}
.convo-item.active { background: #1f2937; color: #6ee7b7; }
.convo-title { font-weight: 500; }
.convo-date { font-size: 0.7rem; color: #6b7280; }
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

Conversation so far:
{history}

User message: {input}

Respond warmly (150–250 words):
""")
        return prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
        return None


def build_history_text(messages, exclude_last=0):
    """Turn stored messages into a short transcript string for LLM context."""
    subset = messages[:-exclude_last] if exclude_last else messages
    lines = []
    for m in subset[-10:]:  # last 10 turns of context is plenty
        speaker = "User" if m["role"] == "user" else "MindCare"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines) if lines else "(no prior messages)"


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "user": None, "messages": [], "current_conv_id": None,
    "flip": False, "search_query": "", "renaming_id": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "llm" not in st.session_state:
    st.session_state.llm = init_llm()

if not LLM_AVAILABLE:
    st.error("❌ Run: pip install -r requirements.txt")
    st.stop()
if not st.session_state.llm:
    st.error("❌ GROQ_API_KEY missing — add it to Streamlit secrets.")
    st.stop()

st.markdown("""
<div class="mc-header">
    <h1>🧠 MindCare</h1>
    <p>Mental Health Chatbot &amp; Risk Predictor</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# LOGIN / SIGNUP GATE
# ══════════════════════════════════════════════
if st.session_state.user is None:
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    tab_login, tab_signup = st.tabs(["🔑 Log In", "🆕 Sign Up"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)
        if submitted:
            ok, user, msg = auth.login(u, p)
            if ok:
                st.session_state.user = user
                convos = db.get_conversations(user["id"])
                if not convos:
                    new_id = db.create_conversation(user["id"], "New Chat")
                    convos = db.get_conversations(user["id"])
                st.session_state.current_conv_id = convos[0]["id"]
                st.session_state.messages = db.get_messages(convos[0]["id"])
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        with st.form("signup_form"):
            new_u = st.text_input("Choose a username")
            new_p = st.text_input("Choose a password", type="password")
            new_p2 = st.text_input("Confirm password", type="password")
            signed_up = st.form_submit_button("Create Account", use_container_width=True)
        if signed_up:
            if new_p != new_p2:
                st.error("Passwords don't match.")
            else:
                ok, msg = auth.signup(new_u, new_p)
                st.success(msg) if ok else st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

user_id = st.session_state.user["id"]

# ══════════════════════════════════════════════
# SIDEBAR — chat history (ChatGPT-style)
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"**👤 {st.session_state.user['username']}**")
    if st.button("Log Out", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.current_conv_id = None
        st.rerun()

    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = db.create_conversation(user_id, "New Chat")
        st.session_state.current_conv_id = new_id
        st.session_state.messages = []
        st.rerun()

    search_query = st.text_input("🔍 Search chats", value=st.session_state.search_query,
                                  placeholder="Search by title or content...")
    st.session_state.search_query = search_query

    st.markdown("**History**")
    convos = db.search_conversations(user_id, search_query) if search_query else db.get_conversations(user_id)

    if not convos:
        st.caption("No conversations found.")

    for c in convos:
        is_active = c["id"] == st.session_state.current_conv_id
        col_a, col_b, col_c = st.columns([5, 1, 1])
        with col_a:
            label = ("🟢 " if is_active else "") + c["title"]
            if st.button(label, key=f"open_{c['id']}", use_container_width=True):
                st.session_state.current_conv_id = c["id"]
                st.session_state.messages = db.get_messages(c["id"])
                st.session_state.renaming_id = None
                st.rerun()
        with col_b:
            if st.button("✏️", key=f"ren_{c['id']}"):
                st.session_state.renaming_id = c["id"]
                st.rerun()
        with col_c:
            if st.button("🗑️", key=f"del_{c['id']}"):
                db.delete_conversation(c["id"])
                if st.session_state.current_conv_id == c["id"]:
                    remaining = db.get_conversations(user_id)
                    if remaining:
                        st.session_state.current_conv_id = remaining[0]["id"]
                        st.session_state.messages = db.get_messages(remaining[0]["id"])
                    else:
                        new_id = db.create_conversation(user_id, "New Chat")
                        st.session_state.current_conv_id = new_id
                        st.session_state.messages = []
                st.rerun()

        if st.session_state.renaming_id == c["id"]:
            with st.form(f"rename_form_{c['id']}", clear_on_submit=True):
                new_title = st.text_input("New title", value=c["title"], key=f"title_input_{c['id']}")
                if st.form_submit_button("Save"):
                    db.rename_conversation(c["id"], new_title.strip() or "Untitled")
                    st.session_state.renaming_id = None
                    st.rerun()

        st.caption(datetime.fromisoformat(c["updated_at"]).strftime("%b %d, %I:%M %p"))

    # Export current conversation
    if st.session_state.current_conv_id and st.session_state.messages:
        transcript = "\n\n".join(
            f"[{m['timestamp']}] {'You' if m['role']=='user' else 'MindCare'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button("⬇️ Export this chat", transcript,
                            file_name=f"mindcare_chat_{st.session_state.current_conv_id}.txt",
                            use_container_width=True)

if st.session_state.current_conv_id is None:
    new_id = db.create_conversation(user_id, "New Chat")
    st.session_state.current_conv_id = new_id
    st.session_state.messages = []

conv_id = st.session_state.current_conv_id

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_chat, tab_risk = st.tabs(["💬 Chat", "⚠️ Risk Assessment"])

# ══════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════
with tab_chat:
    bubbles_html = ""
    if not st.session_state.messages:
        bubbles_html = """
        <div class="bubble bot">
            <div class="sender">MindCare</div>
            Hello! I'm here to listen and support you. How are you feeling today? 💙
        </div>"""
    else:
        for msg in st.session_state.messages[-40:]:
            if msg["role"] == "user":
                bubbles_html += f"""
                <div class="bubble user">
                    <div class="sender">You</div>
                    {msg["content"]}
                    <div class="risk-meta">
                        {msg.get("sentiment") or "—"} &nbsp;·&nbsp;
                        {"🔴" if (msg.get("risk") or 0)>0.7 else "🟡" if (msg.get("risk") or 0)>0.4 else "🟢"}
                        {(msg.get("risk") or 0):.0%} &nbsp;·&nbsp; {msg["timestamp"]}
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

    st.markdown(f'<div class="chat-scroll" id="chat-end">{bubbles_html}</div>', unsafe_allow_html=True)
    st.markdown("""<script>
        const el = document.getElementById('chat-end');
        if(el) el.scrollTop = el.scrollHeight;
        </script>""", unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        key = f"ci_{st.session_state.flip}"
        user_input = st.text_input("msg", key=key,
                                    placeholder="Share how you're feeling...",
                                    label_visibility="collapsed")
    with col2:
        send = st.button("Send", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔁 Regenerate last response", use_container_width=True,
                      disabled=not (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant")):
            last_user_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
            if last_user_msg:
                st.session_state.messages.pop()  # drop old bot reply from view
                with st.spinner("MindCare is rethinking..."):
                    history_text = build_history_text(st.session_state.messages, exclude_last=1)
                    new_response = generate_counseling_response(
                        st.session_state.llm, last_user_msg["content"],
                        last_user_msg.get("sentiment"), last_user_msg.get("risk") or 0)
                now = datetime.now().strftime("%I:%M %p")
                bot_msg = {"role": "assistant", "content": new_response,
                           "crisis": last_user_msg.get("crisis"), "risk": last_user_msg.get("risk"),
                           "timestamp": now}
                st.session_state.messages.append(bot_msg)
                db.save_message(conv_id, "assistant", new_response, None,
                                 last_user_msg.get("risk"), last_user_msg.get("crisis"), now)
                st.rerun()
    with col4:
        if st.button("🗑️ Clear This Chat", use_container_width=True):
            st.session_state.messages = []
            db.clear_messages(conv_id)
            st.rerun()

    if send and user_input.strip():
        text = user_input.strip()
        with st.spinner("MindCare is listening..."):
            sentiment = analyze_sentiment(text)
            crisis_flag = detect_crisis_keywords(text)
            risk_score = calculate_risk_score(text)
            response = generate_counseling_response(
                st.session_state.llm, text, sentiment, risk_score)
            now = datetime.now().strftime("%I:%M %p")

            user_msg = {"role": "user", "content": text, "sentiment": sentiment,
                        "risk": risk_score, "crisis": crisis_flag, "timestamp": now}
            bot_msg = {"role": "assistant", "content": response,
                       "crisis": crisis_flag, "risk": risk_score, "timestamp": now}

            st.session_state.messages.append(user_msg)
            st.session_state.messages.append(bot_msg)

            db.save_message(conv_id, "user", text, sentiment, risk_score, crisis_flag, now)
            db.save_message(conv_id, "assistant", response, None, risk_score, crisis_flag, now)

            # Auto-title the conversation from the first user message
            conv = db.get_conversation(conv_id)
            if conv and conv["title"] == "New Chat":
                title = (text[:40] + "…") if len(text) > 40 else text
                db.rename_conversation(conv_id, title)

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

    sadness = st.slider("1. Sadness / Depression (past 2 weeks)", 0, 10, 0)
    anxiety = st.slider("2. Anxiety / Worry", 0, 10, 0)
    sleep = st.slider("3. Sleep Quality (0=terrible, 10=great)", 0, 10, 5)
    energy = st.slider("4. Energy Levels (0=exhausted, 10=high)", 0, 10, 5)
    selfharm = st.slider("5. Thoughts of self-harm (0=none, 10=constant)", 0, 10, 0)

    if st.button("🔍 Analyze My Risk", use_container_width=True):
        total = sadness + anxiety + (10 - sleep) + (10 - energy) + selfharm
        level = "high" if total > 25 else "medium" if total > 15 else "low"
        db.save_risk_assessment(user_id, sadness, anxiety, sleep, energy, selfharm, total, level)

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

    history = db.get_risk_history(user_id)
    if history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='color:#9ca3af;font-size:0.85rem;font-weight:600;'>Your past assessments</p>", unsafe_allow_html=True)
        for h in history:
            emoji = "🔴" if h["level"] == "high" else "🟡" if h["level"] == "medium" else "🟢"
            st.markdown(
                f"""<div class="history-item">{emoji} {h['level'].capitalize()} —
                Score {h['total_score']}/50 &nbsp;·&nbsp; {h['timestamp']}</div>""",
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="mc-footer">
    ⚠️ MindCare is not a substitute for professional mental health care.<br>
    Made with ❤️ for mental wellness awareness
</div>
""", unsafe_allow_html=True)
