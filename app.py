import streamlit as st
import os
from datetime import datetime
from utils import (
    analyze_sentiment, detect_crisis_keywords,
    calculate_risk_score, stream_counseling_response
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

try:
    from streamlit_cookies_controller import CookieController
    COOKIES_AVAILABLE = True
except ImportError:
    COOKIES_AVAILABLE = False

try:
    from streamlit_oauth import OAuth2Component
    import httpx
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

REMEMBER_ME_COOKIE = "mindcare_token"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindCare – Mental Health Chatbot",
    page_icon="🧠",
    layout="wide"
)

db.init_db()
cookie_controller = CookieController() if COOKIES_AVAILABLE else None


def get_google_oauth():
    """Google OAuth is optional — only active if these secrets are configured."""
    if not OAUTH_AVAILABLE:
        return None
    try:
        client_id = st.secrets["GOOGLE_CLIENT_ID"]
        client_secret = st.secrets["GOOGLE_CLIENT_SECRET"]
        redirect_uri = st.secrets["GOOGLE_REDIRECT_URI"]
    except (KeyError, FileNotFoundError):
        return None
    oauth2 = OAuth2Component(
        client_id, client_secret,
        "https://accounts.google.com/o/oauth2/v2/auth",
        "https://oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com/revoke",
    )
    return oauth2, redirect_uri


# ─────────────────────────────────────────────
# CSS — ChatGPT-inspired dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Fraunces:wght@700;900&display=swap');
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; max-width: 900px !important; }
.stApp { background-color: #0d1117; font-family: 'DM Sans', sans-serif; }
section[data-testid="stSidebar"] { background-color: #10151c; border-right: 1px solid #2d3748; }

.auth-shell { display: flex; justify-content: center; padding-top: 5vh; }
.auth-card {
    background: #161b22; border: 1px solid #2d3748; border-radius: 16px;
    padding: 36px 32px; width: 100%; max-width: 400px;
}
.auth-logo { text-align: center; margin-bottom: 6px; font-size: 2rem; }
.auth-title {
    font-family: 'Fraunces', serif; font-weight: 900; font-size: 1.5rem;
    text-align: center; color: #e5e7eb; margin-bottom: 4px;
}
.auth-subtitle { text-align: center; color: #6b7280; font-size: 0.85rem; margin-bottom: 22px; }
.auth-divider { display: flex; align-items: center; color: #4b5563; font-size: 0.78rem; margin: 18px 0; }
.auth-divider::before, .auth-divider::after { content: ""; flex: 1; height: 1px; background: #2d3748; }
.auth-divider span { padding: 0 10px; }
.auth-switch { text-align: center; margin-top: 16px; color: #6b7280; font-size: 0.85rem; }

.sidebar-section-label {
    color: #6b7280; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; margin: 14px 0 4px 4px;
}
.mc-header { text-align: center; padding: 4px 0 4px; }
.mc-header h1 {
    font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 900;
    color: #6ee7b7; margin: 0; letter-spacing: -0.5px;
}
.mc-header p { color: #6b7280; font-size: 0.88rem; margin: 4px 0 0; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 1px solid #2d3748 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6b7280 !important;
    font-size: 0.9rem !important; font-weight: 500 !important; padding: 10px 22px !important;
}
.stTabs [aria-selected="true"] { color: #6ee7b7 !important; border-bottom: 2px solid #6ee7b7 !important; }

[data-testid="stChatMessage"] { padding: 4px 0; }
.msg-meta { font-size: 0.72rem; color: #6b7280; margin-top: 2px; }
.crisis-banner {
    font-size: 0.82rem; margin-bottom: 8px; padding: 8px 12px;
    background: #7f1d1d; border-radius: 8px; color: #fecaca;
}

.stButton > button {
    background: #6ee7b7 !important; color: #0d1117 !important; border: none !important;
    border-radius: 20px !important; font-weight: 700 !important;
    font-size: 0.85rem !important; font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.msg-action-btn button {
    background: transparent !important; color: #6b7280 !important; font-weight: 400 !important;
    height: 28px !important; padding: 0 8px !important; border: none !important;
}
.msg-action-btn button:hover { color: #6ee7b7 !important; }

.disclaimer-box {
    background: #1c1a0e; border: 1px solid #78350f; border-radius: 10px;
    padding: 12px 16px; color: #fbbf24; font-size: 0.85rem; margin-bottom: 20px;
}
.risk-low { background:#0f291a; border:1px solid #166534; color:#bbf7d0; border-radius:12px; padding:16px 20px; margin-top:12px; }
.risk-medium { background:#1c1a0e; border:1px solid #854d0e; color:#fde68a; border-radius:12px; padding:16px 20px; margin-top:12px; }
.risk-high { background:#450a0a; border:1px solid #f87171; color:#fecaca; border-radius:12px; padding:16px 20px; margin-top:12px; }
.risk-low h3, .risk-medium h3, .risk-high h3 { font-family: 'Fraunces', serif; font-size: 1.15rem; margin-bottom: 6px; }
.history-item {
    background: #1f2937; border-radius: 10px; padding: 10px 14px;
    margin-bottom: 8px; font-size: 0.82rem; color: #9ca3af;
}
.mc-footer { text-align: center; color: #374151; font-size: 0.8rem; padding: 20px 0 10px; }
label { color: #d1d5db !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIT LLM — short, common-sense replies
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
                        temperature=0.75, max_tokens=150)
        prompt = ChatPromptTemplate.from_template("""
You are MindCare AI, a grounded, emotionally intelligent counselor-friend. You listen carefully and respond to what THIS specific person just said — never a generic template.

HOW TO RESPOND (pick whichever fits this message best, don't do all of them every time):
1. Reflect — name the specific feeling or situation they described, in your own words, not theirs.
2. Ask — a short, genuine follow-up question that helps them go one layer deeper.
3. Reframe — gently offer a different, more workable way to look at the situation.
4. Suggest — ONE small, concrete, doable action (not a list, not generic "self-care" advice).
Vary which of these you lean on. Real counseling isn't the same shape every time — sometimes a person just needs to be heard with no advice at all; sometimes they need a nudge to act; sometimes they need a question, not an answer.

HARD RULES:
- LANGUAGE: Reply in the SAME language/style the user just wrote in. If they write in Hinglish (Hindi+English mixed, e.g. "yaar bahut stress ho raha hai"), reply in natural Hinglish too — not pure Hindi, not pure English. If they write in Hindi (Devanagari), reply in Hindi. If they write in English, reply in English. Match their register — casual stays casual.
- React to specific details/words from THEIR message. If you could paste your reply under a different message and it would still "work," rewrite it — that means it's too generic.
- Never open two replies in a row the same way. Avoid overusing "I hear you", "That sounds hard", "I understand" — vary your openings naturally, or skip an opener entirely and respond directly.
- Keep it SHORT: 2–4 sentences, under 60 words. No lectures, no bullet points, no headers, no therapy jargon.
- Never diagnose or prescribe medication.
- If self-harm, suicide, or crisis is mentioned: drop the brevity/style rules, respond with full seriousness and warmth, and clearly and immediately point them to a crisis helpline (mention India's helpline if the message is in Hindi/Hinglish, since that's the likely region).

EXAMPLES OF VARIED, SPECIFIC RESPONSES (for style only — don't reuse this content):
User: "yaar aaj bahut bura din tha, boss ne sabke saamne daata"
Reply: Sabke saamne daant padna sach me bura lagta hai, ego pe seedha lagta hai. Kal calm ho ke unse ek-on-one baat karega?

User: "I bombed my exam and I feel like such a failure"
Reply: One exam doesn't define you, but I get why it stings right now. What's the actual damage — is this recoverable, or does it change something bigger?

User: "I haven't left my room in three days"
Reply: Three days is a long stretch to be that isolated. Even just opening a window or stepping outside for two minutes today could break the pattern a little — what's stopping you right now?

User: "work has just been so much lately, my manager keeps piling stuff on"
Reply: That's a lot to carry without it being acknowledged. Have you actually told your manager where your limit is, or has it been more silent overload?

User message: {input}

Your reply (short, specific to what they said, natural variation in style):
""")
        return prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
        return None


def load_conversation_into_session(user_id: int):
    convos = db.get_conversations(user_id)
    if not convos:
        db.create_conversation(user_id, "New Chat")
        convos = db.get_conversations(user_id)
    st.session_state.current_conv_id = convos[0]["id"]
    st.session_state.messages = db.get_messages(convos[0]["id"])


def group_conversations_by_date(convos):
    """Bucket conversations into ChatGPT-style Today / Yesterday / Previous 7 Days / Older."""
    now = datetime.now()
    buckets = {"Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": []}
    for c in convos:
        updated = datetime.fromisoformat(c["updated_at"])
        delta_days = (now.date() - updated.date()).days
        if delta_days == 0:
            buckets["Today"].append(c)
        elif delta_days == 1:
            buckets["Yesterday"].append(c)
        elif delta_days <= 7:
            buckets["Previous 7 Days"].append(c)
        else:
            buckets["Older"].append(c)
    return buckets


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "user": None, "messages": [], "current_conv_id": None,
    "search_query": "", "renaming_id": None, "editing_msg_id": None,
    "auth_mode": "login",
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

# ══════════════════════════════════════════════
# AUTO-LOGIN from "remember me" cookie
# ══════════════════════════════════════════════
if st.session_state.user is None and cookie_controller is not None:
    saved_token = cookie_controller.get(REMEMBER_ME_COOKIE)
    if saved_token:
        remembered_user = auth.resolve_session_token(saved_token)
        if remembered_user:
            st.session_state.user = remembered_user
            load_conversation_into_session(remembered_user["id"])

# ══════════════════════════════════════════════
# LOGIN / SIGNUP — ChatGPT-style centered card
# ══════════════════════════════════════════════
if st.session_state.user is None:
    st.markdown('<div class="auth-shell"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-logo">🧠</div>', unsafe_allow_html=True)

    is_login = st.session_state.auth_mode == "login"
    st.markdown(f'<div class="auth-title">{"Welcome back" if is_login else "Create your account"}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="auth-subtitle">{"Log in to continue to MindCare" if is_login else "Get started with MindCare"}</div>',
                unsafe_allow_html=True)

    google = get_google_oauth()
    if google:
        oauth2, redirect_uri = google
        result = oauth2.authorize_button(
            "Continue with Google", redirect_uri, "openid email profile",
            icon="https://www.google.com/favicon.ico", use_container_width=True, pkce="S256",
        )
        if result and "token" in result:
            access_token = result["token"]["access_token"]
            resp = httpx.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            info = resp.json()
            user = auth.login_with_google(info.get("email"), info.get("name"))
            st.session_state.user = user
            load_conversation_into_session(user["id"])
            if cookie_controller is not None:
                token = auth.create_remember_me_token(user["id"])
                cookie_controller.set(REMEMBER_ME_COOKIE, token,
                                       max_age=auth.SESSION_LIFETIME_DAYS * 24 * 60 * 60)
            st.rerun()
        st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)
    else:
        st.button("🔒 Continue with Google (needs setup)", use_container_width=True, disabled=True)
        st.caption("Add GOOGLE_CLIENT_ID / SECRET / REDIRECT_URI to secrets to enable this.")
        st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)

    if is_login:
        with st.form("login_form"):
            identifier = st.text_input("Username, Email, or Mobile Number")
            p = st.text_input("Password", type="password")
            remember_me = st.checkbox("Keep me logged in", value=True)
            submitted = st.form_submit_button("Log In", use_container_width=True)
        if submitted:
            ok, user, msg = auth.login(identifier, p)
            if ok:
                st.session_state.user = user
                load_conversation_into_session(user["id"])
                if remember_me and cookie_controller is not None:
                    token = auth.create_remember_me_token(user["id"])
                    cookie_controller.set(REMEMBER_ME_COOKIE, token,
                                           max_age=auth.SESSION_LIFETIME_DAYS * 24 * 60 * 60)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        st.markdown('<div class="auth-switch">New here?</div>', unsafe_allow_html=True)
        if st.button("Create an account", use_container_width=True):
            st.session_state.auth_mode = "signup"
            st.rerun()
    else:
        with st.form("signup_form"):
            new_u = st.text_input("Choose a username")
            new_email = st.text_input("Email (optional if mobile is given)")
            new_mobile = st.text_input("Mobile number (optional if email is given)")
            new_p = st.text_input("Choose a password", type="password")
            new_p2 = st.text_input("Confirm password", type="password")
            signed_up = st.form_submit_button("Create Account", use_container_width=True)
        if signed_up:
            if new_p != new_p2:
                st.error("Passwords don't match.")
            else:
                ok, msg = auth.signup(new_u, new_p, email=new_email, mobile=new_mobile)
                if ok:
                    st.success(msg)
                    st.session_state.auth_mode = "login"
                else:
                    st.error(msg)
        st.markdown('<div class="auth-switch">Already have an account?</div>', unsafe_allow_html=True)
        if st.button("Log in instead", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

user_id = st.session_state.user["id"]

# ══════════════════════════════════════════════
# SIDEBAR — ChatGPT-style grouped history
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"**👤 {st.session_state.user['username']}**")
    if st.button("Log Out", use_container_width=True):
        if cookie_controller is not None:
            saved_token = cookie_controller.get(REMEMBER_ME_COOKIE)
            if saved_token:
                auth.revoke_session_token(saved_token)
                cookie_controller.remove(REMEMBER_ME_COOKIE)
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
                                  placeholder="Search by title or content...",
                                  label_visibility="collapsed")
    st.session_state.search_query = search_query

    convos = db.search_conversations(user_id, search_query) if search_query else db.get_conversations(user_id)

    if not convos:
        st.caption("No conversations yet.")

    grouped = group_conversations_by_date(convos) if not search_query else {"Search results": convos}

    for label, group in grouped.items():
        if not group:
            continue
        st.markdown(f'<div class="sidebar-section-label">{label}</div>', unsafe_allow_html=True)
        for c in group:
            is_active = c["id"] == st.session_state.current_conv_id
            col_a, col_b, col_c = st.columns([5, 1, 1])
            with col_a:
                title = ("🟢 " if is_active else "") + c["title"]
                if st.button(title, key=f"open_{c['id']}", use_container_width=True):
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

    if st.session_state.current_conv_id and st.session_state.messages:
        transcript = "\n\n".join(
            f"[{m['timestamp']}] {'You' if m['role']=='user' else 'MindCare'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.markdown("---")
        st.download_button("⬇️ Export this chat", transcript,
                            file_name=f"mindcare_chat_{st.session_state.current_conv_id}.txt",
                            use_container_width=True)

if st.session_state.current_conv_id is None:
    load_conversation_into_session(user_id)

conv_id = st.session_state.current_conv_id

st.markdown("""
<div class="mc-header">
    <h1>🧠 MindCare</h1>
    <p>Mental Health Chatbot &amp; Risk Predictor</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_risk = st.tabs(["💬 Chat", "⚠️ Risk Assessment"])

# ══════════════════════════════════════════════
# TAB 1 — CHAT (native st.chat_message, streaming, copy/edit/regenerate)
# ══════════════════════════════════════════════
with tab_chat:

    def run_and_store_reply(conv_id, user_text, sentiment, risk_score, crisis_flag):
        """Stream a reply into a live chat bubble, then persist it."""
        prior_bot_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
        last_bot_reply = prior_bot_msgs[-1]["content"] if prior_bot_msgs else None

        with st.chat_message("assistant", avatar="🧠"):
            if crisis_flag:
                st.markdown(
                    '<div class="crisis-banner">🚨 Please reach out to a crisis helpline — you are not alone.</div>',
                    unsafe_allow_html=True,
                )
            full_response = st.write_stream(
                stream_counseling_response(st.session_state.llm, user_text, sentiment, risk_score, last_bot_reply)
            )
        now = datetime.now().strftime("%I:%M %p")
        bot_msg_id = db.save_message(conv_id, "assistant", full_response, None, risk_score, crisis_flag, now)
        st.session_state.messages.append({
            "id": bot_msg_id, "role": "assistant", "content": full_response,
            "crisis": crisis_flag, "risk": risk_score, "timestamp": now,
        })

    def regenerate_last_response():
        last_user_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if not last_user_msg:
            return
        last_bot_msg = st.session_state.messages[-1]
        db.truncate_messages_from(conv_id, last_bot_msg["id"])
        st.session_state.messages.pop()
        run_and_store_reply(conv_id, last_user_msg["content"], last_user_msg.get("sentiment"),
                             last_user_msg.get("risk") or 0, last_user_msg.get("crisis", False))

    def submit_edited_message(msg, new_text):
        """Truncate everything from this message onward, save the edit, and regenerate."""
        db.truncate_messages_from(conv_id, msg["id"])
        idx = next(i for i, m in enumerate(st.session_state.messages) if m["id"] == msg["id"])
        st.session_state.messages = st.session_state.messages[:idx]

        sentiment = analyze_sentiment(new_text)
        crisis_flag = detect_crisis_keywords(new_text)
        risk_score = calculate_risk_score(new_text)
        now = datetime.now().strftime("%I:%M %p")
        new_msg_id = db.save_message(conv_id, "user", new_text, sentiment, risk_score, crisis_flag, now)
        st.session_state.messages.append({
            "id": new_msg_id, "role": "user", "content": new_text,
            "sentiment": sentiment, "risk": risk_score, "crisis": crisis_flag, "timestamp": now,
        })
        st.session_state.editing_msg_id = None
        run_and_store_reply(conv_id, new_text, sentiment, risk_score, crisis_flag)

    def render_message_actions(msg, index):
        """Copy button for every message; Edit for user messages; Regenerate for the last assistant message."""
        is_last = index == len(st.session_state.messages) - 1
        cols = st.columns([1, 1, 1, 8])

        with cols[0]:
            st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
            if st.button("📋", key=f"copy_{msg['id']}", help="Show text to copy"):
                st.session_state[f"show_copy_{msg['id']}"] = True
            st.markdown('</div>', unsafe_allow_html=True)
        if st.session_state.get(f"show_copy_{msg['id']}"):
            st.code(msg["content"], language=None)
            st.session_state[f"show_copy_{msg['id']}"] = False

        if msg["role"] == "user":
            with cols[1]:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if st.button("✏️", key=f"edit_{msg['id']}", help="Edit"):
                    st.session_state.editing_msg_id = msg["id"]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        if msg["role"] == "assistant" and is_last:
            with cols[2]:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if st.button("🔁", key=f"regen_{msg['id']}", help="Regenerate"):
                    regenerate_last_response()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Render existing messages ──
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🧠"):
            st.write("Hello! I'm here to listen and support you. How are you feeling today? 💙")
    else:
        for i, msg in enumerate(st.session_state.messages):
            avatar = "🧑" if msg["role"] == "user" else "🧠"
            with st.chat_message(msg["role"], avatar=avatar):
                if st.session_state.editing_msg_id == msg["id"]:
                    edited_text = st.text_area("Edit message", value=msg["content"],
                                                key=f"editbox_{msg['id']}", label_visibility="collapsed")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        if st.button("Save & Regenerate", key=f"save_{msg['id']}", use_container_width=True):
                            submit_edited_message(msg, edited_text.strip())
                            st.rerun()
                    with ec2:
                        if st.button("Cancel", key=f"cancel_{msg['id']}", use_container_width=True):
                            st.session_state.editing_msg_id = None
                            st.rerun()
                else:
                    if msg["role"] == "assistant" and msg.get("crisis"):
                        st.markdown(
                            '<div class="crisis-banner">🚨 Please reach out to a crisis helpline — you are not alone.</div>',
                            unsafe_allow_html=True,
                        )
                    st.write(msg["content"])
                    if msg["role"] == "user":
                        meta_bits = [msg.get("sentiment") or "—",
                                     f"{(msg.get('risk') or 0):.0%} risk", msg["timestamp"]]
                    else:
                        meta_bits = [msg["timestamp"]]
                    st.markdown(f'<div class="msg-meta">{" · ".join(meta_bits)}</div>', unsafe_allow_html=True)
                    render_message_actions(msg, i)

    # ── Chat input (native Enter-to-send) ──
    user_input = st.chat_input("Share how you're feeling...")

    if user_input and user_input.strip():
        text = user_input.strip()
        sentiment = analyze_sentiment(text)
        crisis_flag = detect_crisis_keywords(text)
        risk_score = calculate_risk_score(text)
        now = datetime.now().strftime("%I:%M %p")

        with st.chat_message("user", avatar="🧑"):
            st.write(text)

        user_msg_id = db.save_message(conv_id, "user", text, sentiment, risk_score, crisis_flag, now)
        st.session_state.messages.append({
            "id": user_msg_id, "role": "user", "content": text,
            "sentiment": sentiment, "risk": risk_score, "crisis": crisis_flag, "timestamp": now,
        })

        conv = db.get_conversation(conv_id)
        if conv and conv["title"] == "New Chat":
            title = (text[:40] + "…") if len(text) > 40 else text
            db.rename_conversation(conv_id, title)

        run_and_store_reply(conv_id, text, sentiment, risk_score, crisis_flag)
        st.rerun()

    if st.button("🗑️ Clear This Chat"):
        st.session_state.messages = []
        db.clear_messages(conv_id)
        st.rerun()

# ══════════════════════════════════════════════
# TAB 2 — RISK ASSESSMENT
# ══════════════════════════════════════════════
with tab_risk:
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

st.markdown("""
<div class="mc-footer">
    ⚠️ MindCare is not a substitute for professional mental health care.<br>
    Made with ❤️ for mental wellness awareness
</div>
""", unsafe_allow_html=True)
