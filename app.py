import streamlit as st
import os
import traceback
from datetime import datetime
import pandas as pd

# Local imports (ensure these files exist)
try:
    from utils import analyze_text, predict_from_form
except ImportError:
    st.error("❌ utils.py not found! Please create it.")
    st.stop()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= CONFIG =================
st.set_page_config(
    page_title="🧠 MindCare AI - Mental Health Assistant", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #4facfe; }
    .risk-high { background: linear-gradient(135deg, #ff6b6b, #ee5a24); }
    .risk-medium { background: linear-gradient(135deg, #ffd93d, #ffcc5c); }
    .risk-low { background: linear-gradient(135deg, #a8e6cf, #88d8a3); }
    .crisis-alert { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

# ================= API KEY =================
@st.cache_resource
def init_llm():
    """Initialize LLM with proper error handling"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("❌ **GROQ_API_KEY not found in environment variables!**")
        st.info("Set it with: `streamlit run app.py --server.headless=true GROQ_API_KEY=your_key`")
        return None
    
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama3-8b-8192",
            temperature=0.1,  # Lower for consistency
            max_tokens=500
        )
        
        # Safety-focused prompt
        prompt = ChatPromptTemplate.from_template("""
        You are MindCare AI, a compassionate mental health assistant. 
        
        IMPORTANT SAFETY RULES:
        1. Always be empathetic and supportive
        2. NEVER give medical diagnoses or prescriptions
        3. If user mentions self-harm/suicide, IMMEDIATELY provide crisis resources
        4. Always suggest professional help when appropriate
        5. End with resources if risk seems high
        
        CRISIS KEYWORDS: suicide, self-harm, kill myself, hurt myself, end it all
        
        User: {input}
        
        Respond supportively:
        """)
        
        chain = prompt | llm | StrOutputParser()
        return chain
        
    except Exception as e:
        st.error(f"❌ LLM initialization failed: {str(e)}")
        return None

# ================= SIDEBAR RESOURCES =================
with st.sidebar:
    st.markdown("## 📞 **CRISIS RESOURCES**")
    st.markdown("""
    **Immediate Help:**
    - **US**: 988 Suicide & Crisis Lifeline
    - **UK**: 116 123 Samaritans  
    - **Australia**: 13 11 14 Lifeline
    - **Canada**: 1-833-456-4566
    
    **This is NOT a substitute for professional care**
    """)
    
    st.markdown("---")
    st.markdown("**🧠 Disclaimer**: AI analysis ≠ medical diagnosis")

# ================= MAIN UI =================
st.markdown('<h1 class="main-header">🧠 MindCare AI</h1>', unsafe_allow_html=True)
st.markdown("*Your compassionate mental health companion*")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = init_llm()

# ================= CHAT SECTION =================
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 **Talk to me**")
    user_input = st.text_input(
        "How are you feeling today?",
        placeholder="I'm feeling overwhelmed at work...",
        key="chat_input"
    )

with col2:
    if st.button("✨ Analyze", type="primary"):
        st.session_state.analyze_clicked = True

# Process chat
if user_input and st.session_state.llm_chain:
    with st.spinner("MindCare is listening..."):
        try:
            # Get AI response
            response = st.session_state.llm_chain.invoke({"input": user_input})
            
            # Risk analysis
            risk_score = analyze_text(user_input)
            risk_label = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
            
            # Store in history
            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%H:%M"),
                "user": user_input,
                "ai": response,
                "risk": risk_label,
                "risk_score": risk_score
            })
            
            # Display response
            st.markdown("### 🤖 **MindCare's Response**")
            st.write(response)
            
            # Risk display
            st.markdown("### 📊 **Risk Assessment**")
            risk_emoji = "🔴" if risk_label == "HIGH" else "🟡" if risk_label == "MEDIUM" else "🟢"
            
            if risk_label == "HIGH":
                st.markdown(f"""
                <div class="risk-high crisis-alert">
                    <h3>{risk_emoji} **HIGH RISK DETECTED**</h3>
                    <p><strong>Please contact emergency services immediately!</strong></p>
                </div>
                """, unsafe_allow_html=True)
            elif risk_label == "MEDIUM":
                st.warning(f"{risk_emoji} **Moderate Risk** - Consider professional support")
            else:
                st.success(f"{risk_emoji} **Low Risk** - Keep up self-care!")
                
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")
            st.error("Please try again or contact support")

# Chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 **Conversation History**")
    
    for chat in st.session_state.chat_history[-5:]:  # Last 5
        with st.expander(f"**{chat['timestamp']}** - Risk: {chat['risk']}"):
            st.write(f"**You:** {chat['user']}")
            st.write(f"**MindCare:** {chat['ai']}")

# ================= FORM SECTION =================
st.markdown("---")
st.subheader("📋 **Quick Risk Assessment**")

form_col1, form_col2 = st.columns(2)

with form_col1:
    age = st.slider("👤 Age", 13, 80, 25)
    gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
    family_history = st.selectbox("👨‍👩‍👧 Family history of mental health issues?", ["No", "Yes"])

with form_col2:
    work_interfere = st.selectbox(
        "💼 Does mental health interfere with work/studies?",
        ["Never", "Rarely", "Sometimes", "Often", "Always"]
    )
    sleep_quality = st.slider("😴 Sleep quality (1-10)", 1, 10, 5)
    mood_score = st.slider("😊 Current mood (1-10)", 1, 10, 5)

if st.button("🔍 **Predict Risk Level**", type="secondary", use_container_width=True):
    try:
        with st.spinner("Analyzing..."):
            # Enhanced prediction with more features
            result = predict_from_form(
                age, gender, family_history, work_interfere,
                sleep_quality=sleep_quality, mood_score=mood_score
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if "HIGH" in result or "Treatment" in result:
                    st.error("🚨 **HIGH RISK** - Please seek professional help immediately")
                    st.info("**Hotlines:** 988 (US) | 116 123 (UK) | 13 11 14 (AU)")
                elif "MEDIUM" in result:
                    st.warning("⚠️ **Moderate Risk** - Consider talking to a professional")
                else:
                    st.success("✅ **Low Risk** - Continue self-care practices")
                    
                st.write(f"**Prediction:** {result}")
                
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 20px;'>
    <p><strong>⚠️ Important:</strong> This AI is <em>NOT</em> a substitute for professional medical advice.</p>
    <p>Always consult qualified healthcare professionals for diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
