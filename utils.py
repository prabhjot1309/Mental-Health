import re
from typing import Optional

# ─────────────────────────────────────────────
# KEYWORD LISTS
# ─────────────────────────────────────────────
SENTIMENT_KEYWORDS = {
    "positive": [
        "happy", "good", "great", "excited", "grateful", "blessed",
        "hopeful", "joyful", "content", "relieved", "motivated", "better",
        "calm", "peaceful", "proud", "loved", "supported",
        # Hindi/Hinglish
        "khush", "accha", "acha", "badhiya", "mast", "shukar"
    ],
    "negative": [
        "sad", "depressed", "anxious", "stressed", "overwhelmed", "hopeless",
        "lonely", "empty", "worthless", "trapped", "tired", "exhausted",
        "scared", "angry", "numb", "broken", "lost", "afraid", "hurt",
        # Hindi/Hinglish
        "udaas", "pareshan", "dukhi", "akela", "akeli", "thak gaya", "thak gayi",
        "thak chuki", "thak chuka", "bekaar", "tanhai", "ghabrahat", "dar lag raha"
    ],
    "neutral": [
        "okay", "fine", "normal", "alright", "average", "so-so",
        "theek", "thik hu", "thik hoon"
    ]
}

# Tiered crisis keywords — high weight vs moderate weight.
# IMPORTANT: covers English AND Hindi/Hinglish (transliterated + Devanagari),
# since this app gets real Hindi/Hinglish messages and missing a crisis
# disclosure because it wasn't in English is a genuine safety failure.
CRISIS_KEYWORDS_HIGH = [
    # English
    "suicide", "kill myself", "end my life", "take my life",
    "better off dead", "want to die", "going to die", "plan to die",
    "overdose", "slit my wrists", "hang myself", "jump off",
    # Hinglish (transliterated Hindi)
    "mujhe marna hai", "marna chahta hoon", "marna chahti hoon",
    "mujhe mar jana hai", "mar jaana chahta hoon", "mar jaana chahti hoon",
    "khud ko khatam", "zindagi khatam karna", "jeena nahi chahta",
    "jeena nahi chahti", "jeene ka man nahi", "suicide karna hai",
    "aatmahatya", "khudkushi",
    # Hindi (Devanagari)
    "मरना है", "मरना चाहता हूं", "मरना चाहती हूं", "आत्महत्या", "खुदकुशी",
    "जीना नहीं चाहता", "जीना नहीं चाहती", "खुद को खत्म",
]

CRISIS_KEYWORDS_MODERATE = [
    # English
    "self-harm", "self harm", "cut myself", "hurt myself",
    "no reason to live", "can't go on", "cannot go on",
    "don't want to be here", "disappear forever", "everyone would be better without me",
    "nothing matters anymore", "give up on life",
    # Hinglish
    "khud ko nuksan", "khud ko takleef", "jeena bekaar",
    "kuch matlab nahi reh gaya", "sab khatam karna hai", "bas khatam karna hai",
    # Hindi (Devanagari)
    "खुद को नुकसान", "जीना बेकार", "सब खत्म करना है",
]

# All crisis keywords combined for detection
ALL_CRISIS_KEYWORDS = CRISIS_KEYWORDS_HIGH + CRISIS_KEYWORDS_MODERATE


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────
def analyze_sentiment(text: str) -> str:
    """
    Keyword-based sentiment analysis.
    In a tie, leans negative — mental health context warrants caution.
    """
    text_lower = text.lower()
    pos_count = sum(1 for w in SENTIMENT_KEYWORDS["positive"] if w in text_lower)
    neg_count = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in text_lower)

    if neg_count >= pos_count and neg_count > 0:
        return "😢 Negative"
    elif pos_count > neg_count:
        return "😊 Positive"
    else:
        return "😐 Neutral"


# ─────────────────────────────────────────────
# CRISIS DETECTION
# ─────────────────────────────────────────────
def detect_crisis_keywords(text: str) -> bool:
    """Return True if any crisis keyword is found in the text."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ALL_CRISIS_KEYWORDS)


# ─────────────────────────────────────────────
# RISK SCORE
# ─────────────────────────────────────────────
def calculate_risk_score(text: str) -> float:
    """
    Calculate a risk score between 0.0 (low) and 1.0 (high).

    Scoring logic:
    - High-tier crisis keywords contribute 0.45 each (capped)
    - Moderate-tier crisis keywords contribute 0.25 each (capped)
    - Negative sentiment words contribute 0.07 each (capped at 0.30)
    - Scores are summed and clamped to [0.0, 1.0]
    """
    text_lower = text.lower()
    high_hits = sum(1 for w in CRISIS_KEYWORDS_HIGH if w in text_lower)
    moderate_hits = sum(1 for w in CRISIS_KEYWORDS_MODERATE if w in text_lower)
    neg_hits = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in text_lower)

    score = (
        min(high_hits * 0.45, 0.90) +
        min(moderate_hits * 0.25, 0.50) +
        min(neg_hits * 0.07, 0.30)
    )
    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# LLM RESPONSE GENERATION
# ─────────────────────────────────────────────
def build_enriched_input(user_input: str, sentiment: str, risk_score: float,
                          last_bot_reply: str = None, crisis_flag: bool = False) -> str:
    """
    Build the enriched prompt input. Only a genuine high-risk/crisis note
    gets injected — everything else is the user's raw message untouched.

    Earlier versions also injected a "moderate distress" note whenever
    risk_score > 0.4, and before that a literal sentiment tag on every
    message. Both were firing on ordinary messages (a handful of words
    like "tired"/"sad"/"lonely" easily crosses 0.4) and injecting the
    exact same boilerplate text each time, which is what was actually
    causing near-identical replies — the model was responding to the
    repeated instruction text more than to what the person said.

    The ALERT note now fires if EITHER risk_score > 0.7 OR crisis_flag is
    True. This matters because a single crisis keyword match (e.g. a Hindi
    phrase like "mujhe marna hai") only scores 0.45 under calculate_risk_score's
    weighting — relying on the score alone would silently miss it.

    If last_bot_reply is given, it's included so the model can deliberately
    avoid repeating the same opening phrase/style two turns in a row.
    """
    variety_note = ""
    if last_bot_reply:
        variety_note = (
            f"[Your previous reply was: \"{last_bot_reply}\" — "
            "do NOT open your new reply the same way or reuse its structure.]\n"
        )

    if risk_score > 0.7 or crisis_flag:
        return (
            f"{variety_note}"
            "[ALERT: Crisis/self-harm language detected. Drop all brevity and style rules. "
            "Respond with full seriousness and warmth. Clearly and immediately urge the user "
            "to contact a crisis helpline right now, and give the actual number "
            "(India: iCall 9152987821 or AASRA 9820466726; if the message isn't in Hindi/Hinglish, "
            "use 988 instead). Do not just suggest talking to friends/family as the main response — "
            "the helpline must be explicitly stated.]\n"
            f"User message: {user_input}"
        )
    return f"{variety_note}User message: {user_input}" if variety_note else user_input


def generate_counseling_response(
    chain,               # LangChain chain (prompt | llm | StrOutputParser)
    user_input: str,
    sentiment: str,
    risk_score: float,
    last_bot_reply: str = None,
    crisis_flag: bool = False
) -> str:
    """
    Invoke the LangChain chain with enriched context injected into the prompt.
    Falls back to a safe empathetic message if the chain fails.
    """
    enriched_input = build_enriched_input(user_input, sentiment, risk_score, last_bot_reply, crisis_flag)
    try:
        response = chain.invoke({"input": enriched_input})
        return response.strip() if isinstance(response, str) else str(response).strip()
    except Exception as e:
        sentiment_clean = sentiment.replace("😢 ", "").replace("😊 ", "").replace("😐 ", "").lower()
        return f"[Error: {str(e)}] — I hear that you're feeling {sentiment_clean}. Would you like to share more? ❤️"


def stream_counseling_response(chain, user_input: str, sentiment: str, risk_score: float,
                                last_bot_reply: str = None, crisis_flag: bool = False):
    """
    Same enrichment/context logic as generate_counseling_response, but yields
    the response incrementally for a live 'typing' effect in the UI.
    Falls back to a single-chunk safe message if streaming fails.
    """
    enriched_input = build_enriched_input(user_input, sentiment, risk_score, last_bot_reply, crisis_flag)
    try:
        for chunk in chain.stream({"input": enriched_input}):
            yield chunk if isinstance(chunk, str) else str(chunk)
    except Exception as e:
        sentiment_clean = sentiment.replace("😢 ", "").replace("😊 ", "").replace("😐 ", "").lower()
        yield f"[Error: {str(e)}] — I hear that you're feeling {sentiment_clean}. Would you like to share more? ❤️"
