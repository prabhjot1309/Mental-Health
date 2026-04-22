import re
from typing import Optional

# ─────────────────────────────────────────────
# KEYWORD LISTS
# ─────────────────────────────────────────────

SENTIMENT_KEYWORDS = {
    "positive": [
        "happy", "good", "great", "excited", "grateful", "blessed",
        "hopeful", "joyful", "content", "relieved", "motivated", "better",
        "calm", "peaceful", "proud", "loved", "supported"
    ],
    "negative": [
        "sad", "depressed", "anxious", "stressed", "overwhelmed", "hopeless",
        "lonely", "empty", "worthless", "trapped", "tired", "exhausted",
        "scared", "angry", "numb", "broken", "lost", "afraid", "hurt"
    ],
    "neutral": [
        "okay", "fine", "normal", "alright", "average", "so-so"
    ]
}

# Tiered crisis keywords — high weight vs moderate weight
CRISIS_KEYWORDS_HIGH = [
    "suicide", "kill myself", "end my life", "take my life",
    "better off dead", "want to die", "going to die", "plan to die",
    "overdose", "slit my wrists", "hang myself", "jump off"
]

CRISIS_KEYWORDS_MODERATE = [
    "self-harm", "self harm", "cut myself", "hurt myself",
    "no reason to live", "can't go on", "cannot go on",
    "don't want to be here", "disappear forever", "everyone would be better without me",
    "nothing matters anymore", "give up on life"
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

    high_hits     = sum(1 for w in CRISIS_KEYWORDS_HIGH     if w in text_lower)
    moderate_hits = sum(1 for w in CRISIS_KEYWORDS_MODERATE if w in text_lower)
    neg_hits      = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in text_lower)

    score = (
        min(high_hits     * 0.45, 0.90) +  # one high keyword → near-crisis
        min(moderate_hits * 0.25, 0.50) +  # moderate keywords add up
        min(neg_hits      * 0.07, 0.30)    # general negativity
    )

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# LLM RESPONSE GENERATION
# ─────────────────────────────────────────────

def generate_counseling_response(
    chain,          # LangChain chain (prompt | llm | StrOutputParser)
    user_input: str,
    sentiment: str,
    risk_score: float
) -> str:
    """
    Invoke the LangChain chain with enriched context injected into the prompt.
    Falls back to a safe empathetic message if the chain fails.
    """
    # Build enriched input that slots into the chain's {input} variable
    risk_context = ""
    if risk_score > 0.7:
        risk_context = (
            "[ALERT: High risk detected. Prioritise crisis resources and urge the user "
            "to call a helpline immediately. Keep tone calm and non-alarming.]"
        )
    elif risk_score > 0.4:
        risk_context = (
            "[Note: Moderate distress detected. Be extra empathetic and suggest "
            "professional support gently.]"
        )

    enriched_input = (
        f"{risk_context}\n"
        f"User sentiment signal: {sentiment}\n"
        f"User message: {user_input}"
    ).strip()

    try:
        response = chain.invoke({"input": enriched_input})
        return response.strip() if isinstance(response, str) else str(response).strip()

    except Exception as e:
        # Safe fallback — never expose raw exception to user
        sentiment_clean = sentiment.replace("😢 ", "").replace("😊 ", "").replace("😐 ", "").lower()
        return (
            f"I hear that you're feeling {sentiment_clean} right now, and I want you to know "
            f"that what you're going through is valid. I'm here to listen — would you like to "
            f"share a bit more about what's been on your mind? ❤️"
        )


# ─────────────────────────────────────────────
# CRISIS RESOURCES
# ─────────────────────────────────────────────
