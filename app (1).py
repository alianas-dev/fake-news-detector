import streamlit as st
import pickle
import re
import nltk
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f1117; }

    .header-block {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .header-block h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .header-block p {
        color: #8b8fa8;
        font-size: 0.95rem;
        margin: 0;
    }

    .verdict-real {
        background: linear-gradient(135deg, #0d2b1a, #143d25);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #2b0d0d, #3d1414);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .verdict-label {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 0.4rem;
    }
    .verdict-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: #a0a8b8;
    }

    .info-box {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.2rem;
        font-size: 0.88rem;
        color: #8b8fa8;
        line-height: 1.6;
    }
    .info-box b { color: #c0c4d8; }

    .examples-label {
        font-size: 0.8rem;
        color: #5a5f78;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .stTextArea textarea {
        background-color: #1a1d2e !important;
        border: 1px solid #2a2d3e !important;
        color: #e0e4f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #4f6ef7 !important;
        box-shadow: 0 0 0 2px rgba(79, 110, 247, 0.15) !important;
    }

    .stButton > button {
        width: 100%;
        background: #4f6ef7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #3d5ce0;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ── Text cleaner (same as training pipeline) ──────────────────────────────────
@st.cache_data
def get_stopwords():
    return set(stopwords.words('english'))

def clean(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = get_stopwords()
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(text: str, model, vectorizer):
    cleaned = clean(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100
    return label, confidence

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🔍 Fake News Detector</h1>
    <p>Paste a news article or headline — the model will classify it as Real or Fake</p>
</div>
""", unsafe_allow_html=True)

# Check model files exist
if not os.path.exists('fake_news_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
    st.error("⚠️  Model files not found. Make sure `fake_news_model.pkl` and `tfidf_vectorizer.pkl` are in the same folder as `app.py`.")
    st.stop()

model, vectorizer = load_model()

# Example buttons
st.markdown('<p class="examples-label">Try an example</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("📰 Real news example"):
        st.session_state['input_text'] = "Scientists at NASA confirmed the James Webb Space Telescope has captured the deepest infrared image of the universe ever taken, revealing thousands of galaxies."
with col2:
    if st.button("🚨 Fake news example"):
        st.session_state['input_text'] = "BREAKING: Government secretly adding mind-control chemicals to drinking water, whistleblower reveals shocking truth that mainstream media refuses to cover."

# Text input
default_text = st.session_state.get('input_text', '')
user_input = st.text_area(
    label="News text",
    value=default_text,
    placeholder="Paste a news headline or article here...",
    height=160,
    label_visibility="collapsed"
)

# Predict button
if st.button("Analyse"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif len(user_input.strip().split()) < 5:
        st.warning("Text is too short — paste a full headline or paragraph for better accuracy.")
    else:
        with st.spinner("Analysing..."):
            label, confidence = predict(user_input, model, vectorizer)

        if label == 'Real':
            st.markdown(f"""
            <div class="verdict-real">
                <div class="verdict-label" style="color:#22c55e;">✓ REAL NEWS</div>
                <div class="verdict-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-fake">
                <div class="verdict-label" style="color:#ef4444;">✗ FAKE NEWS</div>
                <div class="verdict-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(confidence))

# Info box
st.markdown("""
<div class="info-box">
    <b>How it works</b><br>
    Text is cleaned (lowercased, stopwords removed, lemmatized) then converted to TF-IDF features.
    A Random Forest classifier trained on 1,680 labeled news articles predicts the label.<br><br>
    <b>Limitation:</b> The model was trained on a small, domain-specific dataset.
    It works best on English political and general news. It may not generalise well to
    highly technical, scientific, or non-English content.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<br>
<div style="text-align:center; color:#3a3f58; font-size:0.8rem;">
    Built by <b style="color:#4f6ef7">Mohammad Anas (Ali Anas)</b> · KIET · 2026
</div>
""", unsafe_allow_html=True)
