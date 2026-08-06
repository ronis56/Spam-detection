"""
SpamShield — Streamlit UI for a trained TF-IDF + Logistic Regression
spam classifier (matches the pipeline in the user's notebook).

Expects these two files (produced by the notebook) in the same folder
as this script:
    - best_model.pkl   (trained LogisticRegression model)
    - tf.pkl            (fitted TfidfVectorizer)

Run with:
    pip install streamlit scikit-learn pandas nltk joblib
    streamlit run spam_detector_app.py
"""

import os
import string
import time
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SpamShield | Email Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Do NOT hide the header */

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1150px;
        }

        .hero {
            padding: 1.75rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #4338ca 0%, #6366f1 45%, #818cf8 100%);
            color: white;
            margin-bottom: 1.75rem;
            box-shadow: 0 10px 30px rgba(67, 56, 202, 0.25);
        }
        .hero h1 {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0 0 0.25rem 0;
            letter-spacing: -0.02em;
        }
        .hero p {
            font-size: 0.98rem;
            opacity: 0.92;
            margin: 0;
            font-weight: 400;
        }

        .card {
            background: #ffffff;
            border: 1px solid #eef0f5;
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 2px 10px rgba(17, 24, 39, 0.04);
        }
        .card h3 {
            margin-top: 0;
            font-size: 1.02rem;
            font-weight: 700;
            color: #1f2937;
        }

        .verdict-box {
            border-radius: 14px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .verdict-spam {
            background: linear-gradient(135deg, #fef2f2, #fee2e2);
            border: 1px solid #fecaca;
        }
        .verdict-ham {
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border: 1px solid #bbf7d0;
        }
        .verdict-label {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }
        .verdict-spam .verdict-label { color: #b91c1c; }
        .verdict-ham .verdict-label { color: #15803d; }
        .verdict-sub {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        .chip {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #eef2ff;
            color: #4338ca;
            margin-right: 0.4rem;
        }

        section[data-testid="stSidebar"] {
            background: #111827;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.5rem;
        }
        /* Text color for headings/paragraphs only — NOT inside inputs/widgets,
           so widget internals (sliders, dropdowns) keep readable contrast */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stCaption {
            color: #e5e7eb !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #2d3748;
            margin: 0.9rem 0;
        }
        /* Slider track/value */
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            padding-top: 0.3rem;
        }
        section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
            color: #9ca3af !important;
        }
        /* Sidebar buttons — solid, readable, consistent */
        section[data-testid="stSidebar"] .stButton button {
            background: #4338ca !important;
            color: #ffffff !important;
            border: 1px solid #4f46e5 !important;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: #4f46e5 !important;
            border-color: #6366f1 !important;
        }
        section[data-testid="stSidebar"] .stButton button p {
            color: #ffffff !important;
        }

        .stButton button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
            border: none;
        }
        .stButton button[kind="primary"] {
            background: #4338ca;
        }

        .stTextArea textarea {
            border-radius: 10px;
            border: 1.5px solid #e5e7eb;
            font-size: 0.95rem;
        }

        .stDataFrame { border-radius: 10px; overflow: hidden; }

        .status-banner {
            padding: 0.7rem 1rem;
            border-radius: 10px;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }
        .status-ok {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
        }
        .status-missing {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Preprocessing — must exactly match the notebook's `preprocessing()`
# --------------------------------------------------------------------------
@st.cache_resource
def load_nltk_resources():
    import nltk

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    return set(stopwords.words("english")), PorterStemmer()


STOP_WORDS, STEMMER = load_nltk_resources()


def preprocessing(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    words = [STEMMER.stem(word) for word in words]
    return " ".join(words)


# --------------------------------------------------------------------------
# Load the trained model + vectorizer produced by the notebook
# --------------------------------------------------------------------------
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tf.pkl"


@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer, True
    return None, None, False


model, vectorizer, artifacts_found = load_artifacts()


def classify(raw_text: str):
    clean_text = preprocessing(raw_text)
    X = vectorizer.transform([clean_text])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        spam_prob = float(proba[1])
    else:
        spam_prob = 1.0 if pred == 1 else 0.0

    return spam_prob


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # each item: {"id", "Time", "Message", "Verdict", "Spam probability"}
if "history_counter" not in st.session_state:
    st.session_state.history_counter = 0

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ SpamShield")
    st.caption("Email spam classifier")
    st.markdown("---")

    st.markdown("**Model**")
    st.write("TF-IDF + Logistic Regression")
    st.write("F1 score: **0.95** (from training)")

    st.markdown("**Sensitivity threshold**")
    threshold = st.slider("Spam probability cutoff", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Try an example**")
    examples = {
        "Prize scam": "Subject: you have won! click here now to claim your free prize and cash reward immediately",
        "Work email": "Subject: quarterly report attached, please review and send feedback by friday, thanks",
        "Loan spam": "Subject: you are pre approved for a loan no credit check required apply now for free cash",
        "Meeting note": "Subject: re: schedule, can we move our meeting to thursday afternoon instead of wednesday",
    }
    for label, sample in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["input_text"] = sample

    st.markdown("---")
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []

    st.markdown("---")
    st.caption("Built on the TF-IDF + Logistic Regression pipeline from the training notebook.")

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SpamShield — Email Spam Classifier Developed By CryptoRaja</h1>
        <p>Paste an email below and instantly check whether it looks like spam, powered by your trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Artifact status banner
# --------------------------------------------------------------------------
if artifacts_found:
    st.markdown(
        '<div class="status-banner status-ok">✅ Loaded trained model artifacts '
        f'(<code>{MODEL_PATH}</code>, <code>{VECTORIZER_PATH}</code>).</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-banner status-missing">⚠️ Could not find '
        f'<code>{MODEL_PATH}</code> and <code>{VECTORIZER_PATH}</code> in this folder. '
        "Run the training notebook so <code>joblib.dump(...)</code> saves them here, then restart the app.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
col_input, col_result = st.columns([1.2, 1], gap="large")

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✏️ Email to analyze")
    text_input = st.text_area(
        "Message",
        value=st.session_state.get("input_text", ""),
        height=220,
        placeholder="Paste an email here, e.g. Subject: ...",
        label_visibility="collapsed",
    )
    analyze_clicked = st.button(
        "🔍 Analyze Email",
        type="primary",
        use_container_width=True,
        disabled=not artifacts_found,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Result")

    if analyze_clicked and text_input.strip():
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            spam_prob = classify(text_input)
            label = "Spam" if spam_prob >= threshold else "Not Spam"
            confidence = spam_prob if label == "Spam" else 1 - spam_prob

        st.session_state.history_counter += 1
        st.session_state.history.insert(
            0,
            {
                "id": st.session_state.history_counter,
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Message": text_input[:60] + ("..." if len(text_input) > 60 else ""),
                "Verdict": label,
                "Spam probability": f"{spam_prob:.0%}",
            },
        )

        box_class = "verdict-spam" if label == "Spam" else "verdict-ham"
        icon = "🚫" if label == "Spam" else "✅"
        st.markdown(
            f"""
            <div class="verdict-box {box_class}">
                <div class="verdict-label">{icon} {label}</div>
                <div class="verdict-sub">Confidence: {confidence:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Spam probability**")
        st.progress(spam_prob)
        m1, m2 = st.columns(2)
        m1.metric("Spam score", f"{spam_prob:.0%}")
        m2.metric("Ham score", f"{1 - spam_prob:.0%}")

    elif analyze_clicked:
        st.warning("Please enter an email to analyze.")
    else:
        st.info("Results will appear here once you analyze an email.")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# History table — with per-row delete
# --------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

header_col, action_col = st.columns([4, 1])
with header_col:
    st.markdown("### 🕘 Recent checks")
with action_col:
    if st.session_state.history:
        if st.button("🗑️ Clear all", use_container_width=True):
            st.session_state.history = []
            st.rerun()

if st.session_state.history:
    # Column header row
    c1, c2, c3, c4, c5 = st.columns([1, 3, 1.2, 1.2, 0.6])
    c1.markdown("**Time**")
    c2.markdown("**Message**")
    c3.markdown("**Verdict**")
    c4.markdown("**Spam prob.**")
    c5.markdown("**​**")  # empty header for delete column

    for entry in st.session_state.history:
        c1, c2, c3, c4, c5 = st.columns([1, 3, 1.2, 1.2, 0.6])
        c1.write(entry["Time"])
        c2.write(entry["Message"])
        badge = "🚫 Spam" if entry["Verdict"] == "Spam" else "✅ Not Spam"
        c3.write(badge)
        c4.write(entry["Spam probability"])
        if c5.button("✕", key=f"delete_{entry['id']}", help="Delete this entry"):
            st.session_state.history = [
                h for h in st.session_state.history if h["id"] != entry["id"]
            ]
            st.rerun()
else:
    st.caption("No emails analyzed yet in this session.")
st.markdown('</div>', unsafe_allow_html=True)