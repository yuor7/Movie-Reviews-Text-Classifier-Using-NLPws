# ============================================================
# Movie Review Classifier — PRODUCTION VERSION
# ============================================================

import re
import string
import streamlit as st
import nltk
import pickle
import plotly.graph_objects as go

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Review Classifier",
    page_icon="🎬",
    layout="wide"
)

# ─── Download NLTK ───────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ['punkt', 'stopwords']:
        nltk.download(pkg, quiet=True)

download_nltk()

# ─── Load Model ──────────────────────────────────────────────
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ─── Preprocessing ───────────────────────────────────────────
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# ─── UI ──────────────────────────────────────────────────────
st.title("🎬 Movie Review Classifier")
st.write("Classify reviews into Positive or Negative")

user_input = st.text_area(
    "Enter a movie review:",
    placeholder="This movie was amazing..."
)

if st.button("🔍 Predict"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        confidence = max(proba) * 100

        # Display result
        if prediction.lower() == "positive":
            st.success(f"😊 Positive ({confidence:.2f}%)")
        else:
            st.error(f"😠 Negative ({confidence:.2f}%)")

        # Chart
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in proba],
            y=model.classes_,
            orientation='h',
            text=[f"{p*100:.1f}%" for p in proba],
            textposition='outside'
        ))

        st.plotly_chart(fig, use_container_width=True)
