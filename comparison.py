import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import joblib
import os

# --- Load models ---
vader = SentimentIntensityAnalyzer()

@st.cache_resource
def load_ml_model():
    model_path = os.path.join("models", "lr_pipeline.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_ml_model()

# --- VADER ---
def analyze_vader(text):
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(abs(compound), 2)

# --- ML Model ---
def analyze_ml(text):
    if model is None:
        return "Model not found", None
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    label = "Positive" if prediction == 1 else "Negative"
    confidence = round(max(proba), 2)
    return label, confidence

# --- TextBlob ---
def analyze_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(abs(polarity), 2)

# --- UI ---
st.set_page_config(page_title="Model Comparison", page_icon="🔬", layout="centered")
st.title("🔬 Sentiment Model Comparison")
st.caption("VADER vs Logistic Regression vs TextBlob")

text_input = st.text_area("Enter text to analyze:", placeholder="Type something here...", height=120)

if st.button("Compare All Models", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running all three models..."):
            vader_label, vader_conf = analyze_vader(text_input)
            ml_label, ml_conf = analyze_ml(text_input)
            tb_label, tb_conf = analyze_textblob(text_input)

        st.markdown("### Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("VADER", vader_label)
            st.caption(f"Confidence: {vader_conf}")

        with col2:
            st.metric("Logistic Regression", ml_label)
            if ml_conf:
                st.caption(f"Confidence: {ml_conf}")

        with col3:
            st.metric("TextBlob", tb_label)
            st.caption(f"Polarity: {tb_conf}")

        # Agreement check
        labels = [vader_label, ml_label, tb_label]
        labels_clean = [l for l in labels if l != "Model not found"]

        st.markdown("---")
        if len(set(labels_clean)) == 1:
            st.success("✅ All models agree — strong signal")
        elif vader_label == ml_label and tb_label != vader_label:
            st.warning(f"⚠️ VADER & ML say **{vader_label}** but TextBlob says **{tb_label}**")
        elif vader_label == tb_label and ml_label != vader_label:
            st.warning(f"⚠️ ML Model disagrees — predicted **{ml_label}** while others say **{vader_label}**")
        elif ml_label == tb_label and vader_label != ml_label:
            st.warning(f"⚠️ VADER disagrees — predicted **{vader_label}** while others say **{ml_label}**")
        else:
            st.error("❌ All three models disagree — ambiguous text")

        # Detailed scores
        with st.expander("View detailed VADER scores"):
            scores = vader.polarity_scores(text_input)
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Positive", round(scores["pos"], 3))
            col_b.metric("Negative", round(scores["neg"], 3))
            col_c.metric("Neutral", round(scores["neu"], 3))
            col_d.metric("Compound", round(scores["compound"], 3))

st.markdown("---")
st.caption("FYP — Real-Time Social Media Analytics System | CSI-630")