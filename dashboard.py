# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re, string, os

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data_collection    import load_dataset, create_sample_dataset, DEFAULT_DATASET_PATH
from preprocessing      import preprocess_dataframe
from sentiment_analysis import analyse_sentiment
from database           import (save_results, load_all_results,
                                 get_sentiment_counts, clear_database,
                                 initialise_database)

# ── New modules ─────────────────────────────────────────────────────
from ml_models      import load_metrics, models_exist, predict_sentiment
from wordcloud_gen  import generate_wordcloud_figure, generate_single_wordcloud
from hashtag_trends import get_top_hashtags, get_hashtag_sentiment_breakdown

# ── Global NLP setup
STOP_WORDS = set(stopwords.words("english"))
_vader     = SentimentIntensityAnalyzer()

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Sentiment Analytics",
    page_icon="📊",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        color: white; padding: 28px 35px; border-radius: 16px;
        margin-bottom: 28px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-banner h1 { margin: 0 0 8px 0; font-size: 1.9rem; font-weight: 700; }
    .main-banner p  { margin: 0; opacity: 0.8; font-size: 0.95rem; }
    .section-title {
        font-size: 1.2rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 3px solid #0f3460; padding-bottom: 8px; margin: 32px 0 18px 0;
    }
    .result-card {
        background: white; border-radius: 14px; padding: 24px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-top: 4px solid #667eea; margin-bottom: 16px;
    }
    .result-card h2 { font-size: 2.4rem; margin: 8px 0; }
    .result-card p  { color: #666; margin: 0; font-size: 0.9rem; }
    .positive-card  { border-top-color: #4CAF50 !important; }
    .negative-card  { border-top-color: #F44336 !important; }
    .neutral-card   { border-top-color: #2196F3 !important; }
    .info-box {
        background: #f8f9ff; border-left: 4px solid #667eea;
        padding: 14px 18px; border-radius: 8px; margin: 12px 0; font-size: 0.95rem;
    }
    .history-item {
        background: #fafafa; border: 1px solid #eee;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .stTextArea textarea {
        border-radius: 10px !important; border: 2px solid #e0e0e0 !important;
        font-size: 1rem !important; padding: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-banner">
    <h1>📊 Real-Time Social Media Sentiment & Trend Detection</h1>
    <p>CSI-630 Final Year Project &nbsp;·&nbsp;
       Govt. Municipal Graduate College, Faisalabad &nbsp;·&nbsp;
       Python · VADER · TextBlob · Random Forest · Logistic Regression · SQLite · Streamlit</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    st.markdown("---")
    st.subheader("1. Dataset")
    if os.path.exists(DEFAULT_DATASET_PATH):
        st.success(f"Found: `{DEFAULT_DATASET_PATH}`")
    else:
        st.warning("No dataset found.")
        if st.button("🔧 Generate Sample Dataset", use_container_width=True):
            with st.spinner("Creating sample data …"):
                create_sample_dataset()
            st.success("Sample dataset created!")
            st.rerun()

    st.markdown("---")
    st.subheader("2. Run Pipeline")
    run_pipeline = st.button("▶ Run Full Analysis Pipeline",
                              use_container_width=True,
                              help="Load → Preprocess → Analyse → Save to DB")
    st.markdown("---")
    st.subheader("3. ML Models")
    if models_exist():
        st.success("✅ ML models ready")
    else:
        st.warning("⚠️ Not trained yet")
        st.code("python train_models.py", language="bash")

    st.markdown("---")
    st.subheader("4. Utilities")
    if st.button("🗑️ Clear Database", use_container_width=True):
        clear_database()
        st.warning("Database cleared.")
        st.rerun()
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption("Python · VADER · TextBlob · Sklearn · SQLite · Streamlit")


# ══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════
if run_pipeline:
    st.info("⏳ Running pipeline …")
    progress = st.progress(0)
    with st.spinner("Step 1/4 — Loading dataset …"):
        try:
            raw_df = load_dataset(DEFAULT_DATASET_PATH)
            progress.progress(25)
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
    with st.spinner("Step 2/4 — Preprocessing text …"):
        clean_df = preprocess_dataframe(raw_df, text_col="text")
        progress.progress(50)
    with st.spinner("Step 3/4 — Analysing sentiment …"):
        result_df = analyse_sentiment(clean_df, text_col="clean_text")
        progress.progress(75)
    with st.spinner("Step 4/4 — Saving to database …"):
        n_saved = save_results(result_df)
        progress.progress(100)
    st.success(f"✅ Done! **{n_saved}** posts analysed and saved.")


# ══════════════════════════════════════════════════════════════════
# LOAD DATABASE
# ══════════════════════════════════════════════════════════════════
initialise_database()
db_df  = load_all_results()
counts = get_sentiment_counts()
total  = sum(counts.values())


# ══════════════════════════════════════════════════════════════════
# SECTION A — KPI METRICS
# ══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">📈 Sentiment Overview</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
pct = lambda n: f"{round(n/total*100,1)}%" if total > 0 else "0%"
c1.metric("📦 Total Posts",  f"{total:,}")
c2.metric("😊 Positive",     f"{counts['Positive']:,}", pct(counts["Positive"]))
c3.metric("😞 Negative",     f"{counts['Negative']:,}", pct(counts["Negative"]))
c4.metric("😐 Neutral",      f"{counts['Neutral']:,}",  pct(counts["Neutral"]))


# ══════════════════════════════════════════════════════════════════
# SECTION B — CHARTS
# ══════════════════════════════════════════════════════════════════
if total > 0:
    st.markdown('<p class="section-title">📊 Visualisations</p>', unsafe_allow_html=True)
    COLOURS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        labels = [k for k, v in counts.items() if v > 0]
        sizes  = [counts[k] for k in labels]
        clrs   = [COLOURS[k] for k in labels]
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=clrs,
            startangle=140, wedgeprops={"edgecolor":"white","linewidth":2},
            textprops={"fontsize":12})
        for at in autotexts:
            at.set_color("white"); at.set_fontweight("bold")
        ax.set_title("Overall Sentiment Split", fontsize=14, fontweight="bold")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Posts per Category")
        bar_labels = list(counts.keys())
        bar_values = list(counts.values())
        bar_clrs   = [COLOURS[k] for k in bar_labels]
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.bar(bar_labels, bar_values, color=bar_clrs, width=0.5,
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, bar_values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    str(val), ha="center", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sentiment Category", fontsize=12)
        ax.set_ylabel("Number of Posts",    fontsize=12)
        ax.set_title("Posts per Sentiment", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(bar_values)*1.2 if bar_values else 1)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    st.subheader("📉 Sentiment Trend Over Time")
    batch_size = max(10, total // 20)
    sorted_df  = db_df.sort_values("id").reset_index(drop=True)
    sorted_df["batch"] = sorted_df.index // batch_size
    trend_df = sorted_df.groupby("batch")["vader_compound"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(trend_df["batch"], trend_df["vader_compound"],
            color="#9C27B0", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.fill_between(trend_df["batch"], trend_df["vader_compound"], 0,
                    where=(trend_df["vader_compound"] >= 0),
                    alpha=0.15, color="#4CAF50", label="Positive zone")
    ax.fill_between(trend_df["batch"], trend_df["vader_compound"], 0,
                    where=(trend_df["vader_compound"] < 0),
                    alpha=0.15, color="#F44336", label="Negative zone")
    ax.set_xlabel("Batch Number", fontsize=11)
    ax.set_ylabel("Avg VADER Compound", fontsize=11)
    ax.set_title("Sentiment Trend Across Post Batches", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()
else:
    st.info("No data yet. Click **▶ Run Full Analysis Pipeline** in the sidebar.")


# ══════════════════════════════════════════════════════════════════
# SECTION C — RECENT POSTS TABLE
# ══════════════════════════════════════════════════════════════════
if total > 0:
    st.markdown('<p class="section-title">🗂️ Recent Posts</p>', unsafe_allow_html=True)
    n_show = st.slider("Rows to display:", 5, 100, 20, step=5)
    cols   = ["id","raw_text","clean_text","sentiment","vader_compound","tb_polarity","created_at"]
    avail  = [c for c in cols if c in db_df.columns]
    recent = db_df.sort_values("id", ascending=False).head(n_show)[avail]
    def _hl(val):
        return {"Positive":"background-color:#c8e6c9;color:#1b5e20;",
                "Negative":"background-color:#ffcdd2;color:#b71c1c;",
                "Neutral": "background-color:#bbdefb;color:#0d47a1;"}.get(val,"")
    st.dataframe(recent.style.applymap(_hl, subset=["sentiment"]),
                 use_container_width=True, height=400)
    st.download_button("⬇️ Download as CSV",
                       data=recent.to_csv(index=False).encode("utf-8"),
                       file_name="sentiment_results.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# SECTION D — SINGLE POST ANALYSER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">🧠 Live Single Post Analyser</p>', unsafe_allow_html=True)

_EMOJI_RE = re.compile(
    "[""\U0001F600-\U0001F64F""\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF""\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0""\U000024C2-\U0001F251""]+",
    flags=re.UNICODE,
)

def _clean_post(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = _EMOJI_RE.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()

def _analyse_post(text):
    cleaned  = _clean_post(text)
    compound = _vader.polarity_scores(cleaned)["compound"]
    blob     = TextBlob(cleaned)
    polarity = blob.sentiment.polarity
    subj     = blob.sentiment.subjectivity
    avg      = (compound + polarity) / 2.0
    label    = "Positive" if avg >= 0.05 else ("Negative" if avg <= -0.05 else "Neutral")
    return {"cleaned": cleaned, "compound": round(compound,4),
            "polarity": round(polarity,4), "subjectivity": round(subj,4),
            "avg": round(avg,4), "label": label}

def _color(label):
    return {"Positive":"#4CAF50","Negative":"#F44336","Neutral":"#2196F3"}.get(label,"#999")

def _emoji(label):
    return {"Positive":"😊","Negative":"😞","Neutral":"😐"}.get(label,"")

if "post_history" not in st.session_state:
    st.session_state.post_history = []

left_col, right_col = st.columns([3, 2], gap="large")
with left_col:
    st.markdown("##### ✍️ Paste any social media post, tweet, or comment below")
    user_input = st.text_area(
        label="", placeholder="e.g. Real Madrid lost to Osasuna in a shocking upset tonight...",
        height=160, label_visibility="collapsed", key="post_input",
    )
    st.markdown("**💡 Try an example:**")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("😊 Positive"):
        st.session_state["_ex"] = "I absolutely love this! Best experience ever, highly recommend it!"
    if ex2.button("😞 Negative"):
        st.session_state["_ex"] = "Terrible service, completely disappointed. Never coming back again!"
    if ex3.button("😐 Neutral"):
        st.session_state["_ex"] = "The meeting has been rescheduled to Thursday at 3 PM."
    if "_ex" in st.session_state:
        user_input = st.session_state["_ex"]
        del st.session_state["_ex"]
        st.rerun()
    analyse_btn = st.button("🧠 Analyse Sentiment Now", key="analyse_post_btn")

if analyse_btn and user_input.strip():
    result = _analyse_post(user_input)
    label  = result["label"]
    color  = _color(label)
    emoji  = _emoji(label)
    st.session_state.post_history.insert(0, {
        "text":  user_input[:80] + "..." if len(user_input) > 80 else user_input,
        "label": label, "score": result["avg"],
    })
    if len(st.session_state.post_history) > 10:
        st.session_state.post_history = st.session_state.post_history[:10]

    with right_col:
        st.markdown(f"""
        <div class="result-card {label.lower()}-card">
            <p style="font-size:1rem;color:#555;margin-bottom:4px">Final Verdict</p>
            <h2>{emoji} {label}</h2>
            <p>Ensemble Score: <b style="color:{color}">{result['avg']}</b></p>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("VADER", result["compound"],
                  delta="↑ Positive" if result["compound"] > 0 else ("↓ Negative" if result["compound"] < 0 else "—"))
        m2.metric("TextBlob",     result["polarity"])
        m3.metric("Subjectivity", result["subjectivity"], help="0=Objective · 1=Subjective")

        st.markdown("**Sentiment Intensity Gauge**")
        normalised = (result["avg"] + 1) / 2
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(0, 1, color="#eee", height=0.5)
        ax.barh(0, normalised, color=color, height=0.5)
        ax.axvline(x=0.5, color="#888", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["Very\nNegative","Negative","Neutral","Positive","Very\nPositive"], fontsize=8)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.set_title("Sentiment Position", fontsize=10, fontweight="bold")
        st.pyplot(fig); plt.close()

        st.markdown("**Model Score Comparison**")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        models = ["VADER\nCompound","TextBlob\nPolarity","Ensemble\nAverage"]
        scores = [result["compound"], result["polarity"], result["avg"]]
        clrs   = [_color(label) if s >= 0 else "#F44336" for s in scores]
        bars   = ax.barh(models, scores, color=clrs, height=0.5)
        ax.axvline(x=0, color="#888", linewidth=1); ax.set_xlim(-1, 1)
        for bar, score in zip(bars, scores):
            ax.text(score + (0.03 if score >= 0 else -0.03),
                    bar.get_y()+bar.get_height()/2, str(score), va="center",
                    ha="left" if score >= 0 else "right", fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title("VADER vs TextBlob vs Ensemble", fontsize=10, fontweight="bold")
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="info-box">
            <b>🔍 Cleaned Text:</b><br>
            <span style="color:#444">{result['cleaned'] if result['cleaned'] else '(empty after cleaning)'}</span>
        </div>
        """, unsafe_allow_html=True)

elif analyse_btn:
    st.warning("⚠️ Please enter some text before clicking Analyse.")
else:
    with right_col:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#aaa;">
            <div style="font-size:4rem">🧠</div>
            <p style="font-size:1.1rem;margin-top:12px">
                Paste a post on the left<br>and click <b>Analyse Sentiment Now</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.post_history:
    st.markdown("---")
    st.markdown("### 📋 Recent Analysis History")
    st.caption("Last 10 posts analysed in this session.")
    for item in st.session_state.post_history:
        color = _color(item["label"]); emoji = _emoji(item["label"])
        st.markdown(f"""
        <div class="history-item">
            <span style="background:{color};color:white;padding:2px 10px;
                  border-radius:12px;font-size:0.8rem;font-weight:600">{emoji} {item['label']}</span>
            &nbsp;&nbsp;<span style="color:#333">{item['text']}</span>
            &nbsp;&nbsp;<span style="color:#999;font-size:0.8rem">Score: {item['score']}</span>
        </div>
        """, unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.post_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# SECTION E — ML MODEL PERFORMANCE  ★ NEW
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">🤖 ML Model Performance — Random Forest & Logistic Regression</p>',
            unsafe_allow_html=True)
st.caption("Models trained on Sentiment140 — 1.6 million labelled tweets (Stanford NLP).")

if not models_exist():
    st.warning("⚠️ ML models not trained yet. Run the command below once:")
    st.code("python train_models.py", language="bash")
    st.markdown("Training downloads Sentiment140 (~80 MB) and takes ~10 minutes. "
                "After that, this entire section fills in automatically.")
else:
    metrics = load_metrics()
    if not metrics:
        st.error("Metrics file missing — re-run `python train_models.py`")
    else:
        model_names = list(metrics.keys())

        # ── Accuracy + F1 metrics ─────────────────────────────────
        metric_cols = st.columns(len(model_names) * 2)
        for i, name in enumerate(model_names):
            m = metrics[name]
            metric_cols[i*2].metric(
                f"🎯 {name} Accuracy", f"{m['accuracy']*100:.2f}%",
                delta=f"+{(m['accuracy']-0.5)*100:.1f}% vs random"
            )
            metric_cols[i*2+1].metric(f"📐 {name} F1", f"{m['f1_score']*100:.2f}%")

        # ── Confusion matrices ────────────────────────────────────
        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(model_names))
        for col, name in zip(cm_cols, model_names):
            with col:
                cm = np.array(metrics[name]["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["Pred Neg","Pred Pos"], fontsize=10)
                ax.set_yticklabels(["True Neg","True Pos"], fontsize=10)
                ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
                for r in range(2):
                    for c in range(2):
                        ax.text(c, r, f"{cm[r,c]:,}", ha="center", va="center",
                                fontsize=13, fontweight="bold",
                                color="white" if cm[r,c] > cm.max()/2 else "black")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        # ── Comparison bar chart ──────────────────────────────────
        st.markdown("#### Accuracy vs F1 Score — Side by Side")
        x = np.arange(2)   # Accuracy, F1
        w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_palette = ["#0f3460", "#e94560"]
        for i, name in enumerate(model_names):
            vals = [metrics[name]["accuracy"]*100, metrics[name]["f1_score"]*100]
            bars = ax.bar(x + i*w - w/2, vals, w, label=name,
                          color=bar_palette[i], alpha=0.88, edgecolor="white", linewidth=1.2)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "F1 Score"], fontsize=12)
        ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
        ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

        # ── Live prediction on DB data ────────────────────────────
        if total > 0 and "raw_text" in db_df.columns:
            st.markdown("#### Live ML Prediction on Current Dataset Posts")
            sample_texts = db_df["raw_text"].dropna().head(300).tolist()
            chosen = st.radio("Model:", ["Logistic Regression", "Random Forest"],
                              horizontal=True, key="ml_live_radio")
            model_key = "lr" if chosen == "Logistic Regression" else "rf"

            with st.spinner(f"Running {chosen} on {len(sample_texts)} posts …"):
                ml_preds  = predict_sentiment(sample_texts, model=model_key)
                ml_counts = pd.Series(ml_preds).value_counts()

            pie_col, stat_col = st.columns(2)
            with pie_col:
                colours = {"Positive":"#4CAF50","Negative":"#F44336","Neutral":"#2196F3"}
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(ml_counts.values, labels=ml_counts.index, autopct="%1.1f%%",
                       colors=[colours.get(k,"#aaa") for k in ml_counts.index],
                       startangle=140, wedgeprops={"edgecolor":"white","linewidth":2},
                       textprops={"fontsize":11})
                ax.set_title(f"{chosen} Predictions", fontsize=12, fontweight="bold")
                st.pyplot(fig); plt.close()
            with stat_col:
                for lbl, cnt in ml_counts.items():
                    st.metric(f"{_emoji(lbl)} {lbl}", f"{cnt:,}",
                              f"{cnt/len(ml_preds)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION F — WORD CLOUD  ★ NEW
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">☁️ Word Cloud — Positive vs Negative Words</p>',
            unsafe_allow_html=True)
st.caption("Most frequent meaningful words in positive and negative posts.")

if total == 0:
    st.info("Run the pipeline first to generate word clouds.")
else:
    # Find best available text column
    _tcol = next((c for c in ["raw_text","clean_text"] if c in db_df.columns), None)

    if _tcol and "sentiment" in db_df.columns:
        pos_texts = db_df.loc[db_df["sentiment"]=="Positive", _tcol].dropna().tolist()
        neg_texts = db_df.loc[db_df["sentiment"]=="Negative", _tcol].dropna().tolist()

        wcc1, wcc2, wcc3 = st.columns(3)
        wcc1.metric("😊 Positive Posts", f"{len(pos_texts):,}")
        wcc2.metric("😞 Negative Posts", f"{len(neg_texts):,}")
        wcc3.metric("Total", f"{len(pos_texts)+len(neg_texts):,}")

        view_mode = st.radio("Display:", ["Side by Side","Positive Only","Negative Only"],
                             horizontal=True, key="wc_view")
        max_words = st.slider("Max words per cloud:", 50, 200, 100, key="wc_max")

        with st.spinner("Generating word cloud …"):
            try:
                if view_mode == "Side by Side":
                    fig = generate_wordcloud_figure(pos_texts, neg_texts, max_words=max_words)
                elif view_mode == "Positive Only":
                    fig = generate_single_wordcloud(pos_texts, "positive", max_words)
                else:
                    fig = generate_single_wordcloud(neg_texts, "negative", max_words)
                st.pyplot(fig, use_container_width=True); plt.close()
            except Exception as e:
                st.error(f"Word cloud error: {e}")
                st.code("python -m pip install wordcloud --only-binary=all", language="bash")
    else:
        st.info("Text or sentiment column missing. Run the pipeline first.")


# ══════════════════════════════════════════════════════════════════
# SECTION G — HASHTAG TRENDS  ★ NEW
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">📊 Trending Hashtags</p>', unsafe_allow_html=True)
st.caption("Top hashtags ranked by frequency, with sentiment breakdown per hashtag.")

if total == 0:
    st.info("Run the pipeline first to see hashtag trends.")
else:
    _tcol = next((c for c in ["raw_text","clean_text"] if c in db_df.columns), None)
    if _tcol:
        all_texts = db_df[_tcol].dropna().tolist()
        hc1, hc2 = st.columns(2)
        top_n   = hc1.slider("Top N hashtags:", 5, 30, 15, key="ht_topn")
        min_cnt = hc2.slider("Min occurrences:", 1, 10, 2, key="ht_min")

        ht_df = get_top_hashtags(all_texts, top_n=top_n, min_count=min_cnt)

        if ht_df.empty:
            st.info("No hashtags found. Dataset needs posts containing # symbols "
                    "(e.g. tweets). Try loading a Twitter dataset.")
        else:
            # ── Horizontal bar chart ──────────────────────────────
            fig, ax = plt.subplots(figsize=(11, max(4, len(ht_df)*0.45)))
            bar_colours = plt.cm.viridis(np.linspace(0.2, 0.85, len(ht_df)))[::-1]
            bars = ax.barh(ht_df["hashtag"][::-1], ht_df["count"][::-1],
                           color=bar_colours, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, ht_df["count"][::-1]):
                ax.text(bar.get_width() + max(ht_df["count"])*0.01,
                        bar.get_y()+bar.get_height()/2, str(val),
                        va="center", fontsize=9, fontweight="bold")
            ax.set_xlabel("Occurrences", fontsize=11)
            ax.set_title(f"Top {len(ht_df)} Trending Hashtags", fontsize=14, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xlim(0, max(ht_df["count"])*1.18)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

            # ── Sentiment breakdown per hashtag ───────────────────
            if "sentiment" in db_df.columns:
                sentiments  = db_df["sentiment"].tolist()
                ht_sent_df  = get_hashtag_sentiment_breakdown(all_texts, sentiments, top_n=top_n)

                if not ht_sent_df.empty:
                    st.markdown("#### Sentiment Breakdown per Hashtag")
                    fig, ax = plt.subplots(figsize=(11, 5))
                    x = np.arange(len(ht_sent_df)); w = 0.28
                    ax.bar(x - w, ht_sent_df["Positive"], w, label="Positive",
                           color="#4CAF50", alpha=0.88)
                    ax.bar(x,     ht_sent_df["Negative"], w, label="Negative",
                           color="#F44336", alpha=0.88)
                    ax.bar(x + w, ht_sent_df["Neutral"],  w, label="Neutral",
                           color="#2196F3", alpha=0.88)
                    ax.set_xticks(x)
                    ax.set_xticklabels(ht_sent_df["hashtag"], rotation=35, ha="right", fontsize=9)
                    ax.set_ylabel("Post Count"); ax.legend(fontsize=10)
                    ax.set_title("Hashtag Sentiment Breakdown", fontsize=13, fontweight="bold")
                    ax.spines[["top","right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True); plt.close()

                    with st.expander("📋 Hashtag Data Table"):
                        show_df = ht_sent_df.copy()
                        show_df["pos_ratio"] = show_df["pos_ratio"].astype(str) + "%"
                        show_df.columns = ["Hashtag","Total","Positive","Negative","Neutral","Positivity %"]
                        st.dataframe(show_df, use_container_width=True)
    else:
        st.info("Text column not found in database.")


# ══════════════════════════════════════════════════════════════════
# SECTION H — ABOUT
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("ℹ️ About This Project"):
    st.markdown("""
**Project Title:** Real-Time Social Media Analytics for Sentiment and Trend Detection  
**Course:** CSI-630 | Govt. Municipal Graduate College, Faisalabad

**Tech Stack:** Python · Pandas · NumPy · VADER · TextBlob · Scikit-learn · SQLite · Streamlit · Matplotlib · WordCloud

**Modules:**
1. `data_collection.py`    — Load CSV dataset  
2. `preprocessing.py`      — Clean & tokenize text  
3. `sentiment_analysis.py` — VADER + TextBlob ensemble  
4. `database.py`           — SQLite storage  
5. `ml_models.py`          — Random Forest + Logistic Regression on Sentiment140  
6. `wordcloud_gen.py`      — Positive/negative word clouds  
7. `hashtag_trends.py`     — Hashtag extraction & ranking  
8. `train_models.py`       — One-time ML training script  
9. `dashboard.py`          — This interactive dashboard  

**ML Setup:** Run `python train_models.py` once. Downloads Sentiment140 (1.6M tweets), trains both models, saves to `models/`.
    """)
