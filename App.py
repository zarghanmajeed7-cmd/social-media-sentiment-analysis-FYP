# ============================================================
# app.py — Unified Entry Point
# Real-Time Social Media Analytics for Sentiment and Trend Detection
# CSI-630 | Govt. Municipal Graduate College, Faisalabad
#
# RUN:
#   python -m streamlit run app.py
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Social Media Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Tab Navigation ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .top-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        color: white; padding: 22px 35px; border-radius: 16px;
        margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .top-banner h1 { margin: 0 0 6px 0; font-size: 1.7rem; font-weight: 700; }
    .top-banner p  { margin: 0; opacity: 0.75; font-size: 0.9rem; }
</style>
<div class="top-banner">
    <h1>📊 Real-Time Social Media Analytics System</h1>
    <p>CSI-630 Final Year Project &nbsp;·&nbsp; Govt. Municipal Graduate College, Faisalabad &nbsp;·&nbsp; Python · VADER · TextBlob · ML Models · Streamlit</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Main Dashboard",
    "🔴 Reddit Live Fetcher",
    "💬 Comments Analyser",
    "🔬 Model Comparison",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:
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
    from ml_models      import load_metrics, models_exist, predict_sentiment
    from wordcloud_gen  import generate_wordcloud_figure, generate_single_wordcloud
    from hashtag_trends import get_top_hashtags, get_hashtag_sentiment_breakdown

    STOP_WORDS = set(stopwords.words("english"))
    _vader     = SentimentIntensityAnalyzer()

    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)

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

    initialise_database()
    db_df  = load_all_results()
    counts = get_sentiment_counts()
    total  = sum(counts.values())

    st.markdown('<p class="section-title">📈 Sentiment Overview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    pct = lambda n: f"{round(n/total*100,1)}%" if total > 0 else "0%"
    c1.metric("📦 Total Posts",  f"{total:,}")
    c2.metric("😊 Positive",     f"{counts['Positive']:,}", pct(counts["Positive"]))
    c3.metric("😞 Negative",     f"{counts['Negative']:,}", pct(counts["Negative"]))
    c4.metric("😐 Neutral",      f"{counts['Neutral']:,}",  pct(counts["Neutral"]))

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

    if total > 0:
        st.markdown('<p class="section-title">🗂️ Recent Posts</p>', unsafe_allow_html=True)
        n_show = st.slider("Rows to display:", 5, 100, 20, step=5, key="dash_rows")
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

    st.markdown("---")
    st.markdown('<p class="section-title">🧠 Live Single Post Analyser</p>', unsafe_allow_html=True)

    _EMOJI_RE_D = re.compile(
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
        text = _EMOJI_RE_D.sub("", text)
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
        if subj < 0.1:
            label = "Neutral"
        else:
            label = "Positive" if avg >= 0.1 else ("Negative" if avg <= -0.1 else "Neutral")
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
            label="", placeholder="e.g. Real Madrid lost in a shocking upset tonight...",
            height=160, label_visibility="collapsed", key="post_input",
        )
        st.markdown("**💡 Try an example:**")
        ex1, ex2, ex3 = st.columns(3)
        if ex1.button("😊 Positive", key="d_ex1"):
            st.session_state["_ex"] = "I absolutely love this! Best experience ever, highly recommend it!"
        if ex2.button("😞 Negative", key="d_ex2"):
            st.session_state["_ex"] = "Terrible service, completely disappointed. Never coming back again!"
        if ex3.button("😐 Neutral", key="d_ex3"):
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
            m3.metric("Subjectivity", result["subjectivity"])

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
        if st.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.post_history = []
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-title">🤖 ML Model Performance</p>', unsafe_allow_html=True)
    st.caption("Models trained on Sentiment140 — 1.6 million labelled tweets (Stanford NLP).")

    if not models_exist():
        st.warning("⚠️ ML models not trained yet.")
        st.code("python train_models.py", language="bash")
    else:
        metrics = load_metrics()
        if metrics:
            model_names = list(metrics.keys())
            metric_cols = st.columns(len(model_names) * 2)
            for i, name in enumerate(model_names):
                m = metrics[name]
                metric_cols[i*2].metric(
                    f"🎯 {name} Accuracy", f"{m['accuracy']*100:.2f}%",
                    delta=f"+{(m['accuracy']-0.5)*100:.1f}% vs random"
                )
                metric_cols[i*2+1].metric(f"📐 {name} F1", f"{m['f1_score']*100:.2f}%")

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
                        for c_i in range(2):
                            ax.text(c_i, r, f"{cm[r,c_i]:,}", ha="center", va="center",
                                    fontsize=13, fontweight="bold",
                                    color="white" if cm[r,c_i] > cm.max()/2 else "black")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<p class="section-title">☁️ Word Cloud</p>', unsafe_allow_html=True)
    if total == 0:
        st.info("Run the pipeline first to generate word clouds.")
    else:
        _tcol = next((c for c in ["raw_text","clean_text"] if c in db_df.columns), None)
        if _tcol and "sentiment" in db_df.columns:
            pos_texts = db_df.loc[db_df["sentiment"]=="Positive", _tcol].dropna().tolist()
            neg_texts = db_df.loc[db_df["sentiment"]=="Negative", _tcol].dropna().tolist()
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

    st.markdown("---")
    st.markdown('<p class="section-title">📊 Trending Hashtags</p>', unsafe_allow_html=True)
    if total > 0:
        _tcol = next((c for c in ["raw_text","clean_text"] if c in db_df.columns), None)
        if _tcol:
            all_texts = db_df[_tcol].dropna().tolist()
            hc1, hc2 = st.columns(2)
            top_n   = hc1.slider("Top N hashtags:", 5, 30, 15, key="ht_topn")
            min_cnt = hc2.slider("Min occurrences:", 1, 10, 2, key="ht_min")
            ht_df = get_top_hashtags(all_texts, top_n=top_n, min_count=min_cnt)
            if ht_df.empty:
                st.info("No hashtags found in dataset.")
            else:
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
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 2 — REDDIT LIVE FETCHER
# ══════════════════════════════════════════════════════════════════
with tab2:
    import pandas as pd
    import urllib.request
    import xml.etree.ElementTree as ET
    import re, string, sqlite3, os, time
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import nltk
    from nltk.corpus   import stopwords
    from nltk.tokenize import word_tokenize
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from wordcloud import WordCloud

    try:
        from ml_models import predict_sentiment, models_exist as r_models_exist
        R_ML_AVAILABLE = r_models_exist()
    except ImportError:
        R_ML_AVAILABLE = False

    from hashtag_trends import get_top_hashtags as r_get_top_hashtags

    R_STOP_WORDS = set(stopwords.words("english"))
    r_vader      = SentimentIntensityAnalyzer()
    R_DB_PATH    = "data/reddit_keywords.db"
    R_TABLE      = "reddit_posts"

    QUICK_TOPICS = [
        "cricket", "pakistan", "chatgpt", "football", "iphone",
        "climate", "bitcoin", "india", "elon musk", "artificial intelligence"
    ]

    st.markdown("""
    <style>
        .main-banner-r {
            background: linear-gradient(135deg, #ff4500 0%, #ff6534 100%);
            color: white; padding: 24px 30px; border-radius: 16px;
            margin-bottom: 20px; box-shadow: 0 8px 32px rgba(255,69,0,0.3);
        }
        .main-banner-r h2 { margin: 0 0 6px 0; font-size: 1.6rem; font-weight: 700; }
        .main-banner-r p  { margin: 0; opacity: 0.85; }
        .stat-card-r {
            background: white; border-radius: 14px; padding: 20px;
            text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.07);
            border-bottom: 4px solid #ff4500;
        }
        .stat-card-r h3 { font-size: 2rem; margin: 8px 0 4px 0; }
        .stat-card-r p  { color: #888; margin: 0; font-size: 0.85rem; }
    </style>
    <div class="main-banner-r">
        <h2>🔴 Reddit Live Sentiment Analyser</h2>
        <p>Live Reddit Posts · VADER · TextBlob · ML Models · Word Cloud · No API Key Needed</p>
    </div>
    """, unsafe_allow_html=True)

    _R_EMOJI_RE = re.compile(
        "[""\U0001F600-\U0001F64F""\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF""\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0""\U000024C2-\U0001F251""]+",
        flags=re.UNICODE,
    )

    def r_clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = _R_EMOJI_RE.sub("", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in R_STOP_WORDS and len(w) > 1]
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()

    def r_analyse_post(text, use_ml=False, ml_model="lr"):
        cleaned  = r_clean_text(text)
        compound = r_vader.polarity_scores(cleaned)["compound"]
        tb_sent  = TextBlob(cleaned).sentiment
        polarity = tb_sent.polarity
        subj     = tb_sent.subjectivity
        avg      = (compound + polarity) / 2.0
        if subj < 0.1:
            label = "Neutral"
        else:
            label = "Positive" if avg >= 0.1 else ("Negative" if avg <= -0.1 else "Neutral")
        result = {"clean_text": cleaned, "vader_compound": round(compound, 4),
                  "tb_polarity": round(polarity, 4), "sentiment": label, "ml_sentiment": "N/A"}
        if use_ml and R_ML_AVAILABLE:
            result["ml_sentiment"] = predict_sentiment([text], model=ml_model)[0]
        return result

    def r_fetch_posts(topic, limit=50):
        query = topic.strip().replace(" ", "+")
        url   = f"https://www.reddit.com/search.rss?q={query}&sort=relevance&limit={limit}&type=link"
        headers = {"User-Agent": "Mozilla/5.0 (fyp-sentiment/1.0)"}
        try:
            req      = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req, timeout=10)
            xml_data = response.read()
        except Exception as e:
            st.error(f"Connection error: {e}"); return []
        posts = []
        try:
            root = ET.fromstring(xml_data)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                link_el  = entry.find("atom:link",  ns)
                pub_el   = entry.find("atom:published", ns)
                title = title_el.text if title_el is not None else ""
                link  = link_el.get("href", "") if link_el is not None else ""
                pub   = pub_el.text  if pub_el  is not None else ""
                if title:
                    posts.append({"text": title, "url": link, "published": pub})
        except ET.ParseError:
            return []
        return posts

    def r_generate_wc(text_series, title, colormap="viridis"):
        all_text = " ".join(text_series.dropna().tolist())
        if not all_text.strip(): return None
        wc = WordCloud(width=900, height=400, background_color="#0e1117",
                       colormap=colormap, max_words=100,
                       collocations=False, stopwords=R_STOP_WORDS).generate(all_text)
        fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#0e1117")
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="white")
        plt.tight_layout(pad=0.5)
        return fig

    def r_init_db():
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(R_DB_PATH)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {R_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT, raw_text TEXT, clean_text TEXT,
                vader_compound REAL, tb_polarity REAL,
                sentiment TEXT, ml_sentiment TEXT,
                url TEXT, saved_at TEXT
            )
        """)
        conn.commit(); conn.close()

    def r_save_db(df, topic):
        conn = sqlite3.connect(R_DB_PATH)
        now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for _, row in df.iterrows():
            existing = conn.execute(
                f"SELECT id FROM {R_TABLE} WHERE raw_text=? AND topic=?",
                (row.get("text",""), topic)
            ).fetchone()
            if not existing:
                conn.execute(f"""
                    INSERT INTO {R_TABLE}
                        (topic, raw_text, clean_text, vader_compound, tb_polarity,
                         sentiment, ml_sentiment, url, saved_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (topic, row.get("text",""), row.get("clean_text",""),
                      row.get("vader_compound",0), row.get("tb_polarity",0),
                      row.get("sentiment","Neutral"), row.get("ml_sentiment","N/A"),
                      row.get("url",""), now))
        conn.commit(); conn.close()

    def r_load_db(topic_filter=None):
        conn = sqlite3.connect(R_DB_PATH)
        if topic_filter and topic_filter != "All":
            df = pd.read_sql_query(f"SELECT * FROM {R_TABLE} WHERE topic=? ORDER BY id DESC",
                                   conn, params=(topic_filter,))
        else:
            df = pd.read_sql_query(f"SELECT * FROM {R_TABLE} ORDER BY id DESC", conn)
        conn.close()
        return df

    def r_do_fetch(topic, limit, use_ml=False, ml_key="lr"):
        raw = r_fetch_posts(topic, limit)
        if not raw: return 0
        df  = pd.DataFrame(raw)
        results = pd.DataFrame(list(
            df["text"].apply(lambda t: r_analyse_post(t, use_ml=use_ml, ml_model=ml_key))
        ))
        final = pd.concat([df, results], axis=1)
        r_save_db(final, topic)
        return len(raw)

    r_init_db()

    # Sidebar controls for Reddit tab
    r_topic = st.text_input("🔍 Search keyword:", placeholder="cricket, pakistan, ai...",
                             value="cricket", key="r_topic_input")

    st.markdown("**⚡ Quick Topics:**")
    qt_cols = st.columns(5)
    for i, qt in enumerate(QUICK_TOPICS):
        if qt_cols[i % 5].button(qt, key=f"rqt_{qt}", use_container_width=True):
            r_topic = qt

    col_l, col_r = st.columns(2)
    r_limit    = col_l.slider("Posts to fetch:", 5, 50, 25, step=5, key="r_limit")
    r_fetch_btn = col_r.button("🚀 Fetch & Analyse", use_container_width=True, key="r_fetch")

    if R_ML_AVAILABLE:
        r_use_ml   = st.toggle("Use ML Sentiment", value=True, key="r_use_ml")
        r_ml_choice = st.radio("ML Model:", ["Logistic Regression", "Random Forest"],
                               horizontal=True, key="r_ml_choice")
        r_ml_key   = "lr" if r_ml_choice == "Logistic Regression" else "rf"
    else:
        r_use_ml = False; r_ml_key = "lr"
        st.warning("ML models not trained. Run: `python train_models.py`")

    if r_fetch_btn:
        if not r_topic.strip():
            st.warning("Enter a keyword.")
        else:
            with st.spinner(f"Fetching posts about '{r_topic}' from Reddit …"):
                n = r_do_fetch(r_topic.strip(), r_limit, use_ml=r_use_ml, ml_key=r_ml_key)
            if n:
                st.success(f"✅ **{n}** posts fetched for: **'{r_topic}'**")
            else:
                st.error("No posts found. Try another keyword.")

    # Topic filter
    all_topics_r = ["All"]
    try:
        conn = sqlite3.connect(R_DB_PATH)
        topics_df = pd.read_sql_query(f"SELECT DISTINCT topic FROM {R_TABLE}", conn)
        conn.close()
        all_topics_r = ["All"] + topics_df["topic"].tolist()
    except: pass

    selected_r_topic = st.selectbox("📂 Filter by topic:", all_topics_r, key="r_topic_filter")
    r_db_df = r_load_db(topic_filter=selected_r_topic)
    r_total = len(r_db_df)

    if r_total > 0:
        R_COLOURS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}
        r_counts  = r_db_df["sentiment"].value_counts().to_dict()
        r_pos = r_counts.get("Positive", 0)
        r_neg = r_counts.get("Negative", 0)
        r_neu = r_counts.get("Neutral",  0)

        st.markdown("## 📊 Live Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, color in [
            (c1, r_total, "📦 Total Posts", "#ff4500"),
            (c2, r_pos,   "😊 Positive",   "#4CAF50"),
            (c3, r_neg,   "😞 Negative",   "#F44336"),
            (c4, r_neu,   "😐 Neutral",    "#2196F3"),
        ]:
            col.markdown(f"""
            <div class="stat-card-r" style="border-bottom-color:{color}">
                <p>{label}</p>
                <h3 style="color:{color}">{val:,}</h3>
                <p>{round(val/r_total*100,1)}% of total</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("#### Sentiment Distribution")
            labels = [k for k, v in r_counts.items() if v > 0]
            sizes  = [r_counts[k] for k in labels]
            clrs   = [R_COLOURS[k] for k in labels]
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%", colors=clrs,
                startangle=140, wedgeprops={"edgecolor":"white","linewidth":2.5},
                textprops={"fontsize":12})
            for at in autotexts:
                at.set_color("white"); at.set_fontweight("bold")
            ax.set_title(f"Sentiment Split — {r_total} Posts", fontsize=12, fontweight="bold")
            st.pyplot(fig); plt.close()

        with chart2:
            st.markdown("#### Posts per Category")
            bar_labels = list(r_counts.keys())
            bar_values = list(r_counts.values())
            bar_clrs   = [R_COLOURS.get(k, "#999") for k in bar_labels]
            fig, ax = plt.subplots(figsize=(5, 5))
            bars = ax.bar(bar_labels, bar_values, color=bar_clrs,
                          width=0.5, edgecolor="white", linewidth=2)
            for bar, val in zip(bars, bar_values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3, str(val),
                        ha="center", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Posts", fontsize=11)
            ax.set_title("Category Breakdown", fontsize=12, fontweight="bold")
            ax.set_ylim(0, max(bar_values)*1.25)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("## ☁️ Word Clouds")
        t1, t2, t3, t4 = st.tabs(["🌐 All Posts", "😊 Positive", "😞 Negative", "😐 Neutral"])
        with t1:
            fig = r_generate_wc(r_db_df["clean_text"], f"All Posts — '{selected_r_topic}'", "plasma")
            if fig: st.pyplot(fig, use_container_width=True); plt.close()
        with t2:
            d = r_db_df[r_db_df["sentiment"]=="Positive"]
            if len(d) > 0:
                fig = r_generate_wc(d["clean_text"], "Positive Posts", "YlGn")
                if fig: st.pyplot(fig, use_container_width=True); plt.close()
            else: st.info("No positive posts.")
        with t3:
            d = r_db_df[r_db_df["sentiment"]=="Negative"]
            if len(d) > 0:
                fig = r_generate_wc(d["clean_text"], "Negative Posts", "OrRd")
                if fig: st.pyplot(fig, use_container_width=True); plt.close()
            else: st.info("No negative posts.")
        with t4:
            d = r_db_df[r_db_df["sentiment"]=="Neutral"]
            if len(d) > 0:
                fig = r_generate_wc(d["clean_text"], "Neutral Posts", "Blues")
                if fig: st.pyplot(fig, use_container_width=True); plt.close()
            else: st.info("No neutral posts.")

        st.markdown("---")
        st.markdown("## 📊 Trending Hashtags")
        raw_texts_r = r_db_df["raw_text"].dropna().tolist()
        ht_df_r = r_get_top_hashtags(raw_texts_r, top_n=15, min_count=1)
        if ht_df_r.empty:
            st.info("No hashtags found in these posts.")
        else:
            fig, ax = plt.subplots(figsize=(10, max(4, len(ht_df_r)*0.5)))
            bar_clrs = plt.cm.viridis(np.linspace(0.2, 0.85, len(ht_df_r)))[::-1]
            ax.barh(ht_df_r["hashtag"][::-1], ht_df_r["count"][::-1],
                    color=bar_clrs, edgecolor="white", linewidth=0.8)
            ax.set_xlabel("Occurrences"); ax.set_title("Top Hashtags", fontsize=13, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("---")
        st.markdown("## 🗂️ Fetched Posts")
        n_show_r = st.slider("Rows to display:", 5, min(200, r_total), 20, step=5, key="r_rows")
        show_cols = ["id","topic","raw_text","sentiment","ml_sentiment","vader_compound","tb_polarity","saved_at"]
        avail_r = [c for c in show_cols if c in r_db_df.columns]
        def _r_hl(val):
            return {"Positive":"background-color:#c8e6c9;color:#1b5e20;font-weight:600",
                    "Negative":"background-color:#ffcdd2;color:#b71c1c;font-weight:600",
                    "Neutral": "background-color:#bbdefb;color:#0d47a1;font-weight:600"}.get(val,"")
        disp_r = r_db_df.head(n_show_r)[avail_r]
        sent_cols_r = [c for c in ["sentiment","ml_sentiment"] if c in disp_r.columns]
        st.dataframe(disp_r.style.applymap(_r_hl, subset=sent_cols_r),
                     use_container_width=True, height=400)
        st.download_button("⬇️ Export CSV",
                           data=r_db_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"reddit_{selected_r_topic}_sentiment.csv",
                           mime="text/csv", key="r_export")

        if st.button("🗑️ Clear Reddit Data", key="r_clear"):
            conn = sqlite3.connect(R_DB_PATH)
            conn.execute(f"DELETE FROM {R_TABLE}")
            conn.commit(); conn.close()
            st.warning("Reddit database cleared!"); st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px;color:#aaa;">
            <div style="font-size:5rem">🔴</div>
            <h3 style="color:#ccc">No posts yet</h3>
            <p>Enter a keyword above and click <b>Fetch & Analyse</b></p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — COMMENTS ANALYSER
# ══════════════════════════════════════════════════════════════════
with tab3:
    import pandas as pd
    import urllib.request
    import urllib.parse
    import json
    import re, string, os
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import nltk
    from nltk.corpus   import stopwords
    from nltk.tokenize import word_tokenize
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from wordcloud import WordCloud

    try:
        from ml_models import predict_sentiment as ca_predict, models_exist as ca_models_exist
        CA_ML_AVAILABLE = ca_models_exist()
    except ImportError:
        CA_ML_AVAILABLE = False

    CA_STOP_WORDS = set(stopwords.words("english"))
    ca_vader      = SentimentIntensityAnalyzer()
    YOUTUBE_API_KEY = "AIzaSyA91ZTGmAF-gVYWRF3JY9Zcov2pX-0p_Sw"

    st.markdown("""
    <style>
        .main-banner-ca {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white; padding: 24px 30px; border-radius: 16px;
            margin-bottom: 20px; box-shadow: 0 8px 32px rgba(17,153,142,0.3);
        }
        .main-banner-ca h2 { margin: 0 0 6px 0; font-size: 1.6rem; font-weight: 700; }
        .main-banner-ca p  { margin: 0; opacity: 0.9; }
        .comment-card {
            background: white; border-radius: 12px; padding: 16px 20px;
            margin-bottom: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border-left: 4px solid #ccc;
        }
        .comment-positive { border-left-color: #4CAF50 !important; }
        .comment-negative { border-left-color: #F44336 !important; }
        .comment-neutral  { border-left-color: #2196F3 !important; }
    </style>
    <div class="main-banner-ca">
        <h2>💬 Comments Sentiment Analyser</h2>
        <p>Reddit Comments · YouTube Comments · VADER + TextBlob + ML Models · Word Cloud</p>
    </div>
    """, unsafe_allow_html=True)

    _CA_EMOJI_RE = re.compile(
        "[""\U0001F600-\U0001F64F""\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF""\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0""\U000024C2-\U0001F251""]+",
        flags=re.UNICODE,
    )

    def ca_clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = _CA_EMOJI_RE.sub("", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in CA_STOP_WORDS and len(w) > 1]
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()

    def ca_analyse_comment(text, use_ml=False, ml_model="lr"):
        cleaned  = ca_clean_text(text)
        compound = ca_vader.polarity_scores(cleaned)["compound"]
        polarity = TextBlob(cleaned).sentiment.polarity
        avg      = (compound + polarity) / 2.0
        tb_sent  = TextBlob(cleaned).sentiment
        polarity = tb_sent.polarity
        subj     = tb_sent.subjectivity
        if subj < 0.1:
            label = "Neutral"
        else:
            label = "Positive" if avg >= 0.1 else ("Negative" if avg <= -0.1 else "Neutral")
        result = {"clean_text": cleaned, "vader_compound": round(compound, 4),
                  "tb_polarity": round(polarity, 4), "sentiment": label, "ml_sentiment": "N/A"}
        if use_ml and CA_ML_AVAILABLE:
            result["ml_sentiment"] = ca_predict([text], model=ml_model)[0]
        return result

    def ca_sentiment_color(label):
        return {"Positive":"#4CAF50","Negative":"#F44336","Neutral":"#2196F3"}.get(label,"#999")

    def ca_sentiment_emoji(label):
        return {"Positive":"😊","Negative":"😞","Neutral":"😐"}.get(label,"")

    def ca_generate_wordcloud(text_series, title, colormap="viridis"):
        all_text = " ".join(text_series.dropna().tolist())
        if not all_text.strip():
            st.info("Not enough text."); return
        wc = WordCloud(width=800, height=380, background_color="white",
                       colormap=colormap, max_words=80,
                       collocations=False, stopwords=CA_STOP_WORDS).generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        st.pyplot(fig); plt.close()

    def ca_extract_reddit_post_id(url):
        match = re.search(r"/comments/([a-zA-Z0-9]+)", url)
        return match.group(1) if match else ""

    def ca_fetch_reddit_comments(url, limit=50):
        post_id = ca_extract_reddit_post_id(url)
        if not post_id:
            st.error("Invalid Reddit URL."); return []
        api_url = f"https://www.reddit.com/comments/{post_id}.json?limit={limit}"
        headers = {"User-Agent": "Mozilla/5.0 (fyp-sentiment/1.0)"}
        try:
            req      = urllib.request.Request(api_url, headers=headers)
            response = urllib.request.urlopen(req, timeout=10)
            data     = json.loads(response.read())
        except Exception as e:
            st.error(f"Error: {e}"); return []
        comments = []
        try:
            for item in data[1]["data"]["children"]:
                d = item.get("data", {})
                body = d.get("body", "")
                if body and body not in ("[deleted]", "[removed]"):
                    comments.append({"text": body, "author": d.get("author","unknown"),
                                     "score": d.get("score", 0), "platform": "Reddit"})
        except: pass
        return comments

    def ca_fetch_youtube_comments(video_url, limit=50):
        video_id = ""
        patterns = [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"embed/([^?]+)"]
        for p in patterns:
            m = re.search(p, video_url)
            if m: video_id = m.group(1); break
        if not video_id:
            st.error("Invalid YouTube URL."); return []
        api_url = (f"https://www.googleapis.com/youtube/v3/commentThreads"
                   f"?part=snippet&videoId={video_id}&maxResults={min(limit,100)}"
                   f"&key={YOUTUBE_API_KEY}")
        try:
            req      = urllib.request.Request(api_url)
            response = urllib.request.urlopen(req, timeout=10)
            data     = json.loads(response.read())
        except Exception as e:
            st.error(f"YouTube API error: {e}"); return []
        comments = []
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({"text": snippet.get("textDisplay",""),
                              "author": snippet.get("authorDisplayName","unknown"),
                              "score": snippet.get("likeCount", 0),
                              "platform": "YouTube"})
        return comments

    # Platform selector
    platform = st.radio("Select Platform:", ["Reddit", "YouTube"], horizontal=True, key="ca_platform")

    if platform == "Reddit":
        reddit_url = st.text_input("Reddit Post URL:",
                                    placeholder="https://www.reddit.com/r/pakistan/comments/...",
                                    key="ca_reddit_url")
        ca_limit = st.slider("Max comments:", 10, 200, 50, step=10, key="ca_limit_r")
        if st.button("🔍 Fetch & Analyse Reddit Comments", key="ca_reddit_btn"):
            if not reddit_url.strip():
                st.warning("Enter a Reddit post URL.")
            else:
                with st.spinner("Fetching Reddit comments …"):
                    raw_comments = ca_fetch_reddit_comments(reddit_url, limit=ca_limit)
                if raw_comments:
                    st.session_state["comments_df"] = pd.DataFrame(raw_comments)
                    st.session_state["platform"]    = "Reddit"
                    st.session_state["source_url"]  = reddit_url
                    st.success(f"✅ Fetched **{len(raw_comments)}** Reddit comments!")
                else:
                    st.error("No comments found.")

    else:
        yt_url = st.text_input("YouTube Video URL:",
                                placeholder="https://www.youtube.com/watch?v=...",
                                key="ca_yt_url")
        ca_limit_yt = st.slider("Max comments:", 10, 100, 50, step=10, key="ca_limit_yt")
        if st.button("🔍 Fetch & Analyse YouTube Comments", key="ca_yt_btn"):
            if not yt_url.strip():
                st.warning("Enter a YouTube URL.")
            else:
                with st.spinner("Fetching YouTube comments …"):
                    raw_comments = ca_fetch_youtube_comments(yt_url, limit=ca_limit_yt)
                if raw_comments:
                    st.session_state["comments_df"] = pd.DataFrame(raw_comments)
                    st.session_state["platform"]    = "YouTube"
                    st.session_state["source_url"]  = yt_url
                    st.success(f"✅ Fetched **{len(raw_comments)}** YouTube comments!")
                else:
                    st.error("No comments found.")

    if "comments_df" in st.session_state:
        df            = st.session_state["comments_df"].copy()
        platform_name = st.session_state.get("platform", "")
        source        = st.session_state.get("source_url", "")

        st.markdown("---")
        st.markdown(f"## 📊 Analysis Results — {platform_name}")
        st.caption(f"Source: {source}")

        use_ml_ca = False; ml_key_ca = "lr"
        if CA_ML_AVAILABLE:
            ml_c1, ml_c2 = st.columns(2)
            use_ml_ca   = ml_c1.toggle("Use ML Sentiment", value=True, key="ca_ml_toggle")
            ml_choice_c = ml_c2.radio("ML Model:", ["Logistic Regression", "Random Forest"],
                                       horizontal=True, key="ca_ml_model")
            ml_key_ca   = "lr" if ml_choice_c == "Logistic Regression" else "rf"

        with st.spinner("Analysing sentiment …"):
            results = pd.DataFrame(list(
                df["text"].apply(lambda t: ca_analyse_comment(t, use_ml=use_ml_ca, ml_model=ml_key_ca))
            ))
            df = pd.concat([df, results], axis=1)

        counts_ca = df["sentiment"].value_counts().to_dict()
        total_ca  = len(df)
        pos_ca    = counts_ca.get("Positive", 0)
        neg_ca    = counts_ca.get("Negative", 0)
        neu_ca    = counts_ca.get("Neutral",  0)
        CA_COLOURS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💬 Total Comments", total_ca)
        k2.metric("😊 Positive", f"{pos_ca}  ({round(pos_ca/total_ca*100,1)}%)")
        k3.metric("😞 Negative", f"{neg_ca}  ({round(neg_ca/total_ca*100,1)}%)")
        k4.metric("😐 Neutral",  f"{neu_ca}  ({round(neu_ca/total_ca*100,1)}%)")

        col1_ca, col2_ca = st.columns(2)
        with col1_ca:
            st.markdown("#### Sentiment Distribution")
            labels = [k for k, v in counts_ca.items() if v > 0]
            sizes  = [counts_ca[k] for k in labels]
            clrs   = [CA_COLOURS[k] for k in labels]
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%", colors=clrs,
                startangle=140, wedgeprops={"edgecolor":"white","linewidth":2.5},
                textprops={"fontsize":12})
            for at in autotexts:
                at.set_color("white"); at.set_fontweight("bold")
            ax.set_title(f"Comment Sentiment — {platform_name}", fontsize=12, fontweight="bold")
            st.pyplot(fig); plt.close()

        with col2_ca:
            st.markdown("#### Comment Count per Category")
            fig, ax = plt.subplots(figsize=(5, 5))
            bar_labels = list(counts_ca.keys())
            bar_values = list(counts_ca.values())
            bar_clrs   = [CA_COLOURS.get(k,"#999") for k in bar_labels]
            bars = ax.bar(bar_labels, bar_values, color=bar_clrs,
                          width=0.5, edgecolor="white", linewidth=2)
            for bar, val in zip(bars, bar_values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3, str(val),
                        ha="center", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Comments", fontsize=11)
            ax.set_title("Comments per Category", fontsize=12, fontweight="bold")
            ax.set_ylim(0, max(bar_values)*1.25)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig); plt.close()

        st.markdown("## ☁️ Word Clouds")
        wt1, wt2, wt3, wt4 = st.tabs(["🌐 All","😊 Positive","😞 Negative","😐 Neutral"])
        with wt1: ca_generate_wordcloud(df["clean_text"], "All Comments", "viridis")
        with wt2:
            d = df[df["sentiment"]=="Positive"]
            ca_generate_wordcloud(d["clean_text"],"Positive Comments","Greens") if len(d)>0 else st.info("No positive comments.")
        with wt3:
            d = df[df["sentiment"]=="Negative"]
            ca_generate_wordcloud(d["clean_text"],"Negative Comments","Reds") if len(d)>0 else st.info("No negative comments.")
        with wt4:
            d = df[df["sentiment"]=="Neutral"]
            ca_generate_wordcloud(d["clean_text"],"Neutral Comments","Blues") if len(d)>0 else st.info("No neutral comments.")

        st.markdown("## 💬 Individual Comments")
        filter_opt = st.selectbox("Filter by sentiment:", ["All","Positive","Negative","Neutral"], key="ca_filter")
        filtered = df if filter_opt == "All" else df[df["sentiment"]==filter_opt]
        for _, row in filtered.head(30).iterrows():
            label = row["sentiment"]
            color = CA_COLOURS[label]
            emoji = ca_sentiment_emoji(label)
            css_class = f"comment-{label.lower()}"
            st.markdown(f"""
            <div class="comment-card {css_class}">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="font-weight:600;color:#333">@{row.get('author','unknown')}</span>
                    <span style="background:{color};color:white;padding:2px 10px;
                          border-radius:12px;font-size:0.8rem;font-weight:600">
                        {emoji} {label}
                    </span>
                </div>
                <p style="margin:0;color:#444;line-height:1.5">{str(row['text'])[:300]}</p>
                <div style="margin-top:8px;font-size:0.8rem;color:#999">
                    VADER: {row['vader_compound']} &nbsp;|&nbsp;
                    TextBlob: {row['tb_polarity']} &nbsp;|&nbsp;
                    ML: {row.get('ml_sentiment','N/A')} &nbsp;|&nbsp;
                    👍 {row.get('score',0)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        export_cols = ["text","author","sentiment","vader_compound","tb_polarity","score","platform"]
        export_df   = df[[c for c in export_cols if c in df.columns]]
        st.download_button(
            "⬇️ Export Comments as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{platform_name.lower()}_comments_sentiment.csv",
            mime="text/csv", key="ca_export"
        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px;color:#aaa;">
            <div style="font-size:5rem">💬</div>
            <h3 style="color:#ccc">No comments analysed yet</h3>
            <p>Select a platform above, paste a URL, and click Fetch & Analyse</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab4:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import joblib, os
    import matplotlib.pyplot as plt

    mc_vader = SentimentIntensityAnalyzer()

    @st.cache_resource
    def mc_load_model():
        model_path = os.path.join("models", "lr_pipeline.pkl")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    mc_model = mc_load_model()

    def mc_analyze_vader(text):
        scores = mc_vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05: label = "Positive"
        elif compound <= -0.05: label = "Negative"
        else: label = "Neutral"
        return label, round(abs(compound), 2)

    def mc_analyze_ml(text):
        if mc_model is None: return "Model not found", None
        prediction = mc_model.predict([text])[0]
        proba      = mc_model.predict_proba([text])[0]
        label      = "Positive" if prediction == 1 else "Negative"
        return label, round(max(proba), 2)

    def mc_analyze_textblob(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.05: label = "Positive"
        elif polarity < -0.05: label = "Negative"
        else: label = "Neutral"
        return label, round(abs(polarity), 2)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;
                padding:24px 30px;border-radius:16px;margin-bottom:20px;">
        <h2 style="margin:0 0 6px 0;font-size:1.6rem;font-weight:700;">🔬 Sentiment Model Comparison</h2>
        <p style="margin:0;opacity:0.85;">VADER vs Logistic Regression vs TextBlob — Side by Side Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    mc_text = st.text_area("Enter text to analyze:",
                            placeholder="Type something here...",
                            height=120, key="mc_text")

    if st.button("Compare All Models", type="primary", key="mc_compare"):
        if not mc_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running all three models..."):
                vader_label, vader_conf = mc_analyze_vader(mc_text)
                ml_label, ml_conf       = mc_analyze_ml(mc_text)
                tb_label, tb_conf       = mc_analyze_textblob(mc_text)

            st.markdown("### Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VADER", vader_label)
                st.caption(f"Confidence: {vader_conf}")
            with col2:
                st.metric("Logistic Regression", ml_label)
                if ml_conf: st.caption(f"Confidence: {ml_conf}")
            with col3:
                st.metric("TextBlob", tb_label)
                st.caption(f"Polarity: {tb_conf}")

            labels_mc = [vader_label, ml_label, tb_label]
            labels_clean_mc = [l for l in labels_mc if l != "Model not found"]

            st.markdown("---")
            if len(set(labels_clean_mc)) == 1:
                st.success("✅ All models agree — strong signal")
            elif vader_label == ml_label and tb_label != vader_label:
                st.warning(f"⚠️ VADER & ML say **{vader_label}** but TextBlob says **{tb_label}**")
            elif vader_label == tb_label and ml_label != vader_label:
                st.warning(f"⚠️ ML Model disagrees — predicted **{ml_label}** while others say **{vader_label}**")
            elif ml_label == tb_label and vader_label != ml_label:
                st.warning(f"⚠️ VADER disagrees — predicted **{vader_label}** while others say **{ml_label}**")
            else:
                st.error("❌ All three models disagree — ambiguous text")

            with st.expander("View detailed VADER scores"):
                scores = mc_vader.polarity_scores(mc_text)
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Positive", round(scores["pos"], 3))
                col_b.metric("Negative", round(scores["neg"], 3))
                col_c.metric("Neutral", round(scores["neu"], 3))
                col_d.metric("Compound", round(scores["compound"], 3))

    st.markdown("---")
    st.markdown("""
    **Why compare these three models?**

    - **VADER** — Rule-based lexicon, great for social media, understands negation and intensifiers
    - **Logistic Regression** — ML model trained on 1.6M tweets (Sentiment140), learns patterns statistically  
    - **TextBlob** — Pattern-based NLP library, simpler than VADER, good general-purpose baseline

    When all three agree → high confidence result. When they disagree → text is ambiguous or context-dependent.
    This comparison is a key research contribution of this FYP.
    """)

    st.markdown("---")
    st.caption("FYP — Real-Time Social Media Analytics System | CSI-630 | Govt. Municipal Graduate College, Faisalabad")