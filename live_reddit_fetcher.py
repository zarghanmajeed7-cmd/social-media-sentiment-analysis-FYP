# ============================================================
# live_reddit_fetcher.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Purpose : Fetch live Reddit posts by keyword, analyse with
#           VADER + TextBlob + ML Models (RF/LR), show Word Cloud,
#           Hashtag Trends, Topic Comparison — NO API KEY needed.
#
# RUN:
#   python -m streamlit run live_reddit_fetcher.py
# ============================================================

import streamlit as st
import pandas as pd
import urllib.request
import xml.etree.ElementTree as ET
import re, string, sqlite3, os, time
from datetime import datetime

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── ML Models (optional — works without them too) ──────────────────
try:
    from ml_models import predict_sentiment, models_exist
    ML_AVAILABLE = models_exist()
except ImportError:
    ML_AVAILABLE = False

from hashtag_trends import get_top_hashtags, get_hashtag_sentiment_breakdown

STOP_WORDS = set(stopwords.words("english"))
_vader     = SentimentIntensityAnalyzer()
DB_PATH    = "data/reddit_keywords.db"
TABLE      = "reddit_posts"

# ── Predefined popular topics (one-click fetch) ─────────────────────
QUICK_TOPICS = [
    "cricket", "pakistan", "chatgpt", "football", "iphone",
    "climate", "bitcoin", "india", "elon musk", "artificial intelligence"
]

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reddit Live Sentiment",
    page_icon="🔴",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main-banner {
        background: linear-gradient(135deg, #ff4500 0%, #ff6534 100%);
        color: white; padding: 30px 35px; border-radius: 16px;
        margin-bottom: 28px; box-shadow: 0 8px 32px rgba(255,69,0,0.3);
    }
    .main-banner h1 { margin: 0 0 8px 0; font-size: 2rem; font-weight: 700; }
    .main-banner p  { margin: 0; opacity: 0.85; font-size: 1rem; }

    .stat-card {
        background: white; border-radius: 14px; padding: 20px;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.07);
        border-bottom: 4px solid #ff4500;
    }
    .stat-card h3 { font-size: 2rem; margin: 8px 0 4px 0; }
    .stat-card p  { color: #888; margin: 0; font-size: 0.85rem; }

    .refresh-banner {
        background: #e8f5e9; border-left: 5px solid #4CAF50;
        padding: 12px 18px; border-radius: 8px;
        margin-bottom: 16px; font-size: 0.95rem;
    }
    .ml-badge {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white; padding: 6px 14px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
        margin: 4px;
    }
    .stButton button {
        background: linear-gradient(135deg, #ff4500, #ff6534) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255,69,0,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-banner">
    <h1>🔴 Reddit Live Sentiment Analyser</h1>
    <p>CSI-630 Final Year Project &nbsp;·&nbsp;
       Govt. Municipal Graduate College, Faisalabad &nbsp;·&nbsp;
       Live Reddit Posts · VADER · TextBlob · ML Models · Word Cloud · Hashtag Trends</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
_EMOJI_RE = re.compile(
    "[""\U0001F600-\U0001F64F""\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF""\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0""\U000024C2-\U0001F251""]+",
    flags=re.UNICODE,
)

def clean_text(text):
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


def analyse_post(text, use_ml=False, ml_model="lr"):
    cleaned  = clean_text(text)
    compound = _vader.polarity_scores(cleaned)["compound"]
    blob     = TextBlob(cleaned)
    polarity = blob.sentiment.polarity
    avg      = (compound + polarity) / 2.0
    subj     = blob.sentiment.subjectivity
    if subj < 0.1:
        label = "Neutral"
    else:
        label = "Positive" if avg >= 0.1 else ("Negative" if avg <= -0.1 else "Neutral")

    result = {
        "clean_text":     cleaned,
        "vader_compound": round(compound, 4),
        "tb_polarity":    round(polarity, 4),
        "sentiment":      label,
        "ml_sentiment":   "N/A",
    }

    if use_ml and ML_AVAILABLE:
        ml_pred = predict_sentiment([text], model=ml_model)
        result["ml_sentiment"] = ml_pred[0]

    return result


def fetch_posts(topic, limit=50):
    """Fetch Reddit posts via free RSS — no API key needed."""
    query = topic.strip().replace(" ", "+")
    url = f"https://www.reddit.com/search.rss?q={query}&sort=relevance&limit={limit}&type=link"
    headers = {"User-Agent": "Mozilla/5.0 (fyp-sentiment/1.0)"}
    try:
        req      = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=10)
        xml_data = response.read()
    except Exception as e:
        st.error(f"Connection error: {e}")
        return []

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


def generate_wordcloud_fig(text_series, title, colormap="viridis", bg="#0e1117"):
    all_text = " ".join(text_series.dropna().tolist())
    if not all_text.strip():
        return None
    wc = WordCloud(
        width=900, height=400, background_color=bg,
        colormap=colormap, max_words=100,
        collocations=False, stopwords=STOP_WORDS
    ).generate(all_text)
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=bg)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="white")
    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT, raw_text TEXT, clean_text TEXT,
            vader_compound REAL, tb_polarity REAL,
            sentiment TEXT, ml_sentiment TEXT,
            url TEXT, saved_at TEXT
        )
    """)
    conn.commit(); conn.close()


def save_to_db(df, topic):
    conn = sqlite3.connect(DB_PATH)
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for _, row in df.iterrows():
        existing = conn.execute(
            f"SELECT id FROM {TABLE} WHERE raw_text=? AND topic=?",
            (row.get("text",""), topic)
        ).fetchone()
        if not existing:
            conn.execute(f"""
                INSERT INTO {TABLE}
                    (topic, raw_text, clean_text, vader_compound, tb_polarity,
                     sentiment, ml_sentiment, url, saved_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                topic,
                row.get("text",""),
                row.get("clean_text",""),
                row.get("vader_compound", 0),
                row.get("tb_polarity", 0),
                row.get("sentiment","Neutral"),
                row.get("ml_sentiment","N/A"),
                row.get("url",""),
                now,
            ))
    conn.commit(); conn.close()


def load_db(topic_filter=None):
    conn = sqlite3.connect(DB_PATH)
    if topic_filter and topic_filter != "All":
        df = pd.read_sql_query(
            f"SELECT * FROM {TABLE} WHERE topic=? ORDER BY id DESC",
            conn, params=(topic_filter,)
        )
    else:
        df = pd.read_sql_query(
            f"SELECT * FROM {TABLE} ORDER BY id DESC", conn
        )
    conn.close()
    return df


def clear_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"DELETE FROM {TABLE}")
    conn.commit(); conn.close()


def do_fetch_and_save(topic, limit, use_ml=False, ml_model="lr"):
    raw = fetch_posts(topic, limit)
    if not raw: return 0
    df  = pd.DataFrame(raw)
    results = pd.DataFrame(list(
        df["text"].apply(lambda t: analyse_post(t, use_ml=use_ml, ml_model=ml_model))
    ))
    final = pd.concat([df, results], axis=1)
    save_to_db(final, topic)
    return len(raw)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Search Topic")
    st.markdown("---")

    topic = st.text_input(
        "search keyword:",
        placeholder="cricket, pakistan, ai...",
        value="cricket"
    )

    st.markdown("**⚡ Quick Topics:**")
    qt_cols = st.columns(2)
    for i, qt in enumerate(QUICK_TOPICS):
        if qt_cols[i % 2].button(qt, key=f"qt_{qt}", use_container_width=True):
            topic = qt

    st.markdown("---")
    limit     = st.slider("Posts to fetch:", 5, 50, 25, step=5)
    fetch_btn = st.button("🚀 Fetch & Analyse", use_container_width=True)

    st.markdown("---")
    st.markdown("## 🤖 ML Model")
    if ML_AVAILABLE:
        st.success("✅ ML Models Ready")
        use_ml    = st.toggle("Use ML Sentiment", value=True)
        ml_model  = st.radio("Model:", ["Logistic Regression", "Random Forest"],
                             horizontal=True, key="ml_choice")
        ml_key    = "lr" if ml_model == "Logistic Regression" else "rf"
    else:
        st.warning("ML models not trained.\nRun: python train_models.py")
        use_ml   = False
        ml_key   = "lr"

    st.markdown("---")
    st.markdown("## 🔄 Auto Refresh")
    auto_refresh = st.toggle("Enable Auto Refresh", value=False)
    if auto_refresh:
        interval = st.selectbox(
            "Interval:",
            [30, 60, 120, 300],
            format_func=lambda x: f"{x}s" if x < 60 else f"{x//60} min"
        )
        st.success(f"Auto refresh every {interval}s")

    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        clear_db(); st.warning("Database cleared!"); st.rerun()
    if st.button("🔄 Refresh Page", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("No API key needed · Free RSS · Real-time Reddit data")


# ══════════════════════════════════════════════════════════════════
# AUTO REFRESH
# ══════════════════════════════════════════════════════════════════
if auto_refresh:
    st.markdown(f"""
    <div class="refresh-banner">
        🟢 Auto Refresh ON — fetching <b>'{topic}'</b> posts every <b>{interval}s</b>
    </div>""", unsafe_allow_html=True)
    ph = st.empty()
    for r in range(interval, 0, -1):
        ph.info(f"⏱️ Next fetch in **{r}s** …"); time.sleep(1)
    ph.empty()
    n = do_fetch_and_save(topic, limit, use_ml=use_ml, ml_model=ml_key)
    if n: st.success(f"✅ Auto-fetched {n} new posts!")
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# MANUAL FETCH
# ══════════════════════════════════════════════════════════════════
if fetch_btn:
    if not topic.strip():
        st.warning("Enter a keyword to search for Reddit posts.")
    else:
        with st.spinner(f"Fetching posts about '{topic}' from Reddit …"):
            n = do_fetch_and_save(topic.strip(), limit, use_ml=use_ml, ml_model=ml_key)
        if n:
            st.success(f"✅ **{n}** posts fetched and analyzed for topic: **'{topic}'**")
        else:
            st.error("No posts found. Try another keyword.")


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
init_db()

# Topic filter
all_topics_in_db = []
try:
    conn = sqlite3.connect(DB_PATH)
    topics_df = pd.read_sql_query(f"SELECT DISTINCT topic FROM {TABLE}", conn)
    conn.close()
    all_topics_in_db = ["All"] + topics_df["topic"].tolist()
except:
    all_topics_in_db = ["All"]

selected_topic = st.selectbox(
    "📂 Filter by topic:",
    all_topics_in_db,
    key="topic_filter"
)

db_df = load_db(topic_filter=selected_topic)
total = len(db_df)


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════
if total > 0:
    COLOURS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}
    counts  = db_df["sentiment"].value_counts().to_dict()
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral",  0)

    # ── KPI Cards ─────────────────────────────────────────────────
    st.markdown("## 📊 Live Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, total, "📦 Total Posts", "#ff4500"),
        (c2, pos,   "😊 Positive",    "#4CAF50"),
        (c3, neg,   "😞 Negative",    "#F44336"),
        (c4, neu,   "😐 Neutral",     "#2196F3"),
    ]:
        col.markdown(f"""
        <div class="stat-card" style="border-bottom-color:{color}">
            <p>{label}</p>
            <h3 style="color:{color}">{val:,}</h3>
            <p>{round(val/total*100,1)}% of total</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### Sentiment Distribution")
        labels = [k for k, v in counts.items() if v > 0]
        sizes  = [counts[k] for k in labels]
        clrs   = [COLOURS[k] for k in labels]
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=clrs,
            startangle=140, wedgeprops={"edgecolor":"white","linewidth":2.5},
            textprops={"fontsize":12})
        for at in autotexts:
            at.set_color("white"); at.set_fontweight("bold")
        ax.set_title(f"Sentiment Split — {total} Posts", fontsize=12, fontweight="bold")
        st.pyplot(fig); plt.close()

    with chart2:
        st.markdown("#### Posts per Category")
        bar_labels = list(counts.keys())
        bar_values = list(counts.values())
        bar_clrs   = [COLOURS.get(k, "#999") for k in bar_labels]
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

    # ── Sentiment Trend ───────────────────────────────────────────
    if len(db_df) >= 5:
        st.markdown("#### 📉 Sentiment Trend")
        batch_size = max(3, len(db_df)//15)
        trend_df   = (db_df.sort_values("id").reset_index(drop=True)
                      .assign(batch=lambda d: d.index // batch_size)
                      .groupby("batch")["vader_compound"].mean()
                      .reset_index())
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(trend_df["batch"], trend_df["vader_compound"],
                color="#ff4500", linewidth=2, marker="o", markersize=4)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.fill_between(trend_df["batch"], trend_df["vader_compound"], 0,
                        where=(trend_df["vader_compound"] >= 0),
                        alpha=0.15, color="#4CAF50", label="Positive")
        ax.fill_between(trend_df["batch"], trend_df["vader_compound"], 0,
                        where=(trend_df["vader_compound"] < 0),
                        alpha=0.15, color="#F44336", label="Negative")
        ax.set_ylabel("Avg VADER Score"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    # ── ML vs VADER Comparison ────────────────────────────────────
    if ML_AVAILABLE and "ml_sentiment" in db_df.columns:
        ml_data = db_df[db_df["ml_sentiment"] != "N/A"]
        if len(ml_data) > 0:
            st.markdown("---")
            st.markdown("## 🤖 ML Model vs VADER — Comparison")
            ml_counts = ml_data["ml_sentiment"].value_counts().to_dict()

            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown("**VADER + TextBlob (Rule-based)**")
                vader_counts = ml_data["sentiment"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(vader_counts.values,
                       labels=vader_counts.index,
                       autopct="%1.1f%%",
                       colors=[COLOURS.get(k,"#aaa") for k in vader_counts.index],
                       startangle=140,
                       wedgeprops={"edgecolor":"white","linewidth":2})
                ax.set_title("VADER + TextBlob", fontsize=12, fontweight="bold")
                st.pyplot(fig); plt.close()

            with cmp2:
                st.markdown(f"**{ml_model} (Machine Learning)**")
                ml_s = pd.Series(ml_counts)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(ml_s.values,
                       labels=ml_s.index,
                       autopct="%1.1f%%",
                       colors=[COLOURS.get(k,"#aaa") for k in ml_s.index],
                       startangle=140,
                       wedgeprops={"edgecolor":"white","linewidth":2})
                ax.set_title(ml_model, fontsize=12, fontweight="bold")
                st.pyplot(fig); plt.close()

            # Agreement rate
            if len(ml_data) > 0:
                agreed = (ml_data["sentiment"] == ml_data["ml_sentiment"]).sum()
                agree_pct = round(agreed / len(ml_data) * 100, 1)
                if agree_pct >= 70:
                    st.success(f"✅ VADER and ML model only agree on **{agree_pct}%** posts")
                else:
                    st.warning(f"⚠️ VADER and ML model only agree on **{agree_pct}%** posts")

    # ── Word Clouds ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ☁️ Word Clouds")
    st.caption("Big word = appeared many times in the posts")

    t1, t2, t3, t4 = st.tabs(["🌐 All Posts", "😊 Positive", "😞 Negative", "😐 Neutral"])
    colormaps = {"all":"plasma", "pos":"YlGn", "neg":"OrRd", "neu":"Blues"}

    with t1:
        fig = generate_wordcloud_fig(db_df["clean_text"], f"All Posts — '{selected_topic}'", colormaps["all"])
        if fig: st.pyplot(fig, use_container_width=True); plt.close()

    with t2:
        d = db_df[db_df["sentiment"]=="Positive"]
        if len(d) > 0:
            fig = generate_wordcloud_fig(d["clean_text"], "Positive Posts", colormaps["pos"])
            if fig: st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.info("No positive posts so far.")

    with t3:
        d = db_df[db_df["sentiment"]=="Negative"]
        if len(d) > 0:
            fig = generate_wordcloud_fig(d["clean_text"], "Negative Posts", colormaps["neg"])
            if fig: st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.info("No negative posts so far.")

    with t4:
        d = db_df[db_df["sentiment"]=="Neutral"]
        if len(d) > 0:
            fig = generate_wordcloud_fig(d["clean_text"], "Neutral Posts", colormaps["neu"])
            if fig: st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.info("No neutral posts so far.")

    # ── Hashtag Trends ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Trending Hashtags")
    st.caption("#Hashtag detect in the posts")

    raw_texts  = db_df["raw_text"].dropna().tolist()
    sentiments = db_df["sentiment"].tolist()
    ht_df      = get_top_hashtags(raw_texts, top_n=15, min_count=1)

    if ht_df.empty:
        st.info("No hashtags found in these posts. Reddit posts usually use fewer hashtags.")
    else:
        fig, ax = plt.subplots(figsize=(10, max(4, len(ht_df)*0.5)))
        bar_clrs = plt.cm.viridis(np.linspace(0.2, 0.85, len(ht_df)))[::-1]
        ax.barh(ht_df["hashtag"][::-1], ht_df["count"][::-1],
                color=bar_clrs, edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Occurrences"); ax.set_title("Top Hashtags", fontsize=13, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Topic Comparison ──────────────────────────────────────────
    if db_df["topic"].nunique() > 1:
        st.markdown("---")
        st.markdown("## 📊 Topic vs Topic Comparison")
        st.caption("Alag alag keywords ka sentiment compare karo")

        topic_df = (db_df.groupby(["topic","sentiment"]).size()
                    .reset_index(name="count")
                    .pivot(index="topic", columns="sentiment", values="count")
                    .fillna(0).astype(int))

        for col in ["Positive","Negative","Neutral"]:
            if col not in topic_df.columns:
                topic_df[col] = 0
        topic_df["Total"] = topic_df[["Positive","Negative","Neutral"]].sum(axis=1)
        topic_df["Positivity %"] = (topic_df["Positive"] / topic_df["Total"] * 100).round(1)

        st.dataframe(topic_df, use_container_width=True)

        # Visual comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(topic_df))
        w = 0.28
        ax.bar(x - w, topic_df.get("Positive",0), w, label="Positive", color="#4CAF50", alpha=0.88)
        ax.bar(x,      topic_df.get("Negative",0), w, label="Negative", color="#F44336", alpha=0.88)
        ax.bar(x + w,  topic_df.get("Neutral", 0), w, label="Neutral",  color="#2196F3", alpha=0.88)
        ax.set_xticks(x); ax.set_xticklabels(topic_df.index, rotation=20, ha="right")
        ax.set_ylabel("Posts"); ax.legend()
        ax.set_title("Topic Sentiment Comparison", fontsize=13, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Posts Table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🗂️ Fetched Posts")
    n_show = st.slider("Rows to display:", 5, min(200, total), 20, step=5)
    show_cols = ["id","topic","raw_text","sentiment","ml_sentiment",
                 "vader_compound","tb_polarity","saved_at"]
    avail = [c for c in show_cols if c in db_df.columns]

    def _hl(val):
        return {
            "Positive": "background-color:#c8e6c9;color:#1b5e20;font-weight:600",
            "Negative": "background-color:#ffcdd2;color:#b71c1c;font-weight:600",
            "Neutral":  "background-color:#bbdefb;color:#0d47a1;font-weight:600",
        }.get(val, "")

    disp = db_df.head(n_show)[avail]
    sent_cols = [c for c in ["sentiment","ml_sentiment"] if c in disp.columns]
    st.dataframe(disp.style.applymap(_hl, subset=sent_cols),
                 use_container_width=True, height=400)

    st.download_button(
        "⬇️ Export CSV",
        data=db_df.to_csv(index=False).encode("utf-8"),
        file_name=f"reddit_{selected_topic}_sentiment.csv",
        mime="text/csv"
    )

else:
    # ── Empty State ───────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:80px;color:#aaa;">
        <div style="font-size:5rem">🔴</div>
        <h3 style="color:#ccc">no posts found</h3>
        <p>Enter a keyword in the sidebar and click <b>Fetch & Analyze</b></p>
        <p style="font-size:0.9rem;margin-top:20px">
            Popular topics: cricket · pakistan · chatgpt · football · bitcoin
        </p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════
with st.expander("ℹ️ About this module"):
    st.markdown("""
**How it works:**
- Uses Reddit's **free public RSS feed** — no API key required
- URL format: `https://www.reddit.com/search.rss?q=<keyword>&sort=new`
- Jo bhi keyword do — latest Reddit posts fetch ho jaate hain
- VADER + TextBlob + ML Models (agar trained hain) se sentiment lagata hai
- SQLite database mein save karta hai — dashboard pe dikhata hai

**Features:**
- ✅ 10 quick topic buttons
- ✅ Auto Refresh (30s se 5 min tak)
- ✅ ML Model vs VADER comparison
- ✅ Word Cloud (4 tabs — All/Positive/Negative/Neutral)
- ✅ Hashtag detection
- ✅ Topic vs Topic comparison
- ✅ CSV export
    """)