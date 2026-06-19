# ============================================================
# 3_comments_analyser.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Purpose : Fetch & analyse comments from Reddit posts and
#           YouTube videos using free APIs
#
# INSTALL:
#   pip install wordcloud google-api-python-client
#
# RUN:
#   python -m streamlit run 3_comments_analyser.py
# ============================================================

import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import json
import re, string, os
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

STOP_WORDS = set(stopwords.words("english"))
_vader     = SentimentIntensityAnalyzer()

# ── ML Models (optional) ───────────────────────────────────────────
try:
    from ml_models import predict_sentiment, models_exist
    ML_AVAILABLE = models_exist()
except ImportError:
    ML_AVAILABLE = False

# ── YouTube API Key ────────────────────────────────────────────────
YOUTUBE_API_KEY = "AIzaSyA91ZTGmAF-gVYWRF3JY9Zcov2pX-0p_Sw"

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Comments Sentiment Analyser",
    page_icon="💬",
    layout="wide",
)

# ── Professional CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px 35px;
        border-radius: 16px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(17,153,142,0.3);
    }
    .main-banner h1 { margin: 0 0 8px 0; font-size: 2rem; font-weight: 700; }
    .main-banner p  { margin: 0; opacity: 0.9; font-size: 1rem; }

    .platform-card {
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        border: 2px solid transparent;
        margin-bottom: 12px;
    }
    .reddit-card  { background: #fff3f0; border-color: #ff4500; }
    .youtube-card { background: #fff0f0; border-color: #FF0000; }

    .comment-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #ccc;
    }
    .comment-positive { border-left-color: #4CAF50 !important; }
    .comment-negative { border-left-color: #F44336 !important; }
    .comment-neutral  { border-left-color: #2196F3 !important; }

    .stat-pill {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 4px;
        color: white;
    }

    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stTextInput input {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 10px 14px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-banner">
    <h1>💬 Comments Sentiment Analyser</h1>
    <p>CSI-630 Final Year Project &nbsp;·&nbsp;
       Govt. Municipal Graduate College, Faisalabad &nbsp;·&nbsp;
       Reddit Comments · YouTube Comments · VADER + TextBlob + ML Models · Word Cloud</p>
</div>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────
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

def analyse_comment(text, use_ml=False, ml_model="lr"):
    cleaned  = clean_text(text)
    compound = _vader.polarity_scores(cleaned)["compound"]
    blob     = TextBlob(cleaned)
    polarity = blob.sentiment.polarity
    avg      = (compound + polarity) / 2.0
    label    = "Positive" if avg >= 0.05 else ("Negative" if avg <= -0.05 else "Neutral")
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

def sentiment_color(label):
    return {"Positive":"#4CAF50","Negative":"#F44336","Neutral":"#2196F3"}.get(label,"#999")

def sentiment_emoji(label):
    return {"Positive":"😊","Negative":"😞","Neutral":"😐"}.get(label,"")

def generate_wordcloud(text_series, title, colormap="viridis"):
    all_text = " ".join(text_series.dropna().tolist())
    if not all_text.strip():
        st.info("Not enough text."); return
    wc = WordCloud(width=800, height=380, background_color="white",
                   colormap=colormap, max_words=80,
                   collocations=False, stopwords=STOP_WORDS).generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════
# REDDIT COMMENTS FETCHER
# ══════════════════════════════════════════════════════════════════

def extract_reddit_post_id(url: str) -> str:
    """
    Extract the post ID from a Reddit URL.
    Example: https://www.reddit.com/r/pakistan/comments/abc123/title/
    Returns: abc123
    """
    match = re.search(r"/comments/([a-zA-Z0-9]+)", url)
    return match.group(1) if match else ""

def fetch_reddit_comments(url: str, limit: int = 50) -> list:
    """
    Fetch comments from a Reddit post using Reddit's free JSON API.
    No API key needed — Reddit exposes post JSON at <url>.json
    """
    post_id = extract_reddit_post_id(url)
    if not post_id:
        st.error("Invalid Reddit URL. Make sure it contains '/comments/'")
        return []

    json_url = f"https://www.reddit.com/comments/{post_id}.json?limit={limit}"
    headers  = {"User-Agent": "Mozilla/5.0 (fyp-sentiment/1.0)"}

    try:
        req      = urllib.request.Request(json_url, headers=headers)
        response = urllib.request.urlopen(req, timeout=15)
        data     = json.loads(response.read())
    except Exception as e:
        st.error(f"Could not fetch Reddit comments: {e}")
        return []

    comments = []
    try:
        # Reddit JSON structure: data[1] contains comments listing
        comment_listing = data[1]["data"]["children"]
        for item in comment_listing:
            if item["kind"] == "t1":   # t1 = comment
                body = item["data"].get("body", "").strip()
                author = item["data"].get("author", "unknown")
                score  = item["data"].get("score", 0)
                if body and body != "[deleted]" and body != "[removed]":
                    comments.append({
                        "text":     body,
                        "author":   author,
                        "score":    score,
                        "platform": "Reddit",
                    })
    except (KeyError, IndexError) as e:
        st.error(f"Error parsing Reddit response: {e}")
        return []

    return comments[:limit]


# ══════════════════════════════════════════════════════════════════
# YOUTUBE COMMENTS FETCHER
# ══════════════════════════════════════════════════════════════════

def extract_youtube_video_id(url: str) -> str:
    """
    Extract video ID from various YouTube URL formats.
    Works with: youtube.com/watch?v=ID, youtu.be/ID
    """
    patterns = [
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""

def fetch_youtube_comments(url: str, limit: int = 50) -> list:
    """
    Fetch comments from a YouTube video using the free YouTube Data API v3.
    Requires a YouTube API key (free from Google Cloud Console).
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Use format: youtube.com/watch?v=VIDEO_ID")
        return []

    # Build API request URL
    params = urllib.parse.urlencode({
        "part":       "snippet",
        "videoId":    video_id,
        "maxResults": min(limit, 100),
        "order":      "relevance",
        "key":        YOUTUBE_API_KEY,
        "textFormat": "plainText",
    })
    api_url = f"https://www.googleapis.com/youtube/v3/commentThreads?{params}"

    try:
        req      = urllib.request.Request(api_url)
        response = urllib.request.urlopen(req, timeout=15)
        data     = json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_body = json.loads(e.read())
        msg = error_body.get("error", {}).get("message", str(e))
        st.error(f"YouTube API Error: {msg}")
        return []
    except Exception as e:
        st.error(f"Could not fetch YouTube comments: {e}")
        return []

    comments = []
    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        text    = snippet.get("textDisplay", "").strip()
        author  = snippet.get("authorDisplayName", "unknown")
        likes   = snippet.get("likeCount", 0)
        if text:
            comments.append({
                "text":     text,
                "author":   author,
                "score":    likes,
                "platform": "YouTube",
            })

    return comments


# ══════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════

# ── Platform Selector ──────────────────────────────────────────────
st.markdown("## 🌐 Select Platform")
platform = st.radio(
    "",
    options=["🔴 Reddit Post Comments", "▶️ YouTube Video Comments"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")

# ── Input Section ──────────────────────────────────────────────────
if "Reddit" in platform:
    st.markdown("### 🔴 Reddit Post Comment Analyser")
    st.info("📋 **How to get Reddit URL:** Open any Reddit post → Copy the URL from browser address bar")

    reddit_url = st.text_input(
        "Paste Reddit Post URL:",
        placeholder="https://www.reddit.com/r/pakistan/comments/abc123/post_title/",
    )
    reddit_limit  = st.slider("Max comments to fetch:", 10, 100, 50, step=10)
    reddit_btn    = st.button("💬 Fetch & Analyse Reddit Comments",
                               use_container_width=True)

    if reddit_btn:
        if not reddit_url.strip():
            st.warning("Please enter a Reddit post URL.")
        else:
            with st.spinner("Fetching Reddit comments …"):
                raw_comments = fetch_reddit_comments(reddit_url.strip(), reddit_limit)

            if raw_comments:
                st.session_state["comments_df"] = pd.DataFrame(raw_comments)
                st.session_state["source_url"]  = reddit_url
                st.session_state["platform"]    = "Reddit"
                st.success(f"✅ Fetched **{len(raw_comments)}** Reddit comments!")
            else:
                st.error("No comments found. Check the URL and try again.")

else:
    st.markdown("### ▶️ YouTube Video Comment Analyser")
    st.info("📋 **How to get YouTube URL:** Open any YouTube video → Copy URL from address bar")

    yt_url   = st.text_input(
        "Paste YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    yt_limit = st.slider("Max comments to fetch:", 10, 100, 50, step=10)
    yt_btn   = st.button("💬 Fetch & Analyse YouTube Comments",
                          use_container_width=True)

    if yt_btn:
        if not yt_url.strip():
            st.warning("Please enter a YouTube video URL.")
        else:
            with st.spinner("Fetching YouTube comments …"):
                raw_comments = fetch_youtube_comments(yt_url.strip(), yt_limit)

            if raw_comments:
                st.session_state["comments_df"] = pd.DataFrame(raw_comments)
                st.session_state["source_url"]  = yt_url
                st.session_state["platform"]    = "YouTube"
                st.success(f"✅ Fetched **{len(raw_comments)}** YouTube comments!")
            else:
                st.error("No comments found. Comments may be disabled or URL is invalid.")


# ── Results ────────────────────────────────────────────────────────
if "comments_df" in st.session_state:
    df       = st.session_state["comments_df"].copy()
    platform_name = st.session_state.get("platform", "")
    source   = st.session_state.get("source_url", "")

    st.markdown("---")
    st.markdown(f"## 📊 Analysis Results — {platform_name}")
    st.caption(f"Source: {source}")

    # ML settings
    use_ml_comments = False
    ml_key_c = "lr"
    if ML_AVAILABLE:
        st.markdown("---")
        ml_c1, ml_c2 = st.columns(2)
        use_ml_comments = ml_c1.toggle("Use ML Sentiment", value=True, key="ca_ml_toggle")
        ml_choice_c = ml_c2.radio("ML Model:", ["Logistic Regression", "Random Forest"],
                                   horizontal=True, key="ca_ml_model")
        ml_key_c = "lr" if ml_choice_c == "Logistic Regression" else "rf"

    # Analyse all comments
    with st.spinner("Analysing sentiment for all comments …"):
        results = pd.DataFrame(list(
            df["text"].apply(lambda t: analyse_comment(t, use_ml=use_ml_comments, ml_model=ml_key_c))
        ))
        df      = pd.concat([df, results], axis=1)

    counts = df["sentiment"].value_counts().to_dict()
    total  = len(df)
    pos    = counts.get("Positive", 0)
    neg    = counts.get("Negative", 0)
    neu    = counts.get("Neutral",  0)

    COLOURS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}

    # ── KPI Row ───────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💬 Total Comments", total)
    k2.metric("😊 Positive", f"{pos}  ({round(pos/total*100,1)}%)")
    k3.metric("😞 Negative", f"{neg}  ({round(neg/total*100,1)}%)")
    k4.metric("😐 Neutral",  f"{neu}  ({round(neu/total*100,1)}%)")

    # ── ML vs VADER comparison ───────────────────────────────────
    if ML_AVAILABLE and use_ml_comments and "ml_sentiment" in df.columns:
        ml_data = df[df["ml_sentiment"] != "N/A"]
        if len(ml_data) > 0:
            agreed = (ml_data["sentiment"] == ml_data["ml_sentiment"]).sum()
            agree_pct = round(agreed / len(ml_data) * 100, 1)
            ml_counts_c = ml_data["ml_sentiment"].value_counts().to_dict()
            ml_pos = ml_counts_c.get("Positive",0)
            ml_neg = ml_counts_c.get("Negative",0)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("ML Positive",  ml_pos)
            m2.metric("ML Negative",  ml_neg)
            m3.metric("ML Neutral",   ml_counts_c.get("Neutral",0))
            if agree_pct >= 70:
                m4.metric("Model Agreement", f"{agree_pct}%", delta="High")
            else:
                m4.metric("Model Agreement", f"{agree_pct}%", delta="Low", delta_color="inverse")

    # ── Overall Verdict Banner ────────────────────────────────────
    dominant = max(counts, key=counts.get)
    dom_color = COLOURS[dominant]
    dom_emoji = sentiment_emoji(dominant)
    st.markdown(f"""
    <div style="background:{dom_color}15;border:2px solid {dom_color};
                border-radius:12px;padding:16px 24px;margin:16px 0;text-align:center">
        <h3 style="color:{dom_color};margin:0">
            {dom_emoji} Overall Sentiment: <b>{dominant}</b>
        </h3>
        <p style="color:#666;margin:4px 0 0 0">
            Based on {total} comments — {dominant} sentiment dominates with {counts[dominant]} comments
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment Distribution")
        labels = [k for k,v in counts.items() if v > 0]
        sizes  = [counts[k] for k in labels]
        clrs   = [COLOURS[k] for k in labels]
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=clrs,
            startangle=140, wedgeprops={"edgecolor":"white","linewidth":2.5},
            textprops={"fontsize":12})
        for at in autotexts:
            at.set_color("white"); at.set_fontweight("bold")
        ax.set_title(f"Comment Sentiment — {platform_name}",
                     fontsize=12, fontweight="bold")
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Comment Count per Category")
        fig, ax = plt.subplots(figsize=(5, 5))
        bar_labels = list(counts.keys())
        bar_values = list(counts.values())
        bar_clrs   = [COLOURS.get(k,"#999") for k in bar_labels]
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

    # ── Word Clouds ────────────────────────────────────────────────
    st.markdown("## ☁️ Word Clouds")
    wt1, wt2, wt3, wt4 = st.tabs(["🌐 All","😊 Positive","😞 Negative","😐 Neutral"])
    with wt1: generate_wordcloud(df["clean_text"], "All Comments", "viridis")
    with wt2:
        d = df[df["sentiment"]=="Positive"]
        generate_wordcloud(d["clean_text"],"Positive Comments","Greens") if len(d)>0 else st.info("No positive comments.")
    with wt3:
        d = df[df["sentiment"]=="Negative"]
        generate_wordcloud(d["clean_text"],"Negative Comments","Reds") if len(d)>0 else st.info("No negative comments.")
    with wt4:
        d = df[df["sentiment"]=="Neutral"]
        generate_wordcloud(d["clean_text"],"Neutral Comments","Blues") if len(d)>0 else st.info("No neutral comments.")

    # ── Comments List ─────────────────────────────────────────────
    st.markdown("## 💬 Individual Comments")
    filter_opt = st.selectbox("Filter by sentiment:",
                               ["All", "Positive", "Negative", "Neutral"])
    filtered = df if filter_opt == "All" else df[df["sentiment"]==filter_opt]

    for _, row in filtered.head(30).iterrows():
        label = row["sentiment"]
        color = COLOURS[label]
        emoji = sentiment_emoji(label)
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
            <p style="margin:0;color:#444;line-height:1.5">{row['text'][:300]}</p>
            <div style="margin-top:8px;font-size:0.8rem;color:#999">
                VADER: {row['vader_compound']} &nbsp;|&nbsp;
                TextBlob: {row['tb_polarity']} &nbsp;|&nbsp;
                ML: {row.get('ml_sentiment','N/A')} &nbsp;|&nbsp;
                👍 {row.get('score',0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────
    st.markdown("---")
    export_cols = ["text","author","sentiment","vader_compound","tb_polarity","score","platform"]
    export_df   = df[[c for c in export_cols if c in df.columns]]
    st.download_button(
        "⬇️ Export Comments as CSV (for Power BI)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{platform_name.lower()}_comments_sentiment.csv",
        mime="text/csv",
    )

else:
    st.markdown("""
    <div style="text-align:center;padding:80px;color:#aaa;">
        <div style="font-size:5rem">💬</div>
        <h3 style="color:#ccc">No comments analysed yet</h3>
        <p>Select a platform above, paste a URL, and click Fetch & Analyse</p>
    </div>
    """, unsafe_allow_html=True)


# ── How It Works ───────────────────────────────────────────────────
with st.expander("ℹ️ How This Works"):
    st.markdown("""
**Reddit Comments** — Uses Reddit's free public JSON API:
```
https://www.reddit.com/comments/<post_id>.json
```
No API key needed. Just paste any Reddit post URL.

**YouTube Comments** — Uses YouTube Data API v3 (free, 10,000 units/day):
```
https://www.googleapis.com/youtube/v3/commentThreads
```
Free Google API key used automatically.

**Sentiment Analysis:**
- Each comment is cleaned (remove URLs, emojis, stop-words)
- VADER + TextBlob analyse cleaned text
- Ensemble average gives final label
- Results shown per comment + overall summary
    """)