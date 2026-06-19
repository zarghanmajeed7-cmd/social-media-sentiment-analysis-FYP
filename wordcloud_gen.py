# ============================================================
# wordcloud_gen.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Purpose : Generate positive/negative word clouds from post text
#           for display in the Streamlit dashboard.
# ============================================================

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Extended stopwords — remove high-frequency noise words that add no insight
_EXTRA = {
    "rt", "amp", "will", "now", "just", "got", "get", "like",
    "one", "day", "today", "time", "year", "know", "think",
    "want", "going", "said", "say", "go", "made", "make",
    "still", "much", "really", "im", "ive", "dont", "cant",
    "via", "new", "us", "u", "ur", "lol", "http", "https",
    "www", "would", "could", "also", "even", "back", "good",
    "see", "people", "thing", "things", "way", "something",
}
STOPWORDS_SET = STOPWORDS.union(_EXTRA)

# Colour palettes
_COLOURS = {
    "positive": "YlGn",   # green shades
    "negative": "OrRd",   # red/orange shades
}

BG_COLOUR = "#0e1117"   # matches Streamlit dark theme


def _prepare_text(texts: list[str]) -> str:
    """Join and clean a list of post strings into a single corpus."""
    combined = " ".join(str(t) for t in texts if t)
    combined = re.sub(r"http\S+|www\S+",  " ", combined)
    combined = re.sub(r"@\w+",            " ", combined)
    combined = re.sub(r"#(\w+)",          r"\1", combined)  # keep word
    combined = re.sub(r"[^a-zA-Z\s]",    " ", combined)
    combined = re.sub(r"\s+",             " ", combined).strip()
    return combined.lower()


def _make_wc(text: str, colormap: str, max_words: int, w: int, h: int) -> WordCloud:
    return WordCloud(
        width=w,
        height=h,
        background_color=BG_COLOUR,
        colormap=colormap,
        stopwords=STOPWORDS_SET,
        max_words=max_words,
        collocations=False,
        prefer_horizontal=0.85,
        min_font_size=10,
    ).generate(text or "no data available")


def generate_wordcloud_figure(
    positive_texts: list[str],
    negative_texts: list[str],
    max_words: int = 100,
) -> plt.Figure:
    """
    Returns a matplotlib Figure with two side-by-side word clouds:
      Left  — positive posts (green palette)
      Right — negative posts (red palette)
    """
    pos_text = _prepare_text(positive_texts)
    neg_text = _prepare_text(negative_texts)

    wc_pos = _make_wc(pos_text, _COLOURS["positive"], max_words, 800, 400)
    wc_neg = _make_wc(neg_text, _COLOURS["negative"], max_words, 800, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG_COLOUR)

    ax1.imshow(wc_pos, interpolation="bilinear")
    ax1.set_title("😊 Positive Words", color="#00cc66",
                  fontsize=14, pad=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(wc_neg, interpolation="bilinear")
    ax2.set_title("😞 Negative Words", color="#ff4444",
                  fontsize=14, pad=12, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout(pad=1.5)
    return fig


def generate_single_wordcloud(
    texts: list[str],
    sentiment: str = "positive",
    max_words: int = 150,
) -> plt.Figure:
    """
    Returns a full-width word cloud for one sentiment category.

    Parameters
    ----------
    texts     : list of post strings
    sentiment : 'positive' | 'negative'
    max_words : cap on number of words shown
    """
    colormap = _COLOURS.get(sentiment, "Blues")
    text     = _prepare_text(texts)
    wc       = _make_wc(text, colormap, max_words, 1200, 500)

    title_colour = "#00cc66" if sentiment == "positive" else "#ff4444"
    emoji        = "😊" if sentiment == "positive" else "😞"

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG_COLOUR)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(f"{emoji} {sentiment.capitalize()} Word Cloud",
                 color=title_colour, fontsize=15, pad=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    return fig
