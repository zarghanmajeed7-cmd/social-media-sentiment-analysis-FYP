# ============================================================
# hashtag_trends.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Purpose : Extract hashtags from post text, count frequency,
#           and compute sentiment breakdown per hashtag.
# ============================================================

import re
from collections import Counter
import pandas as pd


def extract_hashtags(text: str) -> list[str]:
    """Return list of lowercased hashtag strings (without #) from text."""
    return [tag.lower() for tag in re.findall(r"#(\w+)", str(text))]


def get_top_hashtags(
    texts: list[str],
    top_n: int = 20,
    min_count: int = 2,
) -> pd.DataFrame:
    """
    Count all hashtags across a list of post strings.

    Parameters
    ----------
    texts     : list of raw post strings
    top_n     : maximum hashtags to return
    min_count : ignore hashtags appearing fewer than this many times

    Returns
    -------
    DataFrame with columns: hashtag, count, percentage
    Returns empty DataFrame if no hashtags found.
    """
    all_tags: list[str] = []
    for text in texts:
        all_tags.extend(extract_hashtags(text))

    if not all_tags:
        return pd.DataFrame(columns=["hashtag", "count", "percentage"])

    counts = Counter(all_tags)
    total  = sum(counts.values())

    df = pd.DataFrame(counts.most_common(top_n), columns=["hashtag", "count"])
    df = df[df["count"] >= min_count].reset_index(drop=True)
    df["percentage"] = (df["count"] / total * 100).round(2)
    df["hashtag"]    = "#" + df["hashtag"]
    return df


def get_hashtag_sentiment_breakdown(
    texts: list[str],
    sentiments: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    For each top hashtag, count how many posts are Positive / Negative / Neutral.

    Parameters
    ----------
    texts      : list of post strings
    sentiments : list of sentiment labels aligned with texts
                 (e.g. ['Positive', 'Negative', 'Neutral'])
    top_n      : number of hashtags to include

    Returns
    -------
    DataFrame with columns:
        hashtag, total, Positive, Negative, Neutral, pos_ratio
    """
    records = []
    for text, sent in zip(texts, sentiments):
        for tag in extract_hashtags(text):
            records.append({"hashtag": "#" + tag, "sentiment": sent})

    if not records:
        return pd.DataFrame()

    df      = pd.DataFrame(records)
    grouped = df.groupby("hashtag")["sentiment"].value_counts().unstack(fill_value=0)

    for col in ["Positive", "Negative", "Neutral"]:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["total"]     = grouped[["Positive", "Negative", "Neutral"]].sum(axis=1)
    grouped["pos_ratio"] = (grouped["Positive"] / grouped["total"] * 100).round(1)
    grouped              = grouped.sort_values("total", ascending=False).head(top_n)
    grouped              = grouped.reset_index()

    return grouped[["hashtag", "total", "Positive", "Negative", "Neutral", "pos_ratio"]]
