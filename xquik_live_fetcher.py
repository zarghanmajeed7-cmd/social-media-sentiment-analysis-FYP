"""Streamlit page for optional Xquik live X post loading."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from xquik_source import XquikSourceError, fetch_x_posts

_VADER = SentimentIntensityAnalyzer()


def analyse_text(text: str) -> dict[str, object]:
    """Score one post with the same VADER and TextBlob style used by the project."""
    vader_compound = _VADER.polarity_scores(text)["compound"]
    textblob_polarity = TextBlob(text).sentiment.polarity
    average_score = (vader_compound + textblob_polarity) / 2.0

    if average_score >= 0.05:
        sentiment = "Positive"
    elif average_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "vader_compound": round(vader_compound, 4),
        "tb_polarity": round(textblob_polarity, 4),
        "sentiment": sentiment,
    }


def build_dataframe(posts: list[dict[str, str]]) -> pd.DataFrame:
    """Convert Xquik records into the dashboard-friendly sentiment table."""
    rows: list[dict[str, object]] = []
    for post in posts:
        scores = analyse_text(post["text"])
        rows.append(
            {
                "source": "X",
                "raw_text": post["text"],
                "url": post["url"],
                "published": post["published"],
                **scores,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Xquik Live X Sentiment", page_icon="X", layout="wide")
    st.title("Xquik Live X Sentiment")
    st.caption("Load recent X posts into the same VADER and TextBlob scoring flow.")

    query = st.text_input("X search query", placeholder='social media analytics OR "customer review"')
    limit = st.slider("Posts to fetch", min_value=5, max_value=50, value=25, step=5)

    if st.button("Fetch and Analyse X Posts", type="primary", disabled=(len(query.strip()) < 2)):
        try:
            posts = fetch_x_posts(query, limit=limit)
        except XquikSourceError as exc:
            st.warning(str(exc))
            return

        df = build_dataframe(posts)
        st.success(f"Fetched and analysed {len(df)} X posts.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Export X Sentiment CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="xquik_x_sentiment.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
