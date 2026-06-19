# ============================================================
# MODULE 3: sentiment_analysis.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Author  : Final Year CS Student
# Purpose : Classify each post as Positive, Negative, or Neutral
#           using two pre-trained NLP models:
#             • VADER  — rule-based, best for short social-media text
#             • TextBlob — lexicon-based, adds a second opinion
#           A simple ensemble combines both scores for a final label.
# ============================================================

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialise the VADER analyser once (it loads a lexicon at startup)
vader_analyser = SentimentIntensityAnalyzer()


# ------------------------------------------------------------------
# VADER sentiment scorer
# ------------------------------------------------------------------

def get_vader_sentiment(text: str) -> dict:
    """
    Use VADER to score a piece of text.

    VADER returns four scores:
        neg  — proportion of negative sentiment
        neu  — proportion of neutral sentiment
        pos  — proportion of positive sentiment
        compound — normalised overall score in [-1, +1]

    The compound score is used to decide the final label:
        compound >=  0.05  → Positive
        compound <= -0.05  → Negative
        otherwise          → Neutral

    Parameters
    ----------
    text : str
        Cleaned post text.

    Returns
    -------
    dict with keys: vader_compound, vader_label
    """

    scores = vader_analyser.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "vader_compound": round(compound, 4),
        "vader_label":    label,
    }


# ------------------------------------------------------------------
# TextBlob sentiment scorer
# ------------------------------------------------------------------

def get_textblob_sentiment(text: str) -> dict:
    """
    Use TextBlob to score a piece of text.

    TextBlob provides:
        polarity   — float in [-1.0, +1.0]  (negative ↔ positive)
        subjectivity — float in [0.0, 1.0]  (objective ↔ subjective)

    Label thresholds (same as VADER for consistency):
        polarity >  0.05 → Positive
        polarity < -0.05 → Negative
        otherwise        → Neutral

    Parameters
    ----------
    text : str
        Cleaned post text.

    Returns
    -------
    dict with keys: tb_polarity, tb_subjectivity, tb_label
    """

    blob       = TextBlob(text)
    polarity   = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "tb_polarity":     round(polarity, 4),
        "tb_subjectivity": round(subjectivity, 4),
        "tb_label":        label,
    }


# ------------------------------------------------------------------
# Ensemble — combine VADER and TextBlob into one final label
# ------------------------------------------------------------------

def get_ensemble_sentiment(vader_compound: float,
                            tb_polarity:   float) -> str:
    """
    Combine VADER compound score and TextBlob polarity into a
    single final sentiment label.

    Strategy:
    - Average the two scores.
    - Apply the same ±0.05 threshold to the average.

    This simple ensemble reduces the effect of individual model
    errors and gives a more balanced classification.

    Parameters
    ----------
    vader_compound : float  — VADER compound score in [-1, +1]
    tb_polarity    : float  — TextBlob polarity in [-1, +1]

    Returns
    -------
    str — 'Positive', 'Negative', or 'Neutral'
    """

    avg_score = (vader_compound + tb_polarity) / 2.0

    if avg_score >= 0.05:
        return "Positive"
    elif avg_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ------------------------------------------------------------------
# Main function — analyse an entire DataFrame
# ------------------------------------------------------------------

def analyse_sentiment(df: pd.DataFrame,
                       text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add sentiment columns to a DataFrame by running both VADER
    and TextBlob on every row, then computing the ensemble label.

    New columns added:
        vader_compound   — VADER overall score
        vader_label      — VADER sentiment category
        tb_polarity      — TextBlob polarity score
        tb_subjectivity  — TextBlob subjectivity score
        tb_label         — TextBlob sentiment category
        sentiment        — Final ensemble label (used in dashboard)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a cleaned text column.
    text_col : str
        Name of the column with cleaned text.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with all new sentiment columns appended.
    """

    print(f"[sentiment_analysis] Analysing sentiment for {len(df)} rows …")

    df = df.copy()

    # Apply VADER to every row
    vader_results = df[text_col].apply(get_vader_sentiment)
    vader_df      = pd.DataFrame(list(vader_results))

    # Apply TextBlob to every row
    tb_results = df[text_col].apply(get_textblob_sentiment)
    tb_df      = pd.DataFrame(list(tb_results))

    # Merge scores back into the main DataFrame
    df = pd.concat([df, vader_df, tb_df], axis=1)

    # Compute the final ensemble label
    df["sentiment"] = df.apply(
        lambda row: get_ensemble_sentiment(
            row["vader_compound"], row["tb_polarity"]
        ),
        axis=1,
    )

    # Print a quick summary to the console
    counts = df["sentiment"].value_counts()
    print("[sentiment_analysis] Done.")
    print(f"  Positive: {counts.get('Positive', 0)}")
    print(f"  Negative: {counts.get('Negative', 0)}")
    print(f"  Neutral : {counts.get('Neutral',  0)}")

    return df


# ------------------------------------------------------------------
# Quick test — run this file directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_texts = [
        "I absolutely love this product it is amazing and wonderful",
        "terrible experience never buy this waste of money",
        "the meeting is scheduled for tomorrow morning",
        "feeling so happy today everything going great",
        "really disappointed with the service completely unacceptable",
    ]

    test_df = pd.DataFrame({"clean_text": test_texts})
    result  = analyse_sentiment(test_df)

    print("\nSentiment Analysis Results:")
    print("-" * 70)
    for _, row in result.iterrows():
        print(f"  TEXT   : {row['clean_text'][:55]}")
        print(f"  VADER  : {row['vader_label']}  (compound={row['vader_compound']})")
        print(f"  TextBlob: {row['tb_label']}  (polarity={row['tb_polarity']})")
        print(f"  FINAL  : {row['sentiment']}")
        print()
