# ============================================================
# MODULE 2: preprocessing.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Author  : Final Year CS Student
# Purpose : Clean and tokenize raw social-media text so it is
#           ready for sentiment analysis.
# Pipeline:
#   raw text → lowercase → remove URLs → remove mentions (@user)
#           → remove hashtag symbols → remove emojis
#           → remove punctuation/numbers → remove stop-words
#           → tokenize → rejoin as clean string
# ============================================================

import re
import string

import pandas as pd
import nltk

# Download required NLTK resources the first time this module runs.
# 'stopwords' contains common English words (the, is, a …) that
# carry no sentiment signal.  'punkt' is the tokenizer model.
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

# Build the English stop-word set once (reused for every row)
STOP_WORDS = set(stopwords.words("english"))

# Column name produced by this module
CLEAN_TEXT_COLUMN = "clean_text"


# ------------------------------------------------------------------
# Individual cleaning helper functions
# ------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www. links."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_mentions(text: str) -> str:
    """Remove Twitter/Instagram @mentions."""
    return re.sub(r"@\w+", "", text)


def remove_hashtag_symbols(text: str) -> str:
    """
    Strip the '#' symbol but keep the word itself.
    e.g. '#Python' → 'Python'
    This preserves the content while removing the special character.
    """
    return re.sub(r"#(\w+)", r"\1", text)


def remove_emojis(text: str) -> str:
    """
    Remove emoji characters using a broad Unicode range pattern.
    Covers emoticons, symbols, transport, flags, etc.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map symbols
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_punctuation_and_numbers(text: str) -> str:
    """Remove all punctuation marks and numeric digits."""
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove digits
    text = re.sub(r"\d+", "", text)
    return text


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------
# Main cleaning function — applies all steps in order
# ------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Apply the full text-cleaning pipeline to a single string.

    Steps applied (in order):
    1. Convert to lowercase
    2. Remove URLs
    3. Remove @mentions
    4. Remove hashtag '#' symbols (keep the word)
    5. Remove emojis
    6. Remove punctuation and numbers
    7. Tokenize and strip stop-words
    8. Rejoin tokens into a clean string

    Parameters
    ----------
    text : str
        Raw social-media post or tweet.

    Returns
    -------
    str
        Cleaned, tokenized text ready for sentiment analysis.
    """

    if not isinstance(text, str):
        return ""                        # handle NaN or non-string values

    text = text.lower()                  # Step 1 — lowercase
    text = remove_urls(text)             # Step 2 — URLs
    text = remove_mentions(text)         # Step 3 — @mentions
    text = remove_hashtag_symbols(text)  # Step 4 — hashtag symbols
    text = remove_emojis(text)           # Step 5 — emojis
    text = remove_punctuation_and_numbers(text)  # Step 6

    # Step 7 — tokenize and remove stop-words
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in STOP_WORDS and len(word) > 1   # skip single chars
    ]

    # Step 8 — rejoin into a single clean string
    clean = " ".join(tokens)
    clean = remove_extra_whitespace(clean)

    return clean


# ------------------------------------------------------------------
# DataFrame-level function — cleans an entire column at once
# ------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame,
                          text_col: str = "text") -> pd.DataFrame:
    """
    Add a 'clean_text' column to a DataFrame by cleaning every row
    in the specified text column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw social-media text.
    text_col : str
        Name of the column that holds the raw text.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with a new 'clean_text' column appended.
    """

    print(f"[preprocessing] Cleaning {len(df)} rows …")

    df = df.copy()   # avoid mutating the original DataFrame

    # Apply the cleaning pipeline to every row
    df[CLEAN_TEXT_COLUMN] = df[text_col].apply(clean_text)

    # Drop rows where cleaning produced an empty string
    before = len(df)
    df = df[df[CLEAN_TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)
    removed = before - len(df)

    print(f"[preprocessing] Done. {len(df)} rows retained "
          f"({removed} empty after cleaning).")

    return df


# ------------------------------------------------------------------
# Quick test — run this file directly to verify the pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I LOVE this! 😍 Check it out at https://example.com #Python @john",
        "Terrible service. Never coming back!! 😡 #disappointed @company123",
        "Just another normal Tuesday. Nothing special today.",
        "Big update released! 🚀 Version 2.0 is live now. #coding #dev",
    ]

    sample_df = pd.DataFrame({"text": sample_texts})
    result_df = preprocess_dataframe(sample_df)

    print("\nCleaning Results:")
    print("-" * 60)
    for _, row in result_df.iterrows():
        print(f"  RAW  : {row['text']}")
        print(f"  CLEAN: {row['clean_text']}")
        print()
