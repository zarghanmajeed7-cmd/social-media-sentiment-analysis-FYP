# ============================================================
# MODULE 1: data_collection.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Author  : Final Year CS Student
# Purpose : Load publicly available social media datasets from
#           local CSV files (no paid APIs required).
# ============================================================

import pandas as pd
import os


# ------------------------------------------------------------------
# CONFIGURATION — update this path to point at your dataset file
# ------------------------------------------------------------------
DEFAULT_DATASET_PATH = "data/tweets.csv"

# Column name in your CSV that contains the raw post/tweet text
TEXT_COLUMN = "text"


def load_dataset(filepath: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Load a social-media dataset from a local CSV file.

    Expected CSV structure (minimum required column):
        text  — the raw post or tweet body

    Optional columns that will be preserved if present:
        id, username, created_at, hashtags, likes, retweets

    Parameters
    ----------
    filepath : str
        Path to the CSV file relative to the project root.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the raw dataset rows.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the required 'text' column is missing from the file.
    """

    # Check the file actually exists before trying to read it
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please place a CSV file with a 'text' column inside the 'data/' folder.\n"
            "You can download a free Twitter sentiment dataset from:\n"
            "  https://www.kaggle.com/datasets/kazanova/sentiment140"
        )

    print(f"[data_collection] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    # Validate that the required text column is present
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"The dataset must contain a column named '{TEXT_COLUMN}'.\n"
            f"Columns found: {list(df.columns)}"
        )

    # Drop rows where the text cell is empty / NaN
    original_len = len(df)
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
    dropped = original_len - len(df)

    print(f"[data_collection] Loaded {len(df)} records "
          f"({dropped} empty rows removed).")

    return df


def create_sample_dataset(filepath: str = DEFAULT_DATASET_PATH,
                          n_samples: int = 200) -> None:
    """
    Generate a small synthetic dataset so the project can be
    demonstrated without downloading an external file.

    Parameters
    ----------
    filepath : str
        Where to save the generated CSV.
    n_samples : int
        Number of sample rows to create.
    """

    import random

    # A diverse pool of positive, negative, and neutral sentences
    positive_texts = [
        "I absolutely love this product! It changed my life.",
        "What a beautiful sunny day, feeling grateful today!",
        "Just got promoted at work. So happy and excited!",
        "This new update is amazing, the team did a fantastic job.",
        "Spending time with family is the best therapy. Blessed!",
        "Completed my final year project. Feeling accomplished!",
        "The concert was incredible, best night of my life.",
        "Delicious food at the new restaurant, highly recommend it.",
        "Finally finished reading that book. Absolutely brilliant.",
        "Great customer service, will definitely buy again.",
    ]
    negative_texts = [
        "This service is absolutely terrible, never using it again.",
        "Traffic is a nightmare today, stuck for two hours.",
        "Lost my wallet, worst day ever. So frustrated!",
        "The app keeps crashing, totally useless update.",
        "Really disappointed with the quality of this product.",
        "Cancelled my subscription, customer support was horrible.",
        "Failed my exam despite studying so hard. Heartbroken.",
        "The weather is miserable and my mood matches it.",
        "Prices keep rising but quality keeps falling. Unacceptable!",
        "So tired of all the negativity on social media.",
    ]
    neutral_texts = [
        "Just woke up and made myself a cup of coffee.",
        "The meeting has been rescheduled to Thursday at 3 PM.",
        "Watching the news while having lunch.",
        "New software update available, version 4.2.1 released.",
        "The library closes at 8 PM on weekdays.",
        "Ordered groceries online, delivery expected tomorrow.",
        "There are 24 hours in a day and 7 days in a week.",
        "The temperature today is 22 degrees Celsius in Faisalabad.",
        "Submitted the assignment before the deadline.",
        "The train arrives at platform 4.",
    ]

    all_texts = positive_texts + negative_texts + neutral_texts
    sampled_texts = [random.choice(all_texts) for _ in range(n_samples)]

    # Fake usernames and timestamps for a realistic-looking dataset
    usernames = [f"user_{random.randint(1000, 9999)}" for _ in range(n_samples)]
    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")

    sample_df = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "username": usernames,
        "text": sampled_texts,
        "created_at": timestamps,
    })

    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sample_df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"[data_collection] Sample dataset with {n_samples} rows "
          f"saved to '{filepath}'.")


# ------------------------------------------------------------------
# Quick test — run this file directly to verify loading works
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Create a sample file if none exists, then load it
    if not os.path.exists(DEFAULT_DATASET_PATH):
        print("[data_collection] No dataset found. Generating sample data...")
        create_sample_dataset()

    data = load_dataset()
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns      : {list(data.columns)}")
