# ============================================================
# MODULE 4: database.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Author  : Final Year CS Student
# Purpose : Persist sentiment analysis results to a local SQLite
#           database (no server or cloud setup required).
#           Provides helpers to insert, query, and summarise data.
# ============================================================

import sqlite3
import os
from datetime import datetime

import pandas as pd

# Path to the SQLite database file
DB_PATH = "data/analytics.db"

# Name of the main results table
TABLE_NAME = "sentiment_results"


# ------------------------------------------------------------------
# Connection helper
# ------------------------------------------------------------------

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) a SQLite database and return the connection.

    SQLite stores the entire database as a single file, making it
    ideal for academic and lightweight projects.

    Parameters
    ----------
    db_path : str
        Filepath for the SQLite database file.

    Returns
    -------
    sqlite3.Connection
    """

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    return conn


# ------------------------------------------------------------------
# Table initialisation
# ------------------------------------------------------------------

def initialise_database(db_path: str = DB_PATH) -> None:
    """
    Create the sentiment_results table if it does not already exist.

    Table schema:
        id              INTEGER  — auto-incrementing primary key
        raw_text        TEXT     — original post/tweet content
        clean_text      TEXT     — preprocessed text
        vader_compound  REAL     — VADER overall score [-1, +1]
        vader_label     TEXT     — VADER category
        tb_polarity     REAL     — TextBlob polarity score
        tb_subjectivity REAL     — TextBlob subjectivity score
        tb_label        TEXT     — TextBlob category
        sentiment       TEXT     — final ensemble label
        created_at      TEXT     — timestamp when record was inserted
    """

    conn = get_connection(db_path)
    cursor = conn.cursor()

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_text        TEXT,
        clean_text      TEXT,
        vader_compound  REAL,
        vader_label     TEXT,
        tb_polarity     REAL,
        tb_subjectivity REAL,
        tb_label        TEXT,
        sentiment       TEXT,
        created_at      TEXT
    );
    """

    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

    print(f"[database] Database initialised at '{db_path}'.")


# ------------------------------------------------------------------
# Insert results
# ------------------------------------------------------------------

def save_results(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """
    Insert all rows of a sentiment-analysed DataFrame into the
    SQLite database.

    Columns expected in the DataFrame:
        text, clean_text, vader_compound, vader_label,
        tb_polarity, tb_subjectivity, tb_label, sentiment

    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame from sentiment_analysis.analyse_sentiment().
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    int
        Number of rows successfully inserted.
    """

    # Make sure the table exists before inserting
    initialise_database(db_path)

    conn   = get_connection(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_inserted = 0

    insert_sql = f"""
    INSERT INTO {TABLE_NAME}
        (raw_text, clean_text, vader_compound, vader_label,
         tb_polarity, tb_subjectivity, tb_label, sentiment, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_sql, (
            str(row.get("text",            "")),
            str(row.get("clean_text",      "")),
            float(row.get("vader_compound",  0.0)),
            str(row.get("vader_label",     "Neutral")),
            float(row.get("tb_polarity",     0.0)),
            float(row.get("tb_subjectivity", 0.0)),
            str(row.get("tb_label",        "Neutral")),
            str(row.get("sentiment",       "Neutral")),
            timestamp,
        ))
        rows_inserted += 1

    conn.commit()
    conn.close()

    print(f"[database] {rows_inserted} rows saved to '{db_path}'.")
    return rows_inserted


# ------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------

def load_all_results(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Retrieve all rows from the database as a Pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        All stored sentiment records, or an empty DataFrame if none.
    """

    if not os.path.exists(db_path):
        print("[database] No database file found. Returning empty DataFrame.")
        return pd.DataFrame()

    conn = get_connection(db_path)
    df   = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    print(f"[database] Loaded {len(df)} rows from database.")
    return df


def get_sentiment_counts(db_path: str = DB_PATH) -> dict:
    """
    Return a count of each sentiment label stored in the database.

    Returns
    -------
    dict  e.g. {'Positive': 120, 'Negative': 45, 'Neutral': 35}
    """

    if not os.path.exists(db_path):
        return {"Positive": 0, "Negative": 0, "Neutral": 0}

    conn   = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT sentiment, COUNT(*) as count
        FROM {TABLE_NAME}
        GROUP BY sentiment
    """)

    rows   = cursor.fetchall()
    conn.close()

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for sentiment, count in rows:
        counts[sentiment] = count

    return counts


def get_recent_results(n: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Retrieve the most recently inserted rows.

    Parameters
    ----------
    n : int
        How many rows to return (most recent first).

    Returns
    -------
    pd.DataFrame
    """

    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = get_connection(db_path)
    df   = pd.read_sql_query(
        f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT {n}",
        conn,
    )
    conn.close()
    return df


def clear_database(db_path: str = DB_PATH) -> None:
    """
    Delete all rows from the table (useful during testing/demo).
    The table structure itself is preserved.
    """

    if not os.path.exists(db_path):
        print("[database] Nothing to clear — database does not exist.")
        return

    conn   = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {TABLE_NAME}")
    conn.commit()
    conn.close()
    print("[database] All records cleared from the database.")


# ------------------------------------------------------------------
# Quick test — run this file directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny mock DataFrame that looks like the pipeline output
    mock_data = pd.DataFrame({
        "text":            ["Great product!", "Terrible service.", "Just a normal day."],
        "clean_text":      ["great product",  "terrible service",  "normal day"],
        "vader_compound":  [0.6588, -0.4767,   0.0000],
        "vader_label":     ["Positive", "Negative", "Neutral"],
        "tb_polarity":     [0.8, -0.5, 0.0],
        "tb_subjectivity": [0.75, 1.0, 0.0],
        "tb_label":        ["Positive", "Negative", "Neutral"],
        "sentiment":       ["Positive", "Negative", "Neutral"],
    })

    # Save to database
    save_results(mock_data)

    # Read back and display
    all_results = load_all_results()
    print("\nAll records in database:")
    print(all_results[["id", "raw_text", "sentiment", "created_at"]])

    # Display counts
    print("\nSentiment counts:", get_sentiment_counts())
