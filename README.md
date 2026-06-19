# Real-Time Social Media Analytics for Sentiment and Trend Detection

**Course:** CSI-630 | Govt. Municipal Graduate College, Faisalabad  
**Tech Stack:** Python · Pandas · VADER · TextBlob · SQLite · Streamlit

---

## Project Overview

This system collects social-media data (from a local CSV dataset), cleans the
text, classifies each post as **Positive**, **Negative**, or **Neutral** using
two pre-trained NLP models (VADER and TextBlob), stores the results in a local
SQLite database, and displays live charts and statistics on an interactive
Streamlit dashboard.

---

## Folder Structure

```
social_media_analytics/
│
├── data/                      ← created automatically
│   ├── tweets.csv             ← your dataset (or auto-generated sample)
│   └── analytics.db           ← SQLite database (auto-created)
│
├── data_collection.py         ← Module 1: load dataset
├── preprocessing.py           ← Module 2: clean & tokenize text
├── sentiment_analysis.py      ← Module 3: VADER + TextBlob analysis
├── database.py                ← Module 4: SQLite storage
├── dashboard.py               ← Module 5: Streamlit dashboard
│
├── requirements.txt           ← Python dependencies
└── README.md                  ← this file
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your own dataset
Place a CSV file with a `text` column at `data/tweets.csv`.  
A free Twitter sentiment dataset is available at:  
https://www.kaggle.com/datasets/kazanova/sentiment140

If no file is present, the dashboard will offer to generate a **200-row sample dataset** automatically.

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```

Your browser will open at `http://localhost:8501`.

### 4. Run the pipeline
Click **▶ Run Full Analysis Pipeline** in the left sidebar to:
- Load the dataset
- Clean and tokenize the text
- Analyse sentiment with VADER + TextBlob
- Save results to SQLite
- Refresh all charts automatically

---

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `data_collection.py` | Reads the CSV dataset into a Pandas DataFrame; generates sample data if none exists |
| `preprocessing.py` | Lowercases, removes URLs/mentions/emojis/hashtags/stop-words, and tokenizes each post |
| `sentiment_analysis.py` | Scores each post with VADER and TextBlob, then combines scores into a final ensemble label |
| `database.py` | Creates and manages a SQLite database; inserts and queries sentiment records |
| `dashboard.py` | Streamlit web app with pie/bar/trend charts, a data table, and a live single-post analyser |

---

## Testing Individual Modules

Each module can be run standalone:

```bash
python data_collection.py
python preprocessing.py
python sentiment_analysis.py
python database.py
```

---

## Pipeline Architecture

```
CSV Dataset
    │
    ▼
data_collection.py  ──►  Raw DataFrame
    │
    ▼
preprocessing.py    ──►  Cleaned Text Column
    │
    ▼
sentiment_analysis.py ►  Sentiment Labels + Scores
    │
    ▼
database.py         ──►  SQLite (analytics.db)
    │
    ▼
dashboard.py        ──►  Streamlit Web Dashboard
```
