# Real-Time Social Media Analytics for Sentiment and Trend Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20TextBlob-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **Final Year Project** — BS Computer Science, Govt. Municipal Graduate College, Faisalabad (2026)

A modular, end-to-end NLP pipeline that collects social media text, runs dual-engine sentiment analysis (VADER + TextBlob), persists results in SQLite, and displays live insights on an interactive Streamlit dashboard — all with zero API keys required.

---

## Features

- **Dual-engine sentiment analysis** — VADER and TextBlob scores combined into an ensemble label (Positive / Negative / Neutral)
- **Full NLP preprocessing** — URL removal, mention stripping, stopword filtering, lemmatization, tokenization
- **Interactive Streamlit dashboard** — pie charts, bar charts, trend lines, word cloud, hashtag frequency, live single-post analyser
- **Live Reddit data fetching** — via RSS feed (no API key needed)
- **SQLite persistence** — all results stored locally, queryable across sessions
- **Comparison module** — benchmark VADER vs TextBlob accuracy side by side
- **ML model training** — Logistic Regression and Random Forest classifiers trained on processed data
- **Auto sample generation** — works out of the box with no dataset provided

---

## Screenshots

> Dashboard preview — run locally with `streamlit run App.py`

*(Add a screenshot here: Dashboard → take screenshot → save as `assets/dashboard.png` → update this line)*

---

## Project Structure

```
social_media_analytics/
│
├── data/                       ← auto-created at runtime
│   ├── tweets.csv              ← your dataset (or auto-generated sample)
│   └── analytics.db            ← SQLite database (auto-created)
│
├── models/                     ← saved ML model files
│
├── App.py                      ← main entry point (Streamlit app)
├── dashboard.py                ← dashboard UI components
├── comments_analyser.py        ← Reddit RSS comment fetcher & analyser
├── live_reddit_fetcher.py      ← live Reddit feed via RSS (no API key)
├── sentiment_analysis.py       ← VADER + TextBlob dual engine
├── preprocessing.py            ← full NLP preprocessing pipeline
├── data_collection.py          ← dataset loader + sample generator
├── database.py                 ← SQLite CRUD operations
├── ml_models.py                ← Logistic Regression + Random Forest
├── train_models.py             ← model training script
├── comparison.py               ← VADER vs TextBlob benchmarking
├── hashtag_trends.py           ← hashtag extraction and frequency analysis
├── wordcloud_gen.py            ← word cloud generation
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run App.py
```

Opens at `http://localhost:8501` — no configuration needed.

### 4. (Optional) Add your own dataset
Place a CSV with a `text` column at `data/tweets.csv`.  
Free dataset: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Pipeline Architecture

```
CSV Dataset / Reddit RSS Feed
          │
          ▼
  data_collection.py  ──►  Raw DataFrame
          │
          ▼
  preprocessing.py    ──►  Cleaned Text
          │
          ▼
  sentiment_analysis.py ►  VADER + TextBlob Labels
          │
          ▼
  ml_models.py        ──►  LR / RF Classification
          │
          ▼
  database.py         ──►  SQLite Storage
          │
          ▼
  dashboard.py / App.py ►  Streamlit Dashboard
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `App.py` | Main Streamlit entry point |
| `dashboard.py` | Charts, filters, data table UI |
| `comments_analyser.py` | Fetch and analyse Reddit comments via RSS |
| `live_reddit_fetcher.py` | Real-time Reddit post fetching (RSS, no API key) |
| `sentiment_analysis.py` | Dual-engine VADER + TextBlob scorer |
| `preprocessing.py` | Full NLP preprocessing pipeline |
| `data_collection.py` | CSV loader + auto sample generator |
| `database.py` | SQLite schema, insert, and query |
| `ml_models.py` | Logistic Regression + Random Forest |
| `train_models.py` | Standalone model training |
| `comparison.py` | VADER vs TextBlob accuracy comparison |
| `hashtag_trends.py` | Hashtag extraction and trend ranking |
| `wordcloud_gen.py` | Word cloud generation from corpus |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.9+ |
| NLP | VADER, TextBlob, NLTK |
| ML | Scikit-learn (Logistic Regression, Random Forest) |
| Data | Pandas, NumPy |
| Visualisation | Streamlit, Matplotlib, Seaborn, WordCloud |
| Database | SQLite3 |
| Data Source | Reddit RSS (no API key required) |

---

## Author

**Muhammad Zarghan**  
BS Computer Science — Govt. Municipal Graduate College, Faisalabad (2026)  
[LinkedIn](https://linkedin.com/in/muhammad-zarghan) · [Fiverr](https://www.fiverr.com/m_zarghan)
