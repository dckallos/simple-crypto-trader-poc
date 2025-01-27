#!/usr/bin/env python3
# =============================================================================
# FILE: train_model.py
# =============================================================================
"""
train_model.py

Demonstrates three enhancements to reduce GPT prompt size:

1) **Preprocessing & Summaries**:
   - We gather aggregator data (e.g. from lunarcrush_data + cryptopanic_posts),
   - produce short textual or numeric summaries,
   - and store them into a new table (or the same table) for quick retrieval.

2) **Local Classifier/Regressor**:
   - We use scikit-learn to produce a probability of "UP" vs. "DOWN",
   - store or output that probability so GPT sees only "prob_up=0.72" and doesn't need raw aggregator data.

3) **Embedding-Based Summaries**:
   - We embed aggregator data with a local or external embedding model,
   - possibly reduce dimensionality,
   - store or output the small vector to pass in GPT prompts.

Usage:
    python train_model.py --summaries
    python train_model.py --classifier
    python train_model.py --embeddings

You can run them in sequence or pick/choose. The script stores final results in new columns/tables
like `aggregator_summaries`, `aggregator_classifier_probs`, `aggregator_embeddings`, or
whatever naming you prefer.

Important:
    - Adjust the DB_FILE path in `db.py` if needed.
    - You must run `init_db()` from `db.py` to ensure tables exist (and possibly
      create new tables for aggregator_summaries if you prefer).
    - For embeddings, you might need extra packages (e.g. sentence-transformers, spacy, or openai embeddings).
"""

import os
import sys
import argparse
import logging
import sqlite3
import time
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# We assume you have a 'db.py' with DB_FILE, init_db, etc.
# from db import DB_FILE, init_db
from db import DB_FILE, init_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


########################################
# Section 1: Load aggregator data
########################################
def load_lunarcrush_data() -> pd.DataFrame:
    """
    Loads aggregator data from 'lunarcrush_data'.
    We'll create a DataFrame with columns we want to use for local classification or summarization.

    Return:
        pd.DataFrame with columns:
          [id, timestamp, symbol, name, price, market_cap, volume_24h, volatility, galaxy_score, alt_rank, sentiment]
        ... plus any other columns you find relevant.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        q = """
            SELECT
                id, timestamp, symbol, name, price, market_cap, volume_24h, 
                volatility, galaxy_score, alt_rank, sentiment
            FROM lunarcrush_data
        """
        df = pd.read_sql_query(q, conn)
        logger.info(f"Loaded {len(df)} rows from lunarcrush_data.")
        return df
    except Exception as e:
        logger.exception(f"Error loading lunarcrush_data => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_cryptopanic_data() -> pd.DataFrame:
    """
    Loads cryptopanic_posts, if relevant. We'll do a naive aggregator approach:
      e.g. average sentiment by day or by symbol references.
    Here, we'll just return the entire table for demonstration.

    Return:
        pd.DataFrame with columns [id, title, url, sentiment_score, created_at, ...].
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        q = "SELECT id, title, url, sentiment_score, created_at FROM cryptopanic_posts"
        df = pd.read_sql_query(q, conn)
        logger.info(f"Loaded {len(df)} rows from cryptopanic_posts.")
        return df
    except Exception as e:
        logger.exception(f"Error loading cryptopanic_posts => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


########################################
# Section 2: Summaries
########################################
def create_aggregator_summaries(df_lc: pd.DataFrame, df_cp: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summarized aggregator DataFrame. For example:
      - 'price_bucket', 'sentiment_label', 'volatility_label', ...
      - We might combine data from cryptopanic (daily avg sentiment) to form a single summary row.

    Return:
        A new df_summaries with minimal textual or numeric columns to pass to GPT.
    """
    if df_lc.empty:
        logger.warning("No lunarcrush data => skipping aggregator_summaries.")
        return pd.DataFrame()

    # Suppose we define some bins for price
    def bucket_price(p):
        if p < 1000: return "low"
        elif p < 5000: return "medium"
        else: return "high"

    # Example sentiment label
    def label_sentiment(s):
        if s is None or np.isnan(s):
            return "unknown"
        if s > 0.6:
            return "strong_positive"
        elif s > 0.3:
            return "slightly_positive"
        elif s > 0.0:
            return "neutral"
        else:
            return "negative"

    # We'll build a new DF with 'symbol','price_bucket','galaxy_score','alt_rank','sentiment_label'
    summary_rows = []
    for idx, row in df_lc.iterrows():
        price_cat = bucket_price(row["price"])
        sentiment_cat = label_sentiment(row["sentiment"])
        summary_rows.append({
            "id": row["id"],
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "price_bucket": price_cat,
            "galaxy_score": row["galaxy_score"],
            "alt_rank": row["alt_rank"],
            "sentiment_label": sentiment_cat
        })

    df_summary = pd.DataFrame(summary_rows)
    logger.info(f"Created aggregator summaries => {len(df_summary)} rows.")
    return df_summary


def store_aggregator_summaries(df_summary: pd.DataFrame):
    """
    Persists aggregator summaries into a new table 'aggregator_summaries', for example.
    Adjust columns as needed. You can store them in existing tables if you prefer.
    """
    if df_summary.empty:
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        # create table if not exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_summaries (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp INTEGER,
                price_bucket TEXT,
                galaxy_score REAL,
                alt_rank REAL,
                sentiment_label TEXT
            )
        """)
        conn.commit()

        # now insert each row
        inserts = []
        for idx, r in df_summary.iterrows():
            inserts.append((
                r["id"],
                r["symbol"],
                r["timestamp"],
                r["price_bucket"],
                r["galaxy_score"],
                r["alt_rank"],
                r["sentiment_label"]
            ))

        c.executemany("""
            INSERT OR REPLACE INTO aggregator_summaries
            (id, symbol, timestamp, price_bucket, galaxy_score, alt_rank, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, inserts)
        conn.commit()
        logger.info(f"Stored {len(df_summary)} aggregator summaries.")
    except Exception as e:
        logger.exception(f"Error storing aggregator_summaries => {e}")
    finally:
        conn.close()


########################################
# Section 3: Local Classifier => Probability
########################################
def build_local_classifier(df_lc: pd.DataFrame) -> None:
    """
    Example pipeline: we create a classification label (label_up=1 if price rose next day),
    then train a classifier to produce prob_up. We store that model to disk or in memory.

    We'll pretend df_lc has enough data to see future price changes.
    Real code might reference 'lunarcrush_timeseries' or other historical data.

    For demonstration, we'll do a naive approach:
      label_up => 1 if galaxy_score > 50, else 0
    Then we fit a random forest => predict prob_up.

    Finally, we store the predicted probabilities in 'aggregator_classifier_probs'.
    """
    if df_lc.empty:
        logger.warning("No data => skipping local classifier training.")
        return

    df_lc = df_lc.copy()
    df_lc["label_up"] = (df_lc["galaxy_score"] > 50).astype(int)

    # We'll pick features
    feats = ["price", "market_cap", "volume_24h", "galaxy_score", "alt_rank"]
    # drop rows with missing
    df_lc.dropna(subset=feats + ["label_up"], inplace=True)

    if len(df_lc) < 10:
        logger.warning("Not enough data to train local classifier => skipping.")
        return

    X = df_lc[feats].values
    y = df_lc["label_up"].values

    # Train
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # We'll generate predicted probabilities (for demonstration).
    prob_up = rf.predict_proba(X)[:,1]

    # Store them in aggregator_classifier_probs
    df_probs = pd.DataFrame({
        "id": df_lc["id"],
        "symbol": df_lc["symbol"],
        "timestamp": df_lc["timestamp"],
        "prob_up": prob_up
    })
    store_classifier_probs_table(df_probs)

    # Optionally store the model to disk => trained_model.pkl
    import joblib
    joblib.dump(rf, "trained_classifier.pkl")
    logger.info("Local classifier trained => saved to trained_classifier.pkl")


def store_classifier_probs_table(df_probs: pd.DataFrame):
    """
    Creates aggregator_classifier_probs table => store each row => ID references same row in lunarcrush_data
    """
    if df_probs.empty:
        logger.warning("No classifier probabilities => skip storing.")
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_classifier_probs (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp INTEGER,
                prob_up REAL
            )
        """)
        conn.commit()

        inserts = []
        for idx, r in df_probs.iterrows():
            inserts.append((r["id"], r["symbol"], r["timestamp"], r["prob_up"]))

        c.executemany("""
            INSERT OR REPLACE INTO aggregator_classifier_probs
            (id, symbol, timestamp, prob_up)
            VALUES (?, ?, ?, ?)
        """, inserts)
        conn.commit()
        logger.info(f"Inserted {len(df_probs)} classifier probability rows.")
    except Exception as e:
        logger.exception(f"Error storing aggregator_classifier_probs => {e}")
    finally:
        conn.close()


########################################
# Section 4: Embedding-Based Summaries
########################################
def build_embedding_summaries(df_lc: pd.DataFrame, n_components=3):
    """
    Suppose we embed aggregator data with a local or external model. For demonstration,
    we'll just do a basic numeric approach => use columns as is, then apply PCA to reduce
    dimension to e.g. 3. We call that the 'embedding vector'. In real usage, you'd do
    sentence-transformers or openai embeddings if aggregator_data is textual.

    We'll store those 3 principal components in aggregator_embeddings table => (comp1,comp2,comp3).
    """
    if df_lc.empty:
        logger.warning("No data => skipping embeddings.")
        return

    feats = ["price", "market_cap", "volume_24h", "volatility", "galaxy_score", "alt_rank", "sentiment"]
    df_lc = df_lc.dropna(subset=feats).copy()
    if len(df_lc) < n_components:
        logger.warning("Not enough data to do PCA => skipping.")
        return

    X = df_lc[feats].values
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    # Build results
    df_emb = pd.DataFrame(X_reduced, columns=[f"comp{i+1}" for i in range(n_components)])
    df_emb["id"] = df_lc["id"].values
    df_emb["symbol"] = df_lc["symbol"].values
    df_emb["timestamp"] = df_lc["timestamp"].values

    store_embedding_vectors(df_emb)


def store_embedding_vectors(df_emb: pd.DataFrame):
    """
    Persists embedding-based summaries in aggregator_embeddings table => columns
     id, symbol, timestamp, comp1, comp2, comp3 (or up to N)
    """
    if df_emb.empty:
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        # build a table with flexible columns if you like. We'll do comp1..comp3 for now.
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_embeddings (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp INTEGER,
                comp1 REAL,
                comp2 REAL,
                comp3 REAL
            )
        """)
        conn.commit()

        inserts = []
        for idx, r in df_emb.iterrows():
            inserts.append((r["id"], r["symbol"], r["timestamp"], r["comp1"], r["comp2"], r["comp3"]))

        c.executemany("""
            INSERT OR REPLACE INTO aggregator_embeddings
            (id, symbol, timestamp, comp1, comp2, comp3)
            VALUES (?, ?, ?, ?, ?, ?)
        """, inserts)
        conn.commit()
        logger.info(f"Stored {len(df_emb)} aggregator embedding vectors.")
    except Exception as e:
        logger.exception(f"Error storing aggregator_embeddings => {e}")
    finally:
        conn.close()


########################################
# Main CLI logic
########################################
def main():
    parser = argparse.ArgumentParser(description="Training & aggregator preprocessing script.")
    parser.add_argument("--summaries", action="store_true", help="Generate aggregator summaries.")
    parser.add_argument("--classifier", action="store_true", help="Train local classifier => store probability.")
    parser.add_argument("--embeddings", action="store_true", help="Build PCA-based aggregator embeddings.")
    args = parser.parse_args()

    init_db()  # ensure baseline tables exist (trades, price_history, etc.). We'll also create new aggregator_* tables on the fly.

    # load aggregator data
    df_lc = load_lunarcrush_data()
    df_cp = load_cryptopanic_data()

    # Summaries
    if args.summaries:
        df_summary = create_aggregator_summaries(df_lc, df_cp)
        store_aggregator_summaries(df_summary)

    # Classifier
    if args.classifier:
        build_local_classifier(df_lc)

    # Embeddings
    if args.embeddings:
        build_embedding_summaries(df_lc, n_components=3)


if __name__ == "__main__":
    main()
