#!/usr/bin/env python3
# =============================================================================
# FILE: train_model.py
# =============================================================================
"""
train_model.py

Unifies aggregator data management + a time-series-based scikit-learn training pipeline,
using the updated 'lunarcrush_data' and 'lunarcrush_timeseries' (with new columns
introduced by fetch_lunarcrush.py), plus aggregator tables:

- aggregator_summaries
- aggregator_classifier_probs
- aggregator_embeddings

Additionally:
- Optionally uses 'cryptopanic_posts' data for aggregator merging
- Produces a final scikit-learn model (trained_model.pkl) & metrics (model_info.json)

CLI Flags:
    --summaries => gather aggregator summaries => aggregator_summaries
    --classifier => build local classifier => aggregator_classifier_probs
    --embeddings => do PCA-based embeddings => aggregator_embeddings
    --train => final time-series training => produce 'trained_model.pkl'
    --schedule => runs aggregator + training in intervals (cron-like)
    --interval => seconds for schedule, default=3600

Example:
    python train_model.py --summaries --classifier --embeddings --train

The aggregator pipeline:
  1) load aggregator data from 'lunarcrush_data' & 'cryptopanic_posts'
  2) create aggregator_summaries => aggregator_summaries
  3) local classifier => aggregator_classifier_probs
  4) embeddings => aggregator_embeddings
  5) final time-series training => SHIFT_BARS => label => train random forest => store

Requires:
  - scikit-learn, numpy, pandas
  - joblib for model saving
  - A DB with the extended 'lunarcrush_data' & 'lunarcrush_timeseries' columns,
    as updated by fetch_lunarcrush.py.

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
import joblib
import warnings
from typing import Tuple, Dict, Optional, List
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

###############################################################################
# Global Constants & Configuration
###############################################################################
SHIFT_BARS = 24  # how many bars ahead we label (time shift)
THRESHOLD_UP = 0.02  # e.g. +2% => label_up
MIN_DATA_ROWS = 100  # minimal rows needed to train
MODEL_OUTPUT_PATH = "trained_model.pkl"
MODEL_INFO_FILE = "model_info.json"
DB_FILE = "trades.db"  # Must match your main DB path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################
# SECTION 1: aggregator management => Summaries, Classifier, Embeddings
###############################################################################
def load_lunarcrush_aggregator_data() -> pd.DataFrame:
    """
    Loads aggregator snapshot from 'lunarcrush_data', including new columns
    from fetch_lunarcrush.py (price_btc, galaxy_score_previous, alt_rank_previous, etc.).

    Returns:
        pd.DataFrame: The aggregator snapshot with relevant columns for further steps.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        # Adjust columns as needed; this example includes some new ones:
        query = """
        SELECT
            id,
            symbol,
            timestamp,
            price,
            price_btc,
            market_cap,
            volume_24h,
            volatility,
            galaxy_score,
            galaxy_score_previous,
            alt_rank,
            alt_rank_previous,
            market_dominance,
            sentiment,
            circulating_supply,
            max_supply
        FROM lunarcrush_data
        """
        df = pd.read_sql_query(query, conn)
        logger.info(f"[Aggregator] loaded {len(df)} rows from lunarcrush_data (with new columns).")
        return df
    except Exception as e:
        logger.exception(f"[Aggregator] error loading lunarcrush_data => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_cryptopanic_aggregator_data() -> pd.DataFrame:
    """
    Example aggregator from 'cryptopanic_posts'.
    For demonstration, we just load all rows. You can refine for daily avg merges, etc.

    Returns:
        pd.DataFrame: The cryptopanic posts with columns [id, sentiment_score, created_at].
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        query = """
            SELECT id, sentiment_score, created_at
            FROM cryptopanic_posts
        """
        df = pd.read_sql_query(query, conn)
        logger.info(f"[Aggregator] loaded {len(df)} cryptopanic_posts.")
        return df
    except Exception as e:
        logger.exception(f"[Aggregator] cryptopanic => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def create_aggregator_summaries(df_lc: pd.DataFrame, df_cp: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregator summaries from the expanded 'lunarcrush_data'.
    Optionally merges cryptopanic or any other aggregator data if desired.

    New columns included:
      - galaxy_score_previous
      - alt_rank_previous
      - market_dominance => can be bucketed or used raw

    Args:
        df_lc (pd.DataFrame): DataFrame with lunarcrush_data aggregator snapshot
        df_cp (pd.DataFrame): cryptopanic DataFrame (not extensively used here)

    Returns:
        pd.DataFrame: aggregator_summaries to store in aggregator_summaries table
    """
    if df_lc.empty:
        logger.warning("[Summaries] => No lunarcrush_data => skipping aggregator summaries.")
        return pd.DataFrame()

    def bucket_price(p: float) -> str:
        if p < 1000:
            return "low"
        elif p < 5000:
            return "medium"
        else:
            return "high"

    def label_sentiment(s: float) -> str:
        if s is None or np.isnan(s):
            return "unknown"
        if s > 0.6:
            return "strong_pos"
        elif s > 0.3:
            return "slightly_pos"
        elif s > 0.0:
            return "neutral"
        else:
            return "negative"

    rows = []
    for _, row in df_lc.iterrows():
        price_val = row["price"] if not pd.isnull(row["price"]) else 0.0
        pcat = bucket_price(price_val)

        sentiment_val = row.get("sentiment", 0.0)
        scat = label_sentiment(sentiment_val if not pd.isnull(sentiment_val) else 0.0)

        # Example for market_dominance => create a "dominance_bucket"
        md = row.get("market_dominance", 0.0)
        if md > 0.05:
            dominance_bucket = "dominant"
        elif md > 0.01:
            dominance_bucket = "moderate"
        else:
            dominance_bucket = "low"

        rows.append({
            "id": row["id"],
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "price_bucket": pcat,
            "galaxy_score": row.get("galaxy_score", 0.0),
            "galaxy_score_previous": row.get("galaxy_score_previous", 0.0),
            "alt_rank": row.get("alt_rank", 999999),
            "alt_rank_previous": row.get("alt_rank_previous", 999999),
            "market_dominance": md,
            "dominance_bucket": dominance_bucket,
            "sentiment_label": scat
        })

    df_sum = pd.DataFrame(rows)
    logger.info(f"[Summaries] => built {len(df_sum)} aggregator summaries with new columns.")
    return df_sum


def store_aggregator_summaries(df_summary: pd.DataFrame):
    """
    Store aggregator summaries (with new columns) into aggregator_summaries table.
    Upserts on the 'id' primary key.
    """
    if df_summary.empty:
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        # Extended schema for aggregator_summaries
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_summaries (
                id INTEGER PRIMARY KEY,  -- matches lunarcrush_data.id
                symbol TEXT,
                timestamp INTEGER,
                price_bucket TEXT,
                galaxy_score REAL,
                galaxy_score_previous REAL,
                alt_rank REAL,
                alt_rank_previous REAL,
                market_dominance REAL,
                dominance_bucket TEXT,
                sentiment_label TEXT
            )
        """)
        conn.commit()

        rows_to_insert = []
        for _, r in df_summary.iterrows():
            rows_to_insert.append((
                r["id"],
                r["symbol"],
                r["timestamp"],
                r["price_bucket"],
                r["galaxy_score"],
                r["galaxy_score_previous"],
                r["alt_rank"],
                r["alt_rank_previous"],
                r["market_dominance"],
                r["dominance_bucket"],
                r["sentiment_label"]
            ))

        c.executemany(
            """
            INSERT OR REPLACE INTO aggregator_summaries (
                id, symbol, timestamp, price_bucket,
                galaxy_score, galaxy_score_previous,
                alt_rank, alt_rank_previous,
                market_dominance, dominance_bucket,
                sentiment_label
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows_to_insert
        )
        conn.commit()
        logger.info(f"[Summaries] => stored {len(rows_to_insert)} aggregator_summaries rows.")
    except Exception as e:
        logger.exception(f"[Summaries] => store error => {e}")
    finally:
        conn.close()


def build_local_classifier(df_lc: pd.DataFrame):
    """
    Creates a naive classifier => label_up=1 if galaxy_score>50 => produce prob_up => aggregator_classifier_probs
    Updated to incorporate new columns (price_btc, market_dominance, etc.) in the feature set if desired.
    """
    if df_lc.empty:
        logger.warning("[Classifier] => no lunarcrush_data => skip building local classifier.")
        return

    df_lc = df_lc.copy()

    # Label: galaxy_score>50 => up
    df_lc["label_up"] = (df_lc["galaxy_score"] > 50).astype(int)

    # Example expanded set of features
    feats = [
        "price", "price_btc", "market_cap", "volume_24h",
        "galaxy_score", "galaxy_score_previous",
        "market_dominance", "alt_rank"
    ]
    feats_exist = [f for f in feats if f in df_lc.columns]

    # drop rows missing any feats or label
    df_lc.dropna(subset=feats_exist + ["label_up"], inplace=True)
    if len(df_lc) < 10:
        logger.warning("[Classifier] => not enough rows => skip.")
        return

    from sklearn.ensemble import RandomForestClassifier
    X = df_lc[feats_exist].values
    y = df_lc["label_up"].values

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    prob_up = rf.predict_proba(X)[:, 1]
    df_probs = pd.DataFrame({
        "id": df_lc["id"],
        "symbol": df_lc["symbol"],
        "timestamp": df_lc["timestamp"],
        "prob_up": prob_up
    })
    store_classifier_probs_table(df_probs)

    import joblib
    joblib.dump(rf, "local_classifier.pkl")
    logger.info("[Classifier] => local_classifier.pkl saved.")


def store_classifier_probs_table(df_probs: pd.DataFrame):
    """
    Insert or replace aggregator_classifier_probs with prob_up for each row.
    """
    if df_probs.empty:
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

        data_insert = []
        for _, row in df_probs.iterrows():
            data_insert.append((row["id"], row["symbol"], row["timestamp"], row["prob_up"]))

        c.executemany(
            """
            INSERT OR REPLACE INTO aggregator_classifier_probs
            (id, symbol, timestamp, prob_up)
            VALUES (?,?,?,?)
            """,
            data_insert
        )
        conn.commit()
        logger.info(f"[Classifier] => stored {len(data_insert)} aggregator_classifier_probs rows.")
    except Exception as e:
        logger.exception(f"[Classifier] => store => {e}")
    finally:
        conn.close()


def build_embedding_summaries(df_lc: pd.DataFrame, n_components=3):
    """
    Create PCA-based embeddings from numeric columns, including new columns from fetch_lunarcrush.py
    if you want them to factor into embeddings.
    """
    if df_lc.empty:
        logger.warning("[Embeddings] => no lunarcrush_data => skip.")
        return

    # Potential columns to include in embeddings
    feats = [
        "price", "price_btc", "market_cap", "volume_24h", "volatility",
        "galaxy_score", "galaxy_score_previous",
        "alt_rank", "market_dominance", "sentiment"
    ]
    feats_exist = [f for f in feats if f in df_lc.columns]

    df_lc = df_lc.dropna(subset=feats_exist).copy()
    if len(df_lc) < n_components:
        logger.warning("[Embeddings] => not enough data => skip.")
        return

    from sklearn.decomposition import PCA
    X = df_lc[feats_exist].values
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    df_emb = pd.DataFrame(
        X_reduced,
        columns=[f"comp{i + 1}" for i in range(n_components)]
    )
    df_emb["id"] = df_lc["id"].values
    df_emb["symbol"] = df_lc["symbol"].values
    df_emb["timestamp"] = df_lc["timestamp"].values

    store_embedding_vectors(df_emb)


def store_embedding_vectors(df_emb: pd.DataFrame):
    """
    Persist embeddings in aggregator_embeddings table.
    If you prefer more than 3 PCA components, extend the schema accordingly.
    """
    if df_emb.empty:
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
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
        for _, row in df_emb.iterrows():
            inserts.append((
                row["id"], row["symbol"], row["timestamp"],
                row["comp1"], row["comp2"], row["comp3"]
            ))

        c.executemany(
            """
            INSERT OR REPLACE INTO aggregator_embeddings
            (id, symbol, timestamp, comp1, comp2, comp3)
            VALUES (?,?,?,?,?,?)
            """,
            inserts
        )
        conn.commit()
        logger.info(f"[Embeddings] => stored {len(inserts)} aggregator embeddings.")
    except Exception as e:
        logger.exception(f"[Embeddings] => store => {e}")
    finally:
        conn.close()


###############################################################################
# SECTION 2: Final time-series-based training
###############################################################################
def load_time_series_for_symbol(symbol: str, db_file=DB_FILE, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Loads time-series data for 'symbol' from 'lunarcrush_timeseries' with columns:
      - open_price, close_price, high_price, low_price, volume_24h, market_cap,
        sentiment, galaxy_score, alt_rank, market_dominance, circulating_supply, etc.

    We assume coin_id = UPPER(symbol).
    If limit is provided, read the last 'limit' rows (by timestamp desc).

    Returns:
        pd.DataFrame: time-series rows sorted ascending by timestamp
    """
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM lunarcrush_timeseries WHERE coin_id=?", (symbol.upper(),))
        rowcount = c.fetchone()[0] or 0
        if rowcount < 1:
            logger.warning(f"[TimeSeries] => no rows for symbol={symbol} in lunarcrush_timeseries")
            return pd.DataFrame()

        # build dynamic query
        if limit and rowcount > limit:
            logger.info(f"[TimeSeries] => symbol={symbol} => rowcount={rowcount} => limiting to last {limit} rows.")
            q = f"""
                SELECT *
                FROM (
                  SELECT
                    coin_id, timestamp,
                    open_price, close_price, high_price, low_price,
                    volume_24h, market_cap, sentiment, galaxy_score, alt_rank,
                    market_dominance, circulating_supply
                  FROM lunarcrush_timeseries
                  WHERE coin_id=?
                  ORDER BY timestamp DESC
                  LIMIT {limit}
                ) sub
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(q, conn, params=[symbol.upper()])
        else:
            q = """
                SELECT
                  coin_id, timestamp,
                  open_price, close_price, high_price, low_price,
                  volume_24h, market_cap, sentiment, galaxy_score, alt_rank,
                  market_dominance, circulating_supply
                FROM lunarcrush_timeseries
                WHERE coin_id=?
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(q, conn, params=[symbol.upper()])

        return df

    except Exception as e:
        logger.exception(f"[TimeSeries] => symbol={symbol}, error => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def build_training_features_time_series(symbol: str, df_cp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    1) Load timeseries => SHIFT_BARS => label future price jump.
    2) Optionally incorporate aggregator or cryptopanic merges.
    3) Return (X, y) for modeling.

    For demonstration, we do a basic approach:
      - SHIFT_BARS=24 => label if price is +2% from now => label_up=1
      - We do a few simple rolling features: ma_5, std_10
      - We also keep galaxy_score & sentiment from timeseries.

    Args:
        symbol (str): e.g. "ETH", "XBT"
        df_cp (pd.DataFrame): cryptopanic aggregator, not heavily used here

    Returns:
        (X, y): The feature matrix and target labels as pd.Series
    """
    df_ts = load_time_series_for_symbol(symbol)
    if df_ts.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Basic cleaning
    df_ts["datetime"] = pd.to_datetime(df_ts["timestamp"], unit="s")
    # fill close_price if missing
    df_ts["close_price"] = df_ts["close_price"].fillna(df_ts["open_price"])
    df_ts.sort_values("timestamp", inplace=True)
    df_ts.reset_index(drop=True, inplace=True)

    # SHIFT_BARS => we label "future_close" = close_price SHIFT_BARS ahead
    df_ts["future_close"] = df_ts["close_price"].shift(-SHIFT_BARS)
    df_ts["pct_future_gain"] = (
            (df_ts["future_close"] - df_ts["close_price"]) / (df_ts["close_price"] + 1e-9)
    )
    # label_up = 1 if >= THRESHOLD_UP
    df_ts["label_up"] = (df_ts["pct_future_gain"] >= THRESHOLD_UP).astype(int)
    df_ts.dropna(subset=["future_close"], inplace=True)

    # Rolling features
    df_ts["ma_5"] = df_ts["close_price"].rolling(5).mean()
    df_ts["std_10"] = df_ts["close_price"].rolling(10).std()
    df_ts["ma_5"].fillna(method="bfill", inplace=True)
    df_ts["std_10"].fillna(method="bfill", inplace=True)

    # Additional features from timeseries if desired: 'galaxy_score', 'sentiment'
    feats = [
        "close_price",
        "ma_5",
        "std_10",
        "galaxy_score",
        "sentiment"
    ]
    # drop rows missing these features
    df_ts.dropna(subset=feats, inplace=True)

    X = df_ts[feats].copy()
    y = df_ts["label_up"].copy()

    return X, y


def train_and_backtest(X: pd.DataFrame, y: pd.Series):
    """
    Train a random forest using time-series CV (TimeSeriesSplit),
    then do a small hold-out test => store the final model & best params.

    Returns:
        A fitted scikit-learn model (RandomForestClassifier) or None if not enough data.
    """
    if X.empty or len(X) < 10:
        logger.warning("[Train] => insufficient data => skip training.")
        return None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import f1_score, accuracy_score

    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10]
    }
    base = RandomForestClassifier(random_state=42)
    g = GridSearchCV(base, param_grid, scoring="f1_macro", cv=tscv, n_jobs=-1)
    g.fit(X, y)

    best = g.best_estimator_
    logger.info(f"[Train] => best_params={g.best_params_}, best CV f1={g.best_score_:.4f}")

    # do a final hold-out => last 20% for test
    hold_size = max(1, int(len(X) * 0.2))
    X_train = X.iloc[:-hold_size]
    y_train = y.iloc[:-hold_size]
    X_test = X.iloc[-hold_size:]
    y_test = y.iloc[-hold_size:]

    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    f1m = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"[Train] => hold-out => f1={f1m:.4f}, acc={acc:.4f}")

    joblib.dump(best, MODEL_OUTPUT_PATH)
    logger.info(f"[Train] => model saved => {MODEL_OUTPUT_PATH}")

    info = {"f1": f1m, "acc": acc, "params": g.best_params_}
    with open(MODEL_INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"[Train] => saved model_info => {MODEL_INFO_FILE}")

    return best


###############################################################################
# SECTION 3: Scheduling approach
###############################################################################
def schedule_training(
        interval_seconds: int = 3600,
        do_summaries: bool = True,
        do_classifier: bool = True,
        do_embeddings: bool = True,
        do_train: bool = True
):
    """
    Illustrates a repeated aggregator & training pipeline every interval_seconds.

    Steps each run:
      1) Summaries => aggregator_summaries
      2) Classifier => aggregator_classifier_probs
      3) Embeddings => aggregator_embeddings
      4) final training => produce trained_model.pkl

    Adjust or remove aggregator steps if not needed.
    """
    logger.info(f"[Scheduler] => start aggregator + training every {interval_seconds}s.")
    next_run = time.time()

    while True:
        now = time.time()
        if now >= next_run:
            logger.info("[Scheduler] => Running aggregator + training pipeline...")

            # aggregator steps
            if do_summaries or do_classifier or do_embeddings:
                df_lc = load_lunarcrush_aggregator_data()
                df_cp = load_cryptopanic_aggregator_data()

                if do_summaries:
                    df_summ = create_aggregator_summaries(df_lc, df_cp)
                    store_aggregator_summaries(df_summ)

                if do_classifier:
                    build_local_classifier(df_lc)

                if do_embeddings:
                    build_embedding_summaries(df_lc)

            # final training
            if do_train:
                # example: just train on a small set
                # or parse from config.yaml
                symbols = ["ETH", "XBT", "XRP"]  # demonstration
                all_X, all_y = [], []
                df_cp = load_cryptopanic_aggregator_data()  # if merging needed
                for s in symbols:
                    Xs, ys = build_training_features_time_series(s, df_cp)
                    if len(Xs) > MIN_DATA_ROWS:
                        all_X.append(Xs)
                        all_y.append(ys)
                    else:
                        logger.warning(f"[Train] => symbol={s} => insufficient rows => skip")

                if all_X:
                    Xfull = pd.concat(all_X, ignore_index=True)
                    yfull = pd.concat(all_y, ignore_index=True)
                    logger.info(f"[Scheduler] => final dataset => X={Xfull.shape}, y={yfull.shape}")
                    model = train_and_backtest(Xfull, yfull)
                else:
                    logger.warning("[Train] => no data => skip training.")

            next_run = time.time() + interval_seconds

        time.sleep(2)


###############################################################################
# SECTION 4: main with CLI
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Unified aggregator + training script.")
    parser.add_argument("--summaries", action="store_true",
                        help="Create aggregator_summaries => aggregator_summaries table")
    parser.add_argument("--classifier", action="store_true",
                        help="Create local classifier => aggregator_classifier_probs")
    parser.add_argument("--embeddings", action="store_true",
                        help="Create embeddings => aggregator_embeddings table (PCA)")
    parser.add_argument("--train", action="store_true",
                        help="Time-series training => produce trained_model.pkl")
    parser.add_argument("--schedule", action="store_true",
                        help="Runs aggregator + training every --interval seconds (cron-like).")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between scheduled runs (default=3600).")
    args = parser.parse_args()

    # 1) aggregator steps
    if args.summaries or args.classifier or args.embeddings:
        df_lc = load_lunarcrush_aggregator_data()
        df_cp = load_cryptopanic_aggregator_data()

        # Summaries
        if args.summaries:
            df_summ = create_aggregator_summaries(df_lc, df_cp)
            store_aggregator_summaries(df_summ)

        # Classifier
        if args.classifier:
            build_local_classifier(df_lc)

        # Embeddings
        if args.embeddings:
            build_embedding_summaries(df_lc)

    # 2) time-series training
    if args.train:
        # For demonstration, pick some symbols
        symbols = ["ETH", "XBT", "XRP"]
        df_cp = load_cryptopanic_aggregator_data()  # if needed
        all_X, all_y = [], []
        for s in symbols:
            Xs, ys = build_training_features_time_series(s, df_cp)
            if len(Xs) > MIN_DATA_ROWS:
                all_X.append(Xs)
                all_y.append(ys)
            else:
                logger.warning(f"[Train] => symbol={s} => insufficient rows => skip")

        if not all_X:
            logger.warning("[Train] => no data => skip training.")
        else:
            Xf = pd.concat(all_X, ignore_index=True)
            yf = pd.concat(all_y, ignore_index=True)
            logger.info(f"[Train] => final dataset => X={Xf.shape}, y={yf.shape}")
            train_and_backtest(Xf, yf)

    # 3) schedule approach => aggregator + training every N seconds
    if args.schedule:
        logger.info(f"[Main] => scheduling aggregator + training every {args.interval} s.")
        schedule_training(
            interval_seconds=args.interval,
            do_summaries=args.summaries,
            do_classifier=args.classifier,
            do_embeddings=args.embeddings,
            do_train=args.train
        )


if __name__ == "__main__":
    main()
