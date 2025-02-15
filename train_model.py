#!/usr/bin/env python3
# =============================================================================
# FILE: train_model.py (REVISED)
# =============================================================================
"""
train_model.py (Updated)

Key Features:
-------------
1) Builds aggregator_summaries from 'lunarcrush_data' only (no cryptopanic).
2) Creates a simple local classifier => aggregator_classifier_probs.
3) Builds PCA-based embeddings => aggregator_embeddings.
4) Performs time-series training (SHIFT_BARS=24 => predict next-day 2% up moves).

You can integrate calls from main.py or anywhere else by importing:
    from train_model import (
        build_aggregator_summaries,
        store_aggregator_summaries,
        build_local_classifier,
        store_classifier_probs_table,
        build_embedding_summaries,
        store_embedding_vectors,
        run_time_series_training,
        run_full_training_pipeline
    )

Usage Example from main.py:
---------------------------
    # Suppose you want to refresh aggregator data and then train:

    from train_model import run_full_training_pipeline

    def main():
        # ... your usual aggregator or data updates ...
        run_full_training_pipeline(
            db_path="trades.db",
            symbols_for_timeseries=["ETH","SOL","ADA","XRP","LTC","DOT","LINK","ATOM","NEAR","UNI","FIL","AAVE"]
        )
        # This will:
        #   1) Build aggregator summaries -> aggregator_summaries
        #   2) Build local classifier -> aggregator_classifier_probs
        #   3) Build embeddings -> aggregator_embeddings
        #   4) Time-series train => produce trained_model.pkl & model_info.json

"""

import os
import time
import json
import math
import sqlite3
import logging
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import joblib

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

###############################################################################
# GLOBAL CONFIG
###############################################################################
DB_FILE = "trades.db"
MODEL_OUTPUT_PATH = "trained_model.pkl"
MODEL_INFO_FILE = "model_info.json"

SHIFT_BARS = 24     # time-series shift, e.g. 24 data-points => ~1 day if hour bars
THRESHOLD_UP = 0.01 # +2% threshold => label up
MIN_DATA_ROWS = 100 # minimal data rows required
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################
# SECTION A: AGGREGATOR SUMMARIES
###############################################################################
def load_lunarcrush_data(db_path: str = DB_FILE) -> pd.DataFrame:
    """
    Loads raw aggregator data from 'lunarcrush_data' with new columns
    (e.g. galaxy_score, alt_rank, price_btc, etc.).
    """
    conn = sqlite3.connect(db_path)
    try:
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
        logger.info(f"[Aggregator] loaded {len(df)} rows from lunarcrush_data.")
        return df
    except Exception as e:
        logger.exception(f"[Aggregator] Error loading lunarcrush_data => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def build_aggregator_summaries(
    df_lc: pd.DataFrame,
    db_path: str = DB_FILE,
    merge_tick_size: bool = True
) -> pd.DataFrame:
    """
    Creates aggregator_summaries from the lunarcrush_data fields.
    Optionally merges 'tick_size' from `kraken_asset_pairs` (if available)
    to incorporate minimum increments, though this is purely illustrative.

    :param df_lc: DataFrame from load_lunarcrush_data.
    :param db_path: Path to your SQLite DB for potential merges with kraken_asset_pairs.
    :param merge_tick_size: If True, attempts to fetch 'tick_size' from kraken_asset_pairs.
    :return: aggregator_summaries as a DataFrame.
    """
    if df_lc.empty:
        logger.warning("[Aggregator] No lunarcrush_data => aggregator_summaries empty.")
        return pd.DataFrame()

    def bucket_price(p: float) -> str:
        if p < 10:
            return "low"
        elif p < 200:
            return "medium"
        else:
            return "high"

    rows = []
    for _, row in df_lc.iterrows():
        price_val = row.get("price", 0.0)
        galaxy_val = row.get("galaxy_score", 0.0)
        alt_r = row.get("alt_rank", 999999)
        md = row.get("market_dominance", 0.0)
        sentiment_val = row.get("sentiment", 0.0)

        price_cat = bucket_price(price_val)
        # simple sentiment label
        if sentiment_val > 0.7:
            sent_label = "strong_pos"
        elif sentiment_val > 0.3:
            sent_label = "slightly_pos"
        elif sentiment_val >= 0:
            sent_label = "neutral"
        else:
            sent_label = "negative"

        rows.append({
            "id": row["id"],
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "price_bucket": price_cat,
            "galaxy_score": galaxy_val,
            "galaxy_score_previous": row.get("galaxy_score_previous", 0.0),
            "alt_rank": alt_r,
            "alt_rank_previous": row.get("alt_rank_previous", 999999),
            "market_dominance": md,
            "dominance_bucket": ("dominant" if md > 1.0 else "moderate"),
            "sentiment_label": sent_label
        })

    df_sum = pd.DataFrame(rows)

    # Optionally merge 'tick_size' from kraken_asset_pairs if symbol and wsname match
    if merge_tick_size:
        df_sum = _merge_tick_size_into_summaries(df_sum, db_path)

    logger.info(f"[Aggregator] Built aggregator_summaries => {len(df_sum)} rows.")
    return df_sum


def _merge_tick_size_into_summaries(df_sum: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """
    Example function that tries to match aggregator_summaries symbol
    to 'kraken_asset_pairs' base or altname to incorporate tick_size.

    Because the naming conventions can differ (e.g., "XETH" vs "ETH"),
    you may need a real mapping in practice. This is just an illustration.
    """
    if df_sum.empty:
        return df_sum

    # We'll read pair_name, base, altname, tick_size from kraken_asset_pairs
    conn = sqlite3.connect(db_path)
    try:
        df_pairs = pd.read_sql_query(
            """
            SELECT pair_name, altname, base, quote, tick_size
            FROM kraken_asset_pairs
            """,
            conn
        )
    except Exception as e:
        logger.exception(f"[Aggregator] merge tick_size => {e}")
        conn.close()
        return df_sum
    finally:
        conn.close()

    # Example approach: we assume aggregator "symbol" ~ altname or base minus 'X' prefix
    # We'll do a naive join on uppercase match ignoring 'X' prefix
    def normalize_symbol(x: str) -> str:
        x = x.upper()
        return x.lstrip("X")  # remove leading X if any

    df_pairs["norm"] = df_pairs["altname"].apply(normalize_symbol)
    df_sum["norm"] = df_sum["symbol"].apply(lambda x: str(x).upper())

    merged = pd.merge(
        df_sum,
        df_pairs[["norm","tick_size"]],
        on="norm",
        how="left"
    )
    merged.drop(columns=["norm"], inplace=True)
    return merged


def store_aggregator_summaries(
    df_summary: pd.DataFrame,
    db_path: str = DB_FILE
):
    """
    Persists aggregator_summaries in a dedicated table. Upserts by 'id'.
    Ensure the table has columns matching new fields if needed.
    """
    if df_summary.empty:
        return

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_summaries (
                id INTEGER PRIMARY KEY, 
                symbol TEXT,
                timestamp INTEGER,
                price_bucket TEXT,
                galaxy_score REAL,
                galaxy_score_previous REAL,
                alt_rank REAL,
                alt_rank_previous REAL,
                market_dominance REAL,
                dominance_bucket TEXT,
                sentiment_label TEXT,
                tick_size REAL  -- added if you merged
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
                r["sentiment_label"],
                r.get("tick_size", None)
            ))
        c.executemany(
            """
            INSERT OR REPLACE INTO aggregator_summaries (
                id, symbol, timestamp, price_bucket,
                galaxy_score, galaxy_score_previous,
                alt_rank, alt_rank_previous,
                market_dominance, dominance_bucket,
                sentiment_label, tick_size
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows_to_insert
        )
        conn.commit()
        logger.info(f"[Aggregator] Stored {len(rows_to_insert)} aggregator_summaries rows.")
    except Exception as e:
        logger.exception(f"[Aggregator] store aggregator_summaries => {e}")
    finally:
        conn.close()


###############################################################################
# SECTION B: LOCAL CLASSIFIER => aggregator_classifier_probs
###############################################################################
def build_local_classifier(df_lc: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a naive classifier that labels 'label_up=1' if galaxy_score>50, etc.
    Then trains a RandomForest to produce a probability of 'up' => prob_up.
    Returns a DataFrame with => [id, symbol, timestamp, prob_up].
    """
    if df_lc.empty:
        logger.warning("[Classifier] No data => skipping build.")
        return pd.DataFrame()

    # create label: (galaxy_score>50 => 1, else 0)
    df_lc = df_lc.copy()
    df_lc["label_up"] = (df_lc["galaxy_score"] > 50).astype(int)

    # small set of features
    feats = ["price","market_cap","volume_24h","galaxy_score"]
    feats_exist = [f for f in feats if f in df_lc.columns]
    df_lc.dropna(subset=feats_exist+["label_up"], inplace=True)

    if len(df_lc) < 10:
        logger.warning("[Classifier] Not enough data => skipping build.")
        return pd.DataFrame()

    # train a simple random forest
    X = df_lc[feats_exist].values
    y = df_lc["label_up"].values

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    prob_up = rf.predict_proba(X)[:,1]
    df_out = pd.DataFrame({
        "id": df_lc["id"],
        "symbol": df_lc["symbol"],
        "timestamp": df_lc["timestamp"],
        "prob_up": prob_up
    })

    # save local_classifier to disk (if you want to re-use it)
    joblib.dump(rf, "local_classifier.pkl")
    logger.info("[Classifier] local_classifier.pkl saved.")
    return df_out


def store_classifier_probs_table(df_probs: pd.DataFrame, db_path: str = DB_FILE):
    """
    Insert or replace aggregator_classifier_probs with prob_up.
    """
    if df_probs.empty:
        return
    conn = sqlite3.connect(db_path)
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
        for _, row in df_probs.iterrows():
            inserts.append((
                row["id"], row["symbol"], row["timestamp"], row["prob_up"]
            ))
        c.executemany(
            """
            INSERT OR REPLACE INTO aggregator_classifier_probs
            (id, symbol, timestamp, prob_up)
            VALUES (?,?,?,?)
            """,
            inserts
        )
        conn.commit()
        logger.info(f"[Classifier] => stored {len(inserts)} aggregator_classifier_probs rows.")
    except Exception as e:
        logger.exception(f"[Classifier] store => {e}")
    finally:
        conn.close()


###############################################################################
# SECTION C: EMBEDDINGS => aggregator_embeddings
###############################################################################
def build_embedding_summaries(
    df_lc: pd.DataFrame,
    n_components: int = 3
) -> pd.DataFrame:
    """
    Perform a PCA on numeric columns from lunarcrush_data.
    Return DataFrame => [id,symbol,timestamp, comp1..compN].
    """
    if df_lc.empty:
        logger.warning("[Embeddings] No data => skip.")
        return pd.DataFrame()

    # We can select numeric columns for PCA
    feats = ["price","market_cap","volume_24h","volatility","galaxy_score","alt_rank","market_dominance"]
    feats = [f for f in feats if f in df_lc.columns]
    df_lc.dropna(subset=feats, inplace=True)

    if len(df_lc) < n_components:
        logger.warning("[Embeddings] Not enough rows => skip.")
        return pd.DataFrame()

    X = df_lc[feats].values
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    df_emb = pd.DataFrame(X_reduced, columns=[f"comp{i+1}" for i in range(n_components)])
    df_emb["id"] = df_lc["id"].values
    df_emb["symbol"] = df_lc["symbol"].values
    df_emb["timestamp"] = df_lc["timestamp"].values

    return df_emb


def store_embedding_vectors(df_emb: pd.DataFrame, db_path: str = DB_FILE):
    """
    Persists aggregator_embeddings => upsert by 'id'.
    """
    if df_emb.empty:
        return
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        # Adjust if you want more than 3 PCA comps
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

        items = []
        for _, r in df_emb.iterrows():
            items.append((
                r["id"],
                r["symbol"],
                r["timestamp"],
                r["comp1"],
                r["comp2"],
                r["comp3"]
            ))
        c.executemany(
            """
            INSERT OR REPLACE INTO aggregator_embeddings
            (id, symbol, timestamp, comp1, comp2, comp3)
            VALUES (?,?,?,?,?,?)
            """,
            items
        )
        conn.commit()
        logger.info(f"[Embeddings] Stored {len(items)} aggregator_embeddings rows.")
    except Exception as e:
        logger.exception(f"[Embeddings] => store => {e}")
    finally:
        conn.close()


###############################################################################
# SECTION D: TIME-SERIES TRAINING
###############################################################################
def load_time_series_for_symbol(symbol: str, db_path: str = DB_FILE, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Loads time-series rows from 'lunarcrush_timeseries' for a single symbol,
    sorted ascending by timestamp. If limit is provided, grabs only the newest N rows.
    """
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM lunarcrush_timeseries WHERE UPPER(coin_id)=?", (symbol.upper(),))
        rowcount = c.fetchone()[0] or 0
        if rowcount < 1:
            logger.warning(f"[TimeSeries] No rows for {symbol} in lunarcrush_timeseries")
            return pd.DataFrame()

        if limit and rowcount>limit:
            query = f"""
                SELECT * FROM (
                  SELECT
                    coin_id, timestamp,
                    open_price, close_price, high_price, low_price,
                    volume_24h, market_cap, sentiment, galaxy_score, alt_rank,
                    market_dominance, circulating_supply
                  FROM lunarcrush_timeseries
                  WHERE UPPER(coin_id)=?
                  ORDER BY timestamp DESC
                  LIMIT {limit}
                ) sub
                ORDER BY timestamp ASC
            """
        else:
            query = """
                SELECT 
                    coin_id, timestamp,
                    open_price, close_price, high_price, low_price,
                    volume_24h, market_cap, sentiment, galaxy_score, alt_rank,
                    market_dominance, circulating_supply
                FROM lunarcrush_timeseries
                WHERE UPPER(coin_id)=?
                ORDER BY timestamp ASC
            """

        df = pd.read_sql_query(query, conn, params=[symbol.upper()])
        return df

    except Exception as e:
        logger.exception(f"[TimeSeries] symbol={symbol}, error => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def build_training_features_time_series(
    symbol: str,
    db_path: str = DB_FILE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - SHIFT_BARS => label if future close is +2% from now => label_up=1
    - Rolling features: ma_5, std_10
    - Uses galaxy_score, sentiment from timeseries
    Returns X, y for modeling.
    """
    df_ts = load_time_series_for_symbol(symbol, db_path=db_path)
    if df_ts.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Basic cleaning
    df_ts["close_price"] = df_ts["close_price"].fillna(df_ts["open_price"])
    df_ts.sort_values("timestamp", inplace=True)
    df_ts.reset_index(drop=True, inplace=True)

    # SHIFT_BARS => we label "future_close" SHIFT_BARS ahead
    df_ts["future_close"] = df_ts["close_price"].shift(-SHIFT_BARS)
    df_ts["pct_future_gain"] = (
        (df_ts["future_close"] - df_ts["close_price"]) / (df_ts["close_price"]+1e-12)
    )
    df_ts["label_up"] = (df_ts["pct_future_gain"]>=THRESHOLD_UP).astype(int)
    df_ts.dropna(subset=["future_close"], inplace=True)

    # Rolling features
    df_ts["ma_5"] = df_ts["close_price"].rolling(5).mean()
    df_ts["std_10"] = df_ts["close_price"].rolling(10).std()
    df_ts["ma_5"].fillna(method="bfill", inplace=True)
    df_ts["std_10"].fillna(method="bfill", inplace=True)

    # Keep a handful of features
    feats = ["close_price","ma_5","std_10","galaxy_score","sentiment"]
    df_ts.dropna(subset=feats, inplace=True)

    X = df_ts[feats].copy()
    y = df_ts["label_up"].copy()
    return X,y


def train_and_backtest(X: pd.DataFrame, y: pd.Series):
    """
    Time-series cross-validation => pick best hyperparams => final hold-out => store model.
    Saves 'trained_model.pkl' and 'model_info.json' with metrics.
    """
    if X.empty or len(X)<10:
        logger.warning("[Train] => insufficient data => skip.")
        return None

    # time-series CV
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        "n_estimators":[50,100],
        "max_depth":[None,5,10]
    }
    base = RandomForestClassifier(random_state=42)
    g = GridSearchCV(base, param_grid, scoring="f1_macro", cv=tscv, n_jobs=-1)
    g.fit(X, y)

    best = g.best_estimator_
    logger.info(f"[Train] best_params={g.best_params_}, best_cv_f1={g.best_score_:.4f}")

    # hold-out => last 20% for test
    hold_size = max(1, int(len(X)*0.2))
    X_train, y_train = X.iloc[:-hold_size], y.iloc[:-hold_size]
    X_test, y_test = X.iloc[-hold_size:], y.iloc[-hold_size:]

    best.fit(X_train,y_train)
    y_pred = best.predict(X_test)

    f1m = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test,y_pred)
    logger.info(f"[Train] => hold-out => f1={f1m:.4f}, acc={acc:.4f}")

    joblib.dump(best, MODEL_OUTPUT_PATH)
    info = {"f1": f1m, "acc": acc, "params": g.best_params_}
    with open(MODEL_INFO_FILE,"w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"[Train] => model saved => {MODEL_OUTPUT_PATH}, info => {MODEL_INFO_FILE}")
    return best


def run_time_series_training(
    symbols: List[str],
    db_path: str = DB_FILE
):
    """
    A helper that loops over multiple symbols => aggregates X,y => trains a single model.
    This is your final pipeline for the time-series part only.
    """
    all_X, all_y = [], []
    for s in symbols:
        Xs, ys = build_training_features_time_series(s, db_path=db_path)
        if len(Xs)>MIN_DATA_ROWS:
            all_X.append(Xs)
            all_y.append(ys)
        else:
            logger.warning(f"[Train] => symbol={s} => insufficient rows => skip.")
    if not all_X:
        logger.warning("[Train] => no data => skip.")
        return None

    Xfull = pd.concat(all_X, ignore_index=True)
    yfull = pd.concat(all_y, ignore_index=True)
    logger.info(f"[Train] => final dataset => X={Xfull.shape}, y={yfull.shape}")
    model = train_and_backtest(Xfull, yfull)
    return model


###############################################################################
# SECTION E: FULL PIPELINE
###############################################################################
def run_full_training_pipeline(
    db_path: str = DB_FILE,
    symbols_for_timeseries: List[str] = None
):
    """
    1) aggregator_summaries -> aggregator_summaries table
    2) local classifier -> aggregator_classifier_probs
    3) embeddings -> aggregator_embeddings
    4) time-series training => trained_model.pkl
    """
    # 1) aggregator summaries
    df_lc = load_lunarcrush_data(db_path)
    df_summ = build_aggregator_summaries(df_lc, db_path=db_path, merge_tick_size=True)
    store_aggregator_summaries(df_summ, db_path=db_path)

    # 2) local classifier
    df_probs = build_local_classifier(df_lc)
    if not df_probs.empty:
        store_classifier_probs_table(df_probs, db_path=db_path)

    # 3) embeddings
    df_emb = build_embedding_summaries(df_lc, n_components=3)
    if not df_emb.empty:
        store_embedding_vectors(df_emb, db_path=db_path)

    # 4) time-series training
    if not symbols_for_timeseries:
        symbols_for_timeseries = ["ETH","XRP","ADA","SOL"]  # fallback
    run_time_series_training(symbols_for_timeseries, db_path=db_path)

    logger.info("[train_model] => Full training pipeline completed.")
