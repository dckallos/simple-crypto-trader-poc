#!/usr/bin/env python3
# =============================================================================
# FILE: train_model.py
# =============================================================================
"""
train_model.py

Unifies aggregator data management + a time-series-based scikit-learn training pipeline,
using 'lunarcrush_data', 'lunarcrush_timeseries', 'cryptopanic_posts', plus aggregator tables:
    - aggregator_summaries
    - aggregator_classifier_probs
    - aggregator_embeddings

It supports 4 main CLI flags:
    --summaries => gather aggregator summaries & store in aggregator_summaries
    --classifier => build local classifier => aggregator_classifier_probs
    --embeddings => do PCA-based embeddings => aggregator_embeddings
    --train => do final time-series-driven training, produce 'trained_model.pkl' & stats

Example usage:
    python train_model.py --summaries --classifier --embeddings --train

Internally:
    1) aggregator data loaded from 'lunarcrush_data' & 'cryptopanic_posts'
       => aggregator_summaries => aggregator_classifier_probs => aggregator_embeddings
    2) time-series from 'lunarcrush_timeseries' => SHIFT_BARS => label future price jump
    3) train random forest => backtest => store final model + stats.

Additionally, this script has an optional 'schedule_training()' function showing
how you might queue repeated training calls on a configurable timeframe, e.g. every hour.

Requires:
    - scikit-learn, numpy, pandas
    - joblib for model saving
    - your DB (trades.db or as configured) to hold aggregator & time-series data
    - cryptopanic_posts & lunarcrush_data/lunarcrush_timeseries tables pre-populated
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report
)

# The SHIFT_BARS / THRESHOLD_UP approach for labeling
SHIFT_BARS         = 24
THRESHOLD_UP       = 0.02
MIN_DATA_ROWS      = 100   # minimal data required for training

# We will store final model
MODEL_OUTPUT_PATH  = "trained_model.pkl"
METRICS_OUTPUT_FILE= "training_metrics.csv"
MODEL_INFO_FILE    = "model_info.json"

DB_FILE = "trades.db"  # or your actual DB path. Must hold lunarcrush_data, lunarcrush_timeseries, cryptopanic_posts, etc.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################
# SECTION 1: aggregator management => Summaries, Classifier, Embeddings
###############################################################################
def load_lunarcrush_aggregator_data() -> pd.DataFrame:
    """
    Loads aggregator snapshot from 'lunarcrush_data'. For demonstration,
    we pick columns we care about for aggregator summaries or local classifier.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        q = """
            SELECT
                id,
                symbol,
                timestamp,
                price,
                market_cap,
                volume_24h,
                volatility,
                galaxy_score,
                alt_rank,
                sentiment
            FROM lunarcrush_data
        """
        df = pd.read_sql_query(q, conn)
        logger.info(f"[Aggregator] loaded {len(df)} rows from lunarcrush_data.")
        return df
    except Exception as e:
        logger.exception(f"[Aggregator] error => {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_cryptopanic_aggregator_data() -> pd.DataFrame:
    """
    Example aggregator from 'cryptopanic_posts'.
    We'll do a naive approach => daily avg => returning just the entire table for now.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        q = """
            SELECT id, sentiment_score, created_at
            FROM cryptopanic_posts
        """
        df = pd.read_sql_query(q, conn)
        logger.info(f"[Aggregator] loaded {len(df)} cryptopanic_posts.")
        return df
    except Exception as e:
        logger.exception(f"[Aggregator] cryptopanic => {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Summaries => aggregator_summaries
# ------------------------------------------------------------------------------
def create_aggregator_summaries(df_lc: pd.DataFrame, df_cp: pd.DataFrame) -> pd.DataFrame:
    """
    We produce short textual or numeric summaries, e.g. 'price_bucket', 'sentiment_label'.
    """
    if df_lc.empty:
        logger.warning("[Summaries] => no data => skip.")
        return pd.DataFrame()

    def bucket_price(p: float) -> str:
        if p < 1000: return "low"
        elif p < 5000: return "medium"
        else: return "high"

    def label_sentiment(s: float) -> str:
        if s is None or np.isnan(s): return "unknown"
        if s>0.6: return "strong_pos"
        elif s>0.3: return "slightly_pos"
        elif s>0.0: return "neutral"
        else: return "negative"

    # We'll ignore cryptopanic for brevity here, or you can merge daily sentiment if you prefer.
    rows=[]
    for _, row in df_lc.iterrows():
        pcat = bucket_price(row["price"] if not pd.isnull(row["price"]) else 0.0)
        scat = label_sentiment(row["sentiment"] if not pd.isnull(row["sentiment"]) else 0.0)
        rows.append({
            "id": row["id"],
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "price_bucket": pcat,
            "galaxy_score": row["galaxy_score"],
            "alt_rank": row["alt_rank"],
            "sentiment_label": scat
        })
    df_sum = pd.DataFrame(rows)
    logger.info(f"[Summaries] => built {len(df_sum)} aggregator summaries.")
    return df_sum


def store_aggregator_summaries(df_summary: pd.DataFrame):
    if df_summary.empty:
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_summaries (
                id INTEGER PRIMARY KEY,  -- same as lunarcrush_data.id
                symbol TEXT,
                timestamp INTEGER,
                price_bucket TEXT,
                galaxy_score REAL,
                alt_rank REAL,
                sentiment_label TEXT
            )
        """)
        conn.commit()

        rows_to_insert=[]
        for _, r in df_summary.iterrows():
            rows_to_insert.append((
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
            VALUES (?,?,?,?,?,?,?)
        """, rows_to_insert)
        conn.commit()
        logger.info(f"[Summaries] => stored {len(rows_to_insert)} aggregator_summaries rows.")
    except Exception as e:
        logger.exception(f"[Summaries] => store error => {e}")
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Local Classifier => aggregator_classifier_probs
# ------------------------------------------------------------------------------
def build_local_classifier(df_lc: pd.DataFrame):
    """
    We'll do a naive label => label_up=1 if galaxy_score>50 else 0 => produce prob_up => aggregator_classifier_probs
    """
    if df_lc.empty:
        logger.warning("[Classifier] => no data => skip.")
        return

    df_lc = df_lc.copy()
    df_lc["label_up"] = (df_lc["galaxy_score"]>50).astype(int)

    feats=["price","market_cap","volume_24h","galaxy_score","alt_rank"]
    df_lc.dropna(subset=feats+["label_up"], inplace=True)

    if len(df_lc)<10:
        logger.warning("[Classifier] => not enough data => skip.")
        return

    X = df_lc[feats].values
    y = df_lc["label_up"].values

    rf=RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X,y)

    prob_up = rf.predict_proba(X)[:,1]
    df_probs = pd.DataFrame({
        "id": df_lc["id"],
        "symbol": df_lc["symbol"],
        "timestamp": df_lc["timestamp"],
        "prob_up": prob_up
    })
    store_classifier_probs_table(df_probs)

    import joblib
    joblib.dump(rf, "local_classifier.pkl")
    logger.info("[Classifier] => local classifier saved to local_classifier.pkl")

def store_classifier_probs_table(df_probs: pd.DataFrame):
    if df_probs.empty:
        return
    conn=sqlite3.connect(DB_FILE)
    try:
        c=conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS aggregator_classifier_probs (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp INTEGER,
                prob_up REAL
            )
        """)
        conn.commit()

        to_ins=[]
        for _, row in df_probs.iterrows():
            to_ins.append((row["id"], row["symbol"], row["timestamp"], row["prob_up"]))

        c.executemany("""
            INSERT OR REPLACE INTO aggregator_classifier_probs
            (id, symbol, timestamp, prob_up)
            VALUES (?,?,?,?)
        """, to_ins)
        conn.commit()
        logger.info(f"[Classifier] => stored {len(to_ins)} aggregator_classifier_probs rows.")
    except Exception as e:
        logger.exception(f"[Classifier] => store => {e}")
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Embedding-based Summaries => aggregator_embeddings
# ------------------------------------------------------------------------------
def build_embedding_summaries(df_lc: pd.DataFrame, n_components=3):
    if df_lc.empty:
        logger.warning("[Embeddings] => no data => skip.")
        return

    feats=["price","market_cap","volume_24h","volatility","galaxy_score","alt_rank","sentiment"]
    df_lc = df_lc.dropna(subset=feats).copy()
    if len(df_lc)<n_components:
        logger.warning("[Embeddings] => not enough data => skip.")
        return

    X = df_lc[feats].values
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    df_emb = pd.DataFrame(X_reduced, columns=[f"comp{i+1}" for i in range(n_components)])
    df_emb["id"] = df_lc["id"].values
    df_emb["symbol"] = df_lc["symbol"].values
    df_emb["timestamp"] = df_lc["timestamp"].values

    store_embedding_vectors(df_emb)

def store_embedding_vectors(df_emb: pd.DataFrame):
    if df_emb.empty:
        return
    conn=sqlite3.connect(DB_FILE)
    try:
        c=conn.cursor()
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

        inserts=[]
        for _, r in df_emb.iterrows():
            inserts.append((r["id"], r["symbol"], r["timestamp"],
                            r["comp1"], r["comp2"], r["comp3"]))

        c.executemany("""
            INSERT OR REPLACE INTO aggregator_embeddings
            (id, symbol, timestamp, comp1, comp2, comp3)
            VALUES (?,?,?,?,?,?)
        """, inserts)
        conn.commit()
        logger.info(f"[Embeddings] => stored {len(inserts)} aggregator embeddings.")
    except Exception as e:
        logger.exception(f"[Embeddings] => store => {e}")
    finally:
        conn.close()

###############################################################################
# SECTION 2: final time-series-based training
###############################################################################
def load_time_series_for_symbol(symbol: str, db_file=DB_FILE, limit:Optional[int]=None) -> pd.DataFrame:
    """
    Suppose we store coin_id as uppercase symbol in the 'lunarcrush_timeseries' => coin_id column.
    We'll load that to build SHIFT_BARS features. If you store numeric IDs, adapt accordingly.
    """
    conn=sqlite3.connect(db_file)
    try:
        # We assume coin_id is just UPPER(symbol). If not, adapt.
        q="SELECT COUNT(*) FROM lunarcrush_timeseries WHERE coin_id=?"
        c=conn.cursor()
        c.execute(q,(symbol.upper(),))
        rowcount = c.fetchone()[0] or 0
        if rowcount<1:
            logger.warning(f"[TimeSeries] => no rows for symbol={symbol}")
            return pd.DataFrame()

        if limit and rowcount>limit:
            # read last limit
            logger.info(f"[TimeSeries] => symbol={symbol} => rowcount={rowcount} => limit to last {limit}")
            q = f"""
                SELECT *
                FROM (
                  SELECT
                    coin_id, timestamp,
                    open_price, close_price, high_price, low_price,
                    volume_24h, market_cap, sentiment, galaxy_score, alt_rank
                  FROM lunarcrush_timeseries
                  WHERE coin_id=?
                  ORDER BY timestamp DESC
                  LIMIT {limit}
                ) sub
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(q, conn, params=[symbol.upper()])
        else:
            q2 = """
                SELECT
                  coin_id, timestamp,
                  open_price, close_price, high_price, low_price,
                  volume_24h, market_cap, sentiment, galaxy_score, alt_rank
                FROM lunarcrush_timeseries
                WHERE coin_id=?
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(q2, conn, params=[symbol.upper()])
        return df
    except Exception as e:
        logger.exception(f"[TimeSeries] => symbol={symbol}, {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def build_training_features_time_series(symbol:str, df_cp:pd.DataFrame) -> Tuple[pd.DataFrame,pd.Series]:
    """
    1) load timeseries => SHIFT_BARS => label
    2) optionally merge aggregator or cryptopanic if needed
    3) return (X, y)
    """
    df_ts = load_time_series_for_symbol(symbol)
    if df_ts.empty:
        return pd.DataFrame(), pd.Series([],dtype=int)

    # We'll skip aggregator merges for brevity, or you can do it similarly to build_features() from earlier.
    df_ts["datetime"] = pd.to_datetime(df_ts["timestamp"], unit="s")
    df_ts["close_price"] = df_ts["close_price"].fillna(df_ts["open_price"])
    df_ts.sort_values("timestamp", inplace=True)
    df_ts.reset_index(drop=True,inplace=True)

    # example: SHIFT_BARS => label
    df_ts["future_close"] = df_ts["close_price"].shift(-SHIFT_BARS)
    df_ts["pct_future_gain"] = (df_ts["future_close"]-df_ts["close_price"])/(df_ts["close_price"]+1e-9)
    df_ts["label_up"] = (df_ts["pct_future_gain"]>=THRESHOLD_UP).astype(int)
    df_ts.dropna(subset=["future_close"],inplace=True)

    # build features
    df_ts["ma_5"] = df_ts["close_price"].rolling(5).mean()
    df_ts["std_10"] = df_ts["close_price"].rolling(10).std()
    # fill
    df_ts["ma_5"].fillna(method="bfill", inplace=True)
    df_ts["std_10"].fillna(method="bfill", inplace=True)

    # final feats
    feats=["close_price","ma_5","std_10","galaxy_score","sentiment"]
    df_ts.dropna(subset=feats, inplace=True)

    X = df_ts[feats].copy()
    y = df_ts["label_up"].copy()
    return X,y

def train_and_backtest(X:pd.DataFrame, y:pd.Series):
    """
    Typical random forest with time-series CV => produce model => backtest => store metrics
    """
    if X.empty or len(X)<10:
        logger.warning("[Train] => not enough data => skip.")
        return None

    tscv=TimeSeriesSplit(n_splits=3)
    param_grid={
      "n_estimators":[50,100],
      "max_depth":[None,5,10]
    }
    base = RandomForestClassifier(random_state=42)
    g=GridSearchCV(base, param_grid, scoring="f1_macro", cv=tscv,n_jobs=-1)
    g.fit(X,y)

    best = g.best_estimator_
    logger.info(f"[Train] => best={g.best_params_}, best CV f1={g.best_score_:.4f}")

    hold_size = max(1,int(len(X)*0.2))
    X_train=X.iloc[:-hold_size]
    y_train=y.iloc[:-hold_size]
    X_test=X.iloc[-hold_size:]
    y_test=y.iloc[-hold_size:]

    best.fit(X_train,y_train)
    y_pred=best.predict(X_test)
    f1m=f1_score(y_test,y_pred,average="macro")

    acc=accuracy_score(y_test,y_pred)
    logger.info(f"[Train] => hold-out => f1={f1m:.4f}, acc={acc:.4f}")

    # store the model
    joblib.dump(best, MODEL_OUTPUT_PATH)
    logger.info(f"[Train] => model saved => {MODEL_OUTPUT_PATH}")

    # store metrics => JSON
    info={"f1":f1m,"acc":acc,"params":g.best_params_}
    with open(MODEL_INFO_FILE,"w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"[Train] => saved model_info => {MODEL_INFO_FILE}")

    return best

###############################################################################
# SECTION 3: Subprocess or scheduling approach
###############################################################################
def schedule_training(interval_seconds: int=3600, do_summaries=True, do_classifier=True, do_embeddings=True, do_train=True):
    """
    Illustrates how you might queue repeated aggregator & training updates in intervals:
      - every 'interval_seconds' => run aggregator steps => train
    This is purely an example. In real usage, you'd do a background thread, or cron, or systemd timers, etc.
    """
    logger.info("[Scheduler] => start scheduling aggregator + training every %ds..." % interval_seconds)
    next_run = time.time()
    while True:
        now=time.time()
        if now>=next_run:
            logger.info("[Scheduler] => Time for aggregator + training pipeline...")

            if do_summaries or do_classifier or do_embeddings:
                # aggregator approach => load aggregator data
                df_lc = load_lunarcrush_aggregator_data()
                df_cp = load_cryptopanic_aggregator_data()

                if do_summaries:
                    df_summ = create_aggregator_summaries(df_lc, df_cp)
                    store_aggregator_summaries(df_summ)

                if do_classifier:
                    build_local_classifier(df_lc)

                if do_embeddings:
                    build_embedding_summaries(df_lc)

            if do_train:
                # pick your symbols => or read from config
                # for demonstration, we do a naive approach => "ETH","BTC"
                # or parse from config.yaml
                symbols=["ETH","BTC","XRP"]  # example
                all_X, all_y = [], []
                for s in symbols:
                    Xs, ys = build_training_features_time_series(s, df_cp)
                    if len(Xs)>MIN_DATA_ROWS:
                        all_X.append(Xs)
                        all_y.append(ys)
                if all_X:
                    X_full=pd.concat(all_X, ignore_index=True)
                    y_full=pd.concat(all_y, ignore_index=True)
                    model=train_and_backtest(X_full,y_full)
                    logger.info(f"[Scheduler] => training done for symbols => {symbols}")

            next_run = time.time()+interval_seconds

        time.sleep(2)


###############################################################################
# SECTION 4: main with CLI
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Unified aggregator management + training script.")
    parser.add_argument("--summaries", action="store_true", help="Generate aggregator_summaries => aggregator_summaries table")
    parser.add_argument("--classifier", action="store_true", help="Local classifier => aggregator_classifier_probs")
    parser.add_argument("--embeddings", action="store_true", help="Local PCA => aggregator_embeddings table")
    parser.add_argument("--train", action="store_true", help="Time-series training => produce final model")
    parser.add_argument("--schedule", action="store_true", help="Example approach => run aggregator + training in intervals.")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between scheduled runs => default=3600 (1hr).")
    args=parser.parse_args()

    # aggregator portion
    if args.summaries or args.classifier or args.embeddings:
        # load aggregator data from snapshot tables
        df_lc = load_lunarcrush_aggregator_data()
        df_cp = load_cryptopanic_aggregator_data()

        if args.summaries:
            df_sum = create_aggregator_summaries(df_lc, df_cp)
            store_aggregator_summaries(df_sum)

        if args.classifier:
            build_local_classifier(df_lc)

        if args.embeddings:
            build_embedding_summaries(df_lc)

    # final train
    if args.train:
        # might read config.yaml => traded_pairs
        # or we define a small set for demonstration
        symbols=["ETH","XBT","XRP"]
        df_cp = load_cryptopanic_aggregator_data()  # if we want to merge
        all_X,all_y=[],[]
        for s in symbols:
            Xs, ys = build_training_features_time_series(s, df_cp)
            if len(Xs)>MIN_DATA_ROWS:
                all_X.append(Xs)
                all_y.append(ys)
            else:
                logger.warning(f"[Train] => symbol={s} => insufficient rows => skip")

        if not all_X:
            logger.warning("[Train] => no data => skip training.")
        else:
            Xf=pd.concat(all_X, ignore_index=True)
            yf=pd.concat(all_y, ignore_index=True)
            logger.info(f"[Train] => final dataset => X={Xf.shape}, y={yf.shape}")
            train_and_backtest(Xf,yf)

    # scheduling approach => runs aggregator steps + train every N seconds
    if args.schedule:
        logger.info("[Main] => scheduling approach => aggregator + training every %d s" % args.interval)
        schedule_training(
            interval_seconds=args.interval,
            do_summaries=args.summaries,
            do_classifier=args.classifier,
            do_embeddings=args.embeddings,
            do_train=args.train
        )

if __name__=="__main__":
    main()
