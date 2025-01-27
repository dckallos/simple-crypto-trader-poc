#!/usr/bin/env python3
# =============================================================================
# FILE: train_model.py
# =============================================================================
"""
train_model.py

A script that trains a scikit-learn model primarily from LunarCrush time-series data,
optionally merging aggregator data from:
  - CryptoPanic (daily aggregator) => now stored in 'cryptopanic_posts'
  - LunarCrush snapshot => 'lunarcrush_data' (e.g. galaxy_score, alt_rank, etc.)

Enhancements:
  - We **gracefully skip** coins with minimal time-series. A new 'MIN_DATA_ROWS'
    threshold ensures we only train on coins that have enough valid rows
    after building features.
  - We read 'cryptopanic_posts' to compute daily averages of sentiment
    (instead of referencing the old 'cryptopanic_news').
  - If all coins are skipped, we log a warning instead of forcibly
    preventing the script from continuing.
  - We produce final training/backtest stats in a "model_info.json".

Usage:
    python train_model.py

Config or Hardcoded:
    - SHIFT_BARS: how many bars forward we look for a future price jump
    - THRESHOLD_UP: how large a % jump to label "up"
    - MIN_DATA_ROWS: minimal rows needed for a coin to remain in training
    - These can come from config.yaml or default constants.

Environment:
    - DB_FILE => 'trades.db', or set it here.
    - If config.yaml is present => we read traded_pairs, min_data_rows, etc.
"""

import os
import logging
import sqlite3
import json
import pandas as pd
import numpy as np
import yaml
import joblib
import warnings
from typing import Tuple, Dict, Optional, List
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# Config & Constants
# ------------------------------------------------------------------------------
DB_FILE            = "trades.db"
MODEL_OUTPUT_PATH  = "trained_model.pkl"
METRICS_OUTPUT_FILE= "training_metrics.csv"
MODEL_INFO_FILE    = "model_info.json"

SHIFT_BARS         = 24       # how many bars forward we look for future price jump
THRESHOLD_UP       = 0.02     # +2% => label "price up"
MIN_DATA_ROWS      = 100      # minimal data rows after building features

# If config.yaml => override
if os.path.exists("config.yaml"):
    with open("config.yaml","r") as f:
        config = yaml.safe_load(f)
    TRADED_PAIRS    = config.get("traded_pairs", [])
    MIN_DATA_ROWS   = config.get("min_data_rows", 200)
else:
    logger.warning("No config.yaml => defaulting.")
    TRADED_PAIRS    = []
    MIN_DATA_ROWS   = 200

# ------------------------------------------------------------------------------
# CryptoPanic aggregator => from 'cryptopanic_posts'
# ------------------------------------------------------------------------------
def fetch_cryptopanic_aggregated(db_file: str) -> pd.DataFrame:
    """
    Loads daily aggregator from 'cryptopanic_posts' => [news_date, avg_sentiment].

    Implementation:
      SELECT
        DATE(created_at, 'unixepoch') as news_date,
        AVG(sentiment_score) as avg_sentiment
      FROM cryptopanic_posts
      GROUP BY news_date
      ORDER BY news_date

    If the table or data is missing => returns empty DataFrame,
    so we can skip aggregator merges gracefully.
    """
    if not os.path.exists(db_file):
        logger.error(f"DB file not found => {db_file}. No cryptopanic aggregator.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_file)
    try:
        query = """
            SELECT
              DATE(created_at, 'unixepoch') as news_date,
              AVG(sentiment_score) as avg_sentiment
            FROM cryptopanic_posts
            GROUP BY news_date
            ORDER BY news_date
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.exception(f"Error reading from 'cryptopanic_posts': {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# LunarCrush aggregator => from 'lunarcrush_data'
# ------------------------------------------------------------------------------
def fetch_lunarcrush_data_for_symbol(db_file: str, symbol: str) -> pd.DataFrame:
    """
    Suppose your 'lunarcrush_data' table is daily or snapshot-based.
    We'll do a naive aggregator approach:
      SELECT
        DATE(inserted_at, 'unixepoch') as lc_date,
        AVG(galaxy_score) as galaxy_score,
        AVG(alt_rank) as alt_rank,
        ...
      FROM lunarcrush_data
      WHERE UPPER(symbol)=UPPER(?)
      GROUP BY lc_date
    Then we can merge on date in the time-series build_features step.
    """
    if not os.path.exists(db_file):
        logger.error(f"DB file not found => {db_file}. No lunarcrush snapshot aggregator.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_file)
    try:
        query=f"""
            SELECT
              DATE(inserted_at, 'unixepoch') as lc_date,
              AVG(galaxy_score) as galaxy_score,
              AVG(alt_rank) as alt_rank,
              AVG(price) as avg_price_lc,
              AVG(volume_24h) as avg_vol_24h,
              AVG(market_cap) as avg_mcap,
              AVG(sentiment) as avg_lc_sent
            FROM lunarcrush_data
            WHERE UPPER(symbol)=UPPER('{symbol}')
            GROUP BY lc_date
            ORDER BY lc_date
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.exception(f"Error reading aggregator from lunarcrush_data => {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Time-series => from 'lunarcrush_timeseries'
# ------------------------------------------------------------------------------
def load_lunarcrush_timeseries(db_file: str, coin_id: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Reads from 'lunarcrush_timeseries' => columns typically:
      coin_id, timestamp, open_price, close_price, high_price, low_price,
      volume_24h, market_cap, sentiment, galaxy_score, alt_rank
    Sort ascending by timestamp. If limit => read only last N rows.

    Return => DataFrame
    """
    if not os.path.exists(db_file):
        logger.error(f"DB not found => can't load => {db_file}")
        return pd.DataFrame()

    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM lunarcrush_timeseries WHERE coin_id=?",(coin_id,))
        rowcount = c.fetchone()[0] or 0
        if rowcount==0:
            logger.warning(f"No timeseries for coin_id={coin_id}")
            return pd.DataFrame()

        if limit and rowcount>limit:
            logger.info(f"coin_id={coin_id} => {rowcount} rows => limit to last {limit}")
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
            df = pd.read_sql_query(q, conn, params=[coin_id])
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
            df = pd.read_sql_query(q2, conn, params=[coin_id])
        return df
    except Exception as e:
        logger.exception(f"Error loading timeseries => coin_id={coin_id}, {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Build features
# ------------------------------------------------------------------------------
def build_features(
    df_ts: pd.DataFrame,
    df_cp: pd.DataFrame = None,
    df_lc: pd.DataFrame = None,
    shift_bars: int = 24,
    threshold_up: float = 0.02
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build advanced features from a single coin's time-series, merges aggregator from:
      - cryptopanic_aggregated => [news_date, avg_sentiment]
      - lunarcrush_data aggregator => [lc_date, galaxy_score, alt_rank, avg_lc_sent, ...]
    SHIFT_BARS => how many bars forward we look for a future close price jump
    THRESHOLD_UP => how large a % jump to label "up"

    Return => (X, y). X is features, y is label_up in {0,1}.
    If insufficient data => we can return empty X,y.
    """
    if df_ts.empty or "close_price" not in df_ts.columns:
        return pd.DataFrame(), pd.Series([], dtype=int)

    df = df_ts.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["date"] = df["datetime"].dt.date

    # Basic indicators
    df["price"] = df["close_price"]
    df["ma_5"]  = df["close_price"].rolling(5).mean()
    df["ma_20"] = df["close_price"].rolling(20).mean()
    df["std_10"]= df["close_price"].rolling(10).std()

    # RSI
    wlen=14
    delta = df["close_price"].diff()
    gain  = delta.clip(lower=0)
    loss  = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(wlen).mean()
    avg_loss = loss.rolling(wlen).mean()
    rs = avg_gain/(avg_loss+1e-9)
    df["rsi"] = 100 - (100/(1+rs))

    # local sentiment => if 'sentiment' col
    if "sentiment" in df.columns:
        df["sent_7"] = df["sentiment"].rolling(7).mean()
    else:
        df["sent_7"] = 0.0

    # SHIFT_BARS => label "up"
    df["future_close"] = df["close_price"].shift(-shift_bars)
    df["pct_future_gain"] = (df["future_close"] - df["close_price"]) / (df["close_price"]+1e-9)
    df["label_up"] = (df["pct_future_gain"]>=threshold_up).astype(int)
    df.dropna(subset=["future_close"], inplace=True)

    # aggregator merges
    if df_cp is not None and not df_cp.empty:
        # cryptopanic => [news_date, avg_sentiment]
        # rename => date => merge on date
        df_cp2 = df_cp.rename(columns={"news_date":"date"})
        df = pd.merge(df, df_cp2, on="date", how="left")
        df["avg_sentiment"] = df["avg_sentiment"].fillna(0)

    if df_lc is not None and not df_lc.empty:
        # lunarcrush => [lc_date, galaxy_score, alt_rank, avg_lc_sent, etc.]
        df_lc2 = df_lc.rename(columns={"lc_date":"date"})
        df = pd.merge(df, df_lc2, on="date", how="left")
        for col in ["galaxy_score","alt_rank","avg_lc_sent"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # final features
    feats = ["price","ma_5","ma_20","std_10","rsi","sent_7"]
    if "avg_sentiment" in df.columns:
        feats.append("avg_sentiment")
    if "galaxy_score" in df.columns:
        feats.append("galaxy_score")
    if "alt_rank" in df.columns:
        feats.append("alt_rank")
    if "avg_lc_sent" in df.columns:
        feats.append("avg_lc_sent")

    X = df[feats].copy()
    y = df["label_up"].copy()

    return X, y

# ------------------------------------------------------------------------------
# Train model => time-series split => hold-out
# ------------------------------------------------------------------------------
def train_random_forest(X: pd.DataFrame, y: pd.Series):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    if X.empty or len(X)<10:
        logger.warning("Insufficient data => skip training.")
        return None, None

    logger.info("Label distribution =>\n%s", y.value_counts())

    tscv = TimeSeriesSplit(n_splits=3)
    param_grid={
      "n_estimators": [50,100],
      "max_depth": [None,5,10],
      "class_weight": [None,"balanced"]
    }
    base=RandomForestClassifier(random_state=42)
    g=GridSearchCV(base, param_grid, scoring="f1_macro", cv=tscv, n_jobs=-1)
    g.fit(X,y)

    best=g.best_estimator_
    logger.info(f"Best params => {g.best_params_}, best CV F1 => {g.best_score_:.4f}")

    hold_size = max(1, int(len(X)*0.2))
    X_train = X.iloc[:-hold_size]
    y_train = y.iloc[:-hold_size]
    X_test  = X.iloc[-hold_size:]
    y_test  = y.iloc[-hold_size:]

    best.fit(X_train,y_train)
    y_pred=best.predict(X_test)

    acc=accuracy_score(y_test,y_pred)
    bal_acc=balanced_accuracy_score(y_test,y_pred)
    f1m=f1_score(y_test,y_pred,average="macro")

    logger.info(f"Final hold-out => Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1macro={f1m:.4f}")
    cm=confusion_matrix(y_test,y_pred)
    logger.info("Confusion =>\n%s", cm)
    logger.info("Report =>\n%s", classification_report(y_test,y_pred))

    return best, f1m

def log_accuracy_to_csv(val: float):
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(METRICS_OUTPUT_FILE):
        with open(METRICS_OUTPUT_FILE,"w") as f:
            f.write("timestamp,accuracy\n")
    with open(METRICS_OUTPUT_FILE,"a") as f:
        f.write(f"{stamp},{val:.4f}\n")

# ------------------------------------------------------------------------------
# Probability-based naive backtest
# ------------------------------------------------------------------------------
def backtest_prob(
    X: pd.DataFrame,
    model,
    buy_threshold=0.6,
    sell_threshold=0.4,
    stop_loss_pct=0.03,
    take_profit_pct=0.05,
    shift_bars=24
):
    """
    If p_up> buy_threshold => open long => hold up to shift_bars or stop/take-profit.
    Very naive demonstration.
    """
    if X.empty:
        logger.info("[Backtest] => no data => skip.")
        return pd.DataFrame(), {}

    if not hasattr(model,"predict_proba"):
        logger.warning("[Backtest] => model has no predict_proba => skip.")
        return pd.DataFrame(), {}

    if "price" not in X.columns:
        logger.warning("[Backtest] => missing 'price' column => skip.")
        return pd.DataFrame(), {}

    prob_up = model.predict_proba(X)[:,1]
    prices  = X["price"].values
    trades=[]
    i=0
    position=0
    entry=0.0

    while i<len(X):
        p=prices[i]
        pu=prob_up[i]
        if position==0:
            if pu>buy_threshold:
                # buy
                position=1
                entry=p
                trades.append({"index":i,"action":"BUY_OPEN","price":p})
            i+=1
        else:
            exit_i = i+shift_bars
            if exit_i>=len(X):
                trades.append({"index":len(X)-1,"action":"FORCED_EXIT","price":prices[-1]})
                break
            forced=False
            for j in range(i,exit_i):
                ret=(prices[j]-entry)/entry
                if ret<=-abs(stop_loss_pct):
                    trades.append({"index":j,"action":"STOP_OUT","price":prices[j]})
                    i=j+1
                    position=0
                    forced=True
                    break
                if ret>=take_profit_pct:
                    trades.append({"index":j,"action":"TAKE_PROFIT","price":prices[j]})
                    i=j+1
                    position=0
                    forced=True
                    break
            if forced:
                continue
            # else SHIFT exit
            trades.append({"index":exit_i,"action":"SHIFT_EXIT","price":prices[exit_i]})
            position=0
            i=exit_i+1

    df_trades=pd.DataFrame(trades)
    if df_trades.empty:
        logger.info("[Backtest] => no trades triggered.")
        return df_trades, {}

    # approximate PnL
    open_pos=None
    rets=[]
    for i,row in df_trades.iterrows():
        if row["action"]=="BUY_OPEN":
            open_pos=row
        else:
            if open_pos is not None:
                realized=((row["price"]-open_pos["price"])/open_pos["price"])*100
                rets.append(realized)
                open_pos=None

    if not rets:
        return df_trades,{}

    total_pnl=sum(rets)
    avg_pnl=np.mean(rets)
    stats={
      "num_trades": len(rets),
      "total_pnl_%": total_pnl,
      "avg_pnl_%": avg_pnl
    }
    logger.info(f"[Backtest] => trades={len(rets)}, totalPnL={total_pnl:.2f}%, avgPnL={avg_pnl:.2f}%")
    return df_trades, stats

# ------------------------------------------------------------------------------
# Helper => map symbol => coin_id from 'lunarcrush_data'
# ------------------------------------------------------------------------------
def _map_symbol_to_coinid(db_file:str) -> Dict[str,str]:
    """
    Return e.g. {"ETH":"2","XRP":"3", ...} from 'lunarcrush_data'.
    """
    out={}
    if not os.path.exists(db_file):
        logger.warning("No DB => can't map symbol => coin_id.")
        return out

    conn=sqlite3.connect(db_file)
    try:
        c=conn.cursor()
        q="""
            SELECT symbol, lunarcrush_id
            FROM lunarcrush_data
            WHERE lunarcrush_id IS NOT NULL
            ORDER BY id DESC
        """
        rows=c.execute(q).fetchall()
        for (sym, cid) in rows:
            if sym and cid and sym.upper() not in out:
                out[sym.upper()]=str(cid)
    except Exception as e:
        logger.exception(f"Error mapping => {e}")
    finally:
        conn.close()
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    """
    1) Possibly load aggregator data => cryptopanic_posts aggregator => 'df_cpanic'.
    2) Map symbol => coin_id from 'lunarcrush_data'.
    3) For each pair => find coin_id => load time-series => build features => skip if < MIN_DATA_ROWS
    4) Concatenate => train => backtest => save model => store model_info in JSON
    """
    logger.info("Starting train_model pipeline...")

    # 1) aggregator => cryptopanic_posts daily
    df_cpanic = fetch_cryptopanic_aggregated(DB_FILE)

    # 2) map symbol => coin_id from 'lunarcrush_data'
    coin_map = _map_symbol_to_coinid(DB_FILE)

    all_X=[]
    all_y=[]

    # 3) for each pair => build
    for pair in TRADED_PAIRS:
        symbol = pair.split("/")[0].upper()
        cid = coin_map.get(symbol)
        if not cid:
            logger.warning(f"No coin_id found for symbol={symbol}. Skipping.")
            continue

        df_ts = load_lunarcrush_timeseries(DB_FILE, cid)
        if df_ts.empty:
            logger.warning(f"No timeseries => coin_id={cid} => skip.")
            continue

        # aggregator from snapshot
        df_lc_snap = fetch_lunarcrush_data_for_symbol(DB_FILE, symbol)

        # build features
        X_coin, y_coin = build_features(
            df_ts,
            df_cp=df_cpanic,
            df_lc=df_lc_snap,
            shift_bars=SHIFT_BARS,
            threshold_up=THRESHOLD_UP
        )
        if len(X_coin)<MIN_DATA_ROWS:
            logger.warning(f"coin_id={cid}, after building => only {len(X_coin)} rows => skip (min={MIN_DATA_ROWS})")
            continue

        all_X.append(X_coin)
        all_y.append(y_coin)

    # If all coins were skipped => no final dataset
    if not all_X:
        logger.warning("All coins were skipped => no data => no training done.")
        # We do not forcibly exit. We just do no training.
        return

    # 4) train
    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)
    logger.info(f"Final dataset => X={X_full.shape}, y={y_full.shape}")

    model, final_metric = train_random_forest(X_full, y_full)
    if model is None:
        logger.error("No model => training aborted.")
        return

    # backtest
    df_trades, stats = backtest_prob(X_full, model)
    logger.info(f"[train_model] Backtest => {stats}")

    # Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Saved model => {MODEL_OUTPUT_PATH}")

    # log metric
    if final_metric:
        log_accuracy_to_csv(final_metric)

    # store model info in JSON
    model_info = {
        "training_timestamp": datetime.utcnow().isoformat(),
        "final_metric": final_metric,
        "backtest_stats": stats
    }
    with open(MODEL_INFO_FILE, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Saved model info => {MODEL_INFO_FILE}")


if __name__=="__main__":
    main()
