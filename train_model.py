# ==============================================================================
# FILE: train_model.py
# ==============================================================================
"""
train_model.py

1) Reads historical price data from the 'price_history' table in 'trades.db'
   for multiple pairs.
2) Constructs advanced features (Volume changes, RSI, MACD, Bollinger, correlation, etc.).
3) Optionally merges aggregator data from:
   - CryptoPanic (daily sentiment)
   - LunarCrush (galaxy_score, alt_rank, etc.)
4) Combines all pairs' data into one dataset, trains a scikit-learn model, saves to disk.
5) Optionally logs the model's validation accuracy to "training_metrics.csv".

UPDATED ENHANCEMENTS:
- Shifted label to look 3 bars ahead with a +1% threshold
  (instead of next-bar up).
- Added hour_of_day, day_of_week, rolling 10-bar volatility & volume
  as new features.
- Incorporated a small GridSearchCV to tune n_estimators & max_depth
  for the RandomForest.

You can run:
    python train_model.py

Adapt logic if you want to do full merges of time-series data from LunarCrush
instead of daily or snapshot data.
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DB_FILE = "trades.db"                   # The same DB as main.py uses
MODEL_OUTPUT_PATH = "trained_model.pkl" # Where we save the scikit model
METRICS_OUTPUT_FILE = "training_metrics.csv"

# If you want correlation with BTC, define the pair used in 'price_history' for BTC
BTC_PAIR = "XBT/USD"

# Load pairs from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
TRADED_PAIRS = config.get("traded_pairs", [])

# We will define how many bars ahead we look, and the % threshold for "up"
SHIFT_BARS = 3
THRESHOLD_UP = 0.01  # +1%

# ------------------------------------------------------------------------------
# Step 1: Load Data from DB
# ------------------------------------------------------------------------------
def load_data_for_pair(db_file: str, pair: str) -> pd.DataFrame:
    """
    Loads data from 'price_history' for a single pair, returning columns:
        [timestamp, pair, bid_price, ask_price, last_price, volume].
    Sorted ascending by timestamp. Empty if no data or error.
    """
    if not os.path.exists(db_file):
        logger.error(f"Database file {db_file} not found.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_file)
    try:
        query = f"""
            SELECT timestamp, pair, bid_price, ask_price, last_price, volume
            FROM price_history
            WHERE pair='{pair}'
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.exception(f"Error reading from DB for pair={pair}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def load_data_for_btc(db_file: str, btc_pair: str = BTC_PAIR) -> pd.DataFrame:
    """
    A convenience function to load BTC data if you want correlation.
    Same logic as load_data_for_pair but singled out for clarity.
    """
    if not btc_pair:
        return pd.DataFrame()
    return load_data_for_pair(db_file, btc_pair)

# ------------------------------------------------------------------------------
# Step 2: CryptoPanic Aggregator (Daily) - Merging
# ------------------------------------------------------------------------------
def fetch_cryptopanic_aggregated(db_file: str) -> pd.DataFrame:
    """
    Example aggregator that returns daily average sentiment from the
    'cryptopanic_news' table. If you have a different aggregator table, adapt.
    We'll produce columns: [news_date, avg_sentiment].
    """
    conn = sqlite3.connect(db_file)
    try:
        query = """
            SELECT
                DATE(timestamp, 'unixepoch') as news_date,
                AVG(sentiment_score) as avg_sentiment
            FROM cryptopanic_news
            GROUP BY news_date
            ORDER BY news_date ASC
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.exception(f"Error reading cryptopanic aggregator: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Step 3: LunarCrush Aggregator (Daily or Single Snapshot)
# ------------------------------------------------------------------------------
def fetch_lunarcrush_data_for_symbol(db_file: str, symbol: str) -> pd.DataFrame:
    """
    Example approach: If your 'lunarcrush_data' table has 1 row per day or
    multiple daily snapshots, you can transform it to daily data: [date, galaxy_score, alt_rank, etc.].
    Or if you store everything with a timestamp but only 1 snapshot daily, we do a daily grouping.

    We'll do something naive: just get the daily approach or last approach for the symbol.
    For a robust approach, you'd store a time series. For demonstration, we'll produce:
      [lc_date, galaxy_score, alt_rank, sentiment, ...].
    """
    conn = sqlite3.connect(db_file)
    try:
        # Suppose you do daily grouping by date(timestamp).
        # Or you might do last known row if you don't store daily.
        query = f"""
            SELECT
                DATE(timestamp, 'unixepoch') as lc_date,
                AVG(galaxy_score) as galaxy_score,
                AVG(alt_rank) as alt_rank,
                AVG(price) as avg_price_lc,
                AVG(volume_24h) as avg_vol_24h,
                AVG(market_cap) as avg_mcap,
                AVG(social_volume_24h) as avg_soc_vol,
                AVG(sentiment) as avg_lc_sent
            FROM lunarcrush_data
            WHERE symbol='{symbol}'
            GROUP BY lc_date
            ORDER BY lc_date
        """
        df_lc = pd.read_sql_query(query, conn)
        return df_lc
    except Exception as e:
        logger.exception(f"Error reading lunarcrush data for symbol={symbol}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Step 4: Build Features, Possibly Merge BTC correlation, CryptoPanic daily sentiment, LunarCrush, etc.
# ------------------------------------------------------------------------------
def build_features_and_labels(
    df: pd.DataFrame,
    df_btc: pd.DataFrame = None,
    df_cpanic: pd.DataFrame = None,
    df_lc: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build advanced features for a single pair. Then merges optional:
      - BTC correlation
      - CryptoPanic aggregator daily sentiment
      - LunarCrush aggregator daily approach (galaxy_score, alt_rank, etc.)

    Returns (X, y) => Feature DataFrame, label Series

    UPDATES:
    - Instead of next price > current price, we do a SHIFT_BARS=3 look-ahead
      and require a +1% threshold for label=1.
    - Added hour_of_day, day_of_week, rolling volatility, rolling volume.
    """

    import pandas as pd
    if df.empty or "last_price" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------------------------
    # ADD: time-based columns
    # --------------------------------------------------------------------------
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.weekday  # Monday=0, Sunday=6

    # Basic Indicators
    df["feature_price"] = df["last_price"]
    df["feature_ma_3"] = df["last_price"].rolling(window=3).mean()
    df["feature_spread"] = df["ask_price"] - df["bid_price"]

    # Volume change
    df["vol_change"] = df["volume"].pct_change().fillna(0)

    # RSI
    window_length = 14
    close_delta = df["last_price"].diff()
    gain = close_delta.clip(lower=0)
    loss = (-1 * close_delta.clip(upper=0))
    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["last_price"].ewm(span=12).mean()
    ema26 = df["last_price"].ewm(span=26).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9).mean()

    # Bollinger
    sma20 = df["last_price"].rolling(20).mean()
    std20 = df["last_price"].rolling(20).std()
    df["boll_upper"] = sma20 + (2 * std20)
    df["boll_lower"] = sma20 - (2 * std20)

    # --------------------------------------------------------------------------
    # ADD: rolling volatility & rolling average volume (window=10)
    # --------------------------------------------------------------------------
    df["rolling_vol_10"] = df["last_price"].rolling(10).std()
    df["rolling_volume_10"] = df["volume"].rolling(10).mean()

    # If we want correlation with BTC
    if df_btc is not None and not df_btc.empty:
        df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
        df_btc_ren = df_btc[["timestamp", "last_price"]].rename(columns={"last_price": "btc_price"})
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_btc_ren,
            on="timestamp",
            direction="nearest",
            tolerance=30
        )
        df["corr_with_btc"] = df["last_price"].rolling(30).corr(df["btc_price"])

    # If we want to merge CryptoPanic aggregator daily => we create date col
    if df_cpanic is not None and not df_cpanic.empty:
        df["trade_date"] = df["datetime"].dt.date
        df_cpanic_ren = df_cpanic.rename(columns={"news_date": "trade_date"})
        df = pd.merge(
            df,
            df_cpanic_ren,
            on="trade_date",
            how="left"
        )
        df["avg_sentiment"] = df["avg_sentiment"].fillna(0)

    # If we want to merge LunarCrush aggregator => we also create date col
    if df_lc is not None and not df_lc.empty:
        df["lc_date"] = df["datetime"].dt.date
        df_lc_ren = df_lc.rename(columns={"lc_date": "lc_date"})
        df = pd.merge(
            df,
            df_lc_ren,
            left_on="lc_date",
            right_on="lc_date",
            how="left"
        )
        for col in ["galaxy_score", "alt_rank", "avg_lc_sent"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

    # Forward/backward fill for any partial merges
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Replace infinite with NaN, then drop if needed
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --------------------------------------------------------------------------
    # OLD LOGIC (commented):
    # df["future_price"] = df["last_price"].shift(-1)
    # df["label_up"] = (df["future_price"] > df["last_price"]).astype(int)
    #
    # NEW LOGIC: SHIFT_BARS=3, +1% threshold
    # --------------------------------------------------------------------------
    # We look 3 bars ahead, find the max in that window, measure % difference
    df["future_max"] = df["last_price"].rolling(SHIFT_BARS).max().shift(-SHIFT_BARS)
    df["pct_future_gain"] = (df["future_max"] - df["last_price"]) / (df["last_price"] + 1e-9)
    df["label_up"] = (df["pct_future_gain"] >= THRESHOLD_UP).astype(int)

    # Remove the rows where we don't have a valid future window
    df.dropna(subset=["future_max"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Potential feature columns
    feature_cols = [
        # Original columns
        "feature_price", "feature_ma_3", "feature_spread",
        "vol_change", "rsi", "macd_line", "macd_signal",
        "boll_upper", "boll_lower",
        # New
        "hour_of_day", "day_of_week",
        "rolling_vol_10", "rolling_volume_10"
    ]
    # If correlation was used
    if "corr_with_btc" in df.columns:
        feature_cols.append("corr_with_btc")
    # If cryptopanic aggregator is used
    if "avg_sentiment" in df.columns:
        feature_cols.append("avg_sentiment")
    # If lunarcrush aggregator is used
    if "galaxy_score" in df.columns:
        feature_cols.append("galaxy_score")
    if "alt_rank" in df.columns:
        feature_cols.append("alt_rank")
    if "avg_lc_sent" in df.columns:
        feature_cols.append("avg_lc_sent")

    # Drop rows missing essential features
    df.dropna(subset=feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[feature_cols]
    y = df["label_up"]
    return X, y

# ------------------------------------------------------------------------------
# Step 5: The main training routine (with GridSearchCV)
# ------------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a random forest classifier on the given features and labels.
    Returns (model, accuracy).

    UPDATED:
    - We now do a small GridSearchCV with a TimeSeriesSplit to find
      good hyperparams for the random forest (n_estimators, max_depth).
    """
    if X.empty or len(X) < 10:
        logger.warning("Not enough data to train a robust model!")
        return None, None

    # Instead of a single train_test_split, let's do a time series approach.
    # However, for demonstration, we will do a final fit on the entire dataset
    # after we pick the best hyperparams from the CV.
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    logger.info(f"Best params from GridSearchCV: {grid.best_params_}")

    # We can compute an "overall" accuracy via a final train/test or
    # via the best_score_ from cross-validation:
    cv_best_score = grid.best_score_
    logger.info(f"Best CV accuracy: {cv_best_score:.4f}")

    # If you still want a final hold-out approach, you can do:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Hold-out Accuracy = {acc:.4f}")

    # Return the best_model and hold-out accuracy
    return best_model, acc

def log_accuracy_to_csv(accuracy: float):
    """
    Optionally log the accuracy to a CSV file over time.
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(METRICS_OUTPUT_FILE):
        with open(METRICS_OUTPUT_FILE, "w") as f:
            f.write("timestamp,accuracy\n")

    with open(METRICS_OUTPUT_FILE, "a") as f:
        f.write(f"{timestamp},{accuracy:.4f}\n")

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    """
    1) Load BTC data if desired for correlation.
    2) Load aggregator from CryptoPanic daily if desired.
    3) For each pair in config.yaml, load 'price_history', build features, possibly merges aggregator data.
    4) Combine everything into X_full, y_full, do hyperparameter search + train a RandomForest, save it.
    5) Log or print final accuracy.
    """
    # 1) Possibly load BTC data for correlation
    df_btc = load_data_for_btc(DB_FILE, BTC_PAIR)

    # 2) Possibly load aggregator from cryptopanic
    df_cpanic = fetch_cryptopanic_aggregated(DB_FILE)

    # 3) If you want to merge LunarCrush aggregator, fetch them on a per-symbol basis
    lunarcrush_dict = {}
    unique_symbols = set(pair.split("/")[0].upper() for pair in TRADED_PAIRS)
    for sym in unique_symbols:
        df_lc = fetch_lunarcrush_data_for_symbol(DB_FILE, sym)
        lunarcrush_dict[sym] = df_lc

    all_X = []
    all_y = []

    for pair in TRADED_PAIRS:
        logger.info(f"Processing pair={pair}...")

        df = load_data_for_pair(DB_FILE, pair)
        if df.empty:
            logger.warning(f"No data for {pair}. Skipping.")
            continue

        symbol = pair.split("/")[0].upper()
        df_lc_for_symbol = lunarcrush_dict.get(symbol, pd.DataFrame())

        # Build features
        X_pair, y_pair = build_features_and_labels(
            df,
            df_btc=df_btc,
            df_cpanic=df_cpanic,
            df_lc=df_lc_for_symbol
        )
        if X_pair.empty:
            logger.warning(f"No valid features after building labels for {pair}. Skipping.")
            continue

        all_X.append(X_pair)
        all_y.append(y_pair)

    if not all_X:
        logger.error("No data retrieved for any pairs. Exiting.")
        return

    # Combine all pairs
    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)
    logger.info(f"Combined dataset size: X={X_full.shape}, y={y_full.shape}")

    # 4) Train the model with hyperparam search
    model, final_accuracy = train_model(X_full, y_full)
    if model is None:
        logger.error("Model training returned None. Exiting.")
        return

    logger.info(f"Final Validation Accuracy = {final_accuracy:.4f}")

    # 5) Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")

    # 6) Log accuracy
    log_accuracy_to_csv(final_accuracy)
    logger.info(f"Appended accuracy={final_accuracy:.4f} to {METRICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
