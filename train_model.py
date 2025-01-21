# ==============================================================================
# FILE: train_model.py
# ==============================================================================
"""
train_model.py

1) Reads historical price data from the 'price_history' table in 'trades.db' for multiple pairs.
2) Constructs basic features and labels for each pair; combines them.
3) Trains a simple scikit-learn model (RandomForest).
4) Saves the trained model to disk ("trained_model.pkl").
5) Optionally logs the model's validation accuracy to "training_metrics.csv".

We expand build_features_and_labels() to compute:
- Volume changes
- RSI
- MACD
- Bollinger bands
- Rolling correlation with BTC
- Merging optional CryptoPanic sentiment (if aggregated by day).

All existing code is preserved.
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "trades.db"
MODEL_OUTPUT_PATH = "trained_model.pkl"
METRICS_OUTPUT_FILE = "training_metrics.csv"

# NEW: We'll define a BTC symbol to unify correlation logic
BTC_PAIR = "XBT/USD"

# Load pairs from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
TRADED_PAIRS = config.get("traded_pairs", [])

# ------------------------------------------------------------------------------
# Step 1: Load Data from DB
# ------------------------------------------------------------------------------
def load_data_for_pair(db_file: str, pair: str) -> pd.DataFrame:
    """
    Loads data from 'price_history' for a single pair, returning a DataFrame with:
        [timestamp, pair, bid_price, ask_price, last_price, volume]
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

# NEW: Load aggregated CryptoPanic sentiment, if any
def load_cryptopanic_sentiment(db_file: str) -> pd.DataFrame:
    """
    Example: Suppose we stored one row per hour/day in a separate aggregator table,
    or we want to transform the raw 'cryptopanic_news' data into a time series.
    For now, let's assume we do something naive: average sentiment per day.

    This function returns a DataFrame with columns [date, avg_sentiment].
    Then we'll merge that into our main price DataFrame if timestamps match up.
    """
    conn = sqlite3.connect(db_file)
    try:
        # Suppose we do daily aggregated sentiment, e.g. group by date
        query = """
            SELECT
                DATE(timestamp, 'unixepoch') as news_date,
                AVG(sentiment_score) as avg_sentiment
            FROM cryptopanic_news
            GROUP BY news_date
            ORDER BY news_date ASC
        """
        df_sent = pd.read_sql_query(query, conn)
        return df_sent
    except Exception as e:
        logger.exception(f"Error reading cryptopanic data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------------------------------------------------------------
# Step 2: Feature Engineering
# ------------------------------------------------------------------------------
def build_features_and_labels(df: pd.DataFrame, df_btc: pd.DataFrame=None, df_news: pd.DataFrame=None) -> (pd.DataFrame, pd.Series):
    """
    Given a DataFrame of price data for one pair, create features and a target label.

    Add:
      - Volume changes
      - RSI
      - MACD
      - Bollinger
      - correlation with BTC (if df_btc is provided)
      - optional sentiment from cryptopanic (if df_news is provided)

    :param df: DataFrame of a single pair: [timestamp, pair, bid_price, ask_price, last_price, volume]
    :param df_btc: DataFrame of the reference BTC data, if correlation is needed.
    :param df_news: DataFrame of aggregated news sentiment by day, if using CryptoPanic data.
    :return: (X, y)
    """
    if df.empty or "last_price" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic features from before
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
    sma20 = df["last_price"].rolling(window=20).mean()
    std20 = df["last_price"].rolling(window=20).std()
    df["boll_upper"] = sma20 + (2 * std20)
    df["boll_lower"] = sma20 - (2 * std20)

    # If we have df_btc for correlation, merge or align timestamps
    if df_btc is not None and not df_btc.empty:
        # Merge df_btc on timestamp
        # rename last_price to something like btc_price
        df_btc_ren = df_btc[["timestamp", "last_price"]].copy()
        df_btc_ren.rename(columns={"last_price": "btc_price"}, inplace=True)
        # Merge on timestamp
        df_merged = pd.merge_asof(
            df.sort_values("timestamp"),
            df_btc_ren.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=30  # e.g. 30 seconds tolerance
        )
        df = df_merged

        # Now compute rolling correlation of pair's price vs. btc_price
        df["corr_with_btc"] = df["last_price"].rolling(window=30).corr(df["btc_price"])

    # If we have df_news, merge it by date
    if df_news is not None and not df_news.empty:
        # We'll create a date column in df
        df["trade_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        df_news["news_date"] = pd.to_datetime(df_news["news_date"]).dt.date
        # Merge on these date columns
        df = pd.merge(
            df,
            df_news.rename(columns={"news_date": "trade_date"}),
            on="trade_date",
            how="left"
        )
        # rename avg_sentiment to cryptopanic_sent, fill missing with 0
        df["avg_sentiment"] = df["avg_sentiment"].fillna(0)

    # Forward/backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Construct label: next price > current price
    df["future_price"] = df["last_price"].shift(-1)
    df["label_up"] = (df["future_price"] > df["last_price"]).astype(int)

    df.dropna(subset=["future_price"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Potential feature columns
    feature_cols = [
        "feature_price", "feature_ma_3", "feature_spread",
        "vol_change", "rsi", "macd_line", "macd_signal",
        "boll_upper", "boll_lower"
    ]
    # If we have correlation
    if "corr_with_btc" in df.columns:
        feature_cols.append("corr_with_btc")
    # If we have CryptoPanic sentiment
    if "avg_sentiment" in df.columns:
        feature_cols.append("avg_sentiment")

    X = df[feature_cols]
    y = df["label_up"]

    return X, y

# ------------------------------------------------------------------------------
# Step 3: Training Routine
# ------------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    if X.empty or len(X) < 10:
        logger.warning("Not enough data to train a robust model!")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc

def log_accuracy_to_csv(accuracy: float):
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
    # 1) Load BTC reference data if we want correlation
    df_btc = load_data_for_pair(DB_FILE, BTC_PAIR)  # e.g. "XBT/USD"
    # 2) Load CryptoPanic aggregated sentiment
    df_news = load_cryptopanic_sentiment(DB_FILE)

    all_X = []
    all_y = []

    # Load each pair
    for pair in TRADED_PAIRS:
        if pair == BTC_PAIR:
            # We can skip or keep it if you also want to classify BTC
            logger.info(f"Pair={pair} is BTC itself. We'll still process or skip it.")
            # We'll still do it but it's optional
        df = load_data_for_pair(DB_FILE, pair)
        if df.empty:
            logger.warning(f"No data for {pair}. Skipping.")
            continue

        # Build features with optional BTC for correlation, plus optional news
        X_pair, y_pair = build_features_and_labels(df, df_btc, df_news)
        if X_pair.empty:
            logger.warning(f"No valid features after building labels for {pair}. Skipping.")
            continue

        all_X.append(X_pair)
        all_y.append(y_pair)

    if not all_X:
        logger.error("No data retrieved for any pairs. Exiting.")
        return

    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)
    logger.info(f"Combined dataset size: X={X_full.shape}, y={y_full.shape}")

    model, accuracy = train_model(X_full, y_full)
    if model is None:
        logger.error("Model training returned None. Exiting.")
        return

    logger.info(f"Validation Accuracy = {accuracy:.4f}")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")

    log_accuracy_to_csv(accuracy)
    logger.info(f"Appended accuracy={accuracy:.4f} to {METRICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
