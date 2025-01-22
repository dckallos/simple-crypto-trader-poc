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
from sklearn.model_selection import train_test_split
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
                AVG(price) as avg_price_lc,  -- or median?
                AVG(volume_24h) as avg_vol_24h,
                AVG(market_cap) as avg_mcap,
                AVG(social_volume_24h) as avg_soc_vol,
                AVG(sentiment) as avg_lc_sent  -- if you're storing that
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
    """
    import pandas as pd
    if df.empty or "last_price" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

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

    # If we want correlation with BTC
    if df_btc is not None and not df_btc.empty:
        # We'll do a naive asof-merge
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
        df["trade_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        # df_cpanic => [news_date, avg_sentiment]
        # rename for join => "news_date" => "trade_date"
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
        # df_lc => [lc_date, galaxy_score, alt_rank, avg_lc_sent, ...]
        df["lc_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        df_lc_ren = df_lc.rename(columns={"lc_date": "lc_date"})
        df = pd.merge(
            df,
            df_lc_ren,
            left_on="lc_date",
            right_on="lc_date",
            how="left"
        )
        # fill numeric columns with 0 or ffill
        for col in ["galaxy_score", "alt_rank", "avg_lc_sent"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

    # Forward/backward fill for any partial merges
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Replace infinite with NaN, then drop if needed
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Construct label => next price > current price
    df["future_price"] = df["last_price"].shift(-1)
    df["label_up"] = (df["future_price"] > df["last_price"]).astype(int)

    # Drop the last row (no future price)
    df.dropna(subset=["future_price"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Potential feature columns
    feature_cols = [
        "feature_price", "feature_ma_3", "feature_spread",
        "vol_change", "rsi", "macd_line", "macd_signal",
        "boll_upper", "boll_lower"
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
# Step 5: The main training routine
# ------------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a random forest classifier on the given features and labels.
    Returns (model, accuracy).
    """
    if X.empty or len(X) < 10:
        logger.warning("Not enough data to train a robust model!")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

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
    4) Combine everything into X_full, y_full, train a RandomForest, save it.
    5) Log or print accuracy.
    """
    # 1) Possibly load BTC data for correlation
    df_btc = load_data_for_btc(DB_FILE, BTC_PAIR)

    # 2) Possibly load aggregator from cryptopanic
    df_cpanic = fetch_cryptopanic_aggregated(DB_FILE)

    # 3) If you want to merge LunarCrush aggregator, you can fetch them on a per-symbol basis
    #    For demonstration, we'll do a dictionary: symbol -> aggregator DataFrame
    #    e.g. "ETH", "SOL", ...
    #    This requires that your "lunarcrush_data" has daily rows. If not, adapt logic.
    lunarcrush_dict = {}

    # We'll guess that pairs have the format "ETH/USD", so symbol is "ETH"
    # We'll do a simple approach: For each pair, parse symbol => build aggregator => store in a dict
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

        # Symbol
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

    # 4) Train the model
    model, accuracy = train_model(X_full, y_full)
    if model is None:
        logger.error("Model training returned None. Exiting.")
        return

    logger.info(f"Validation Accuracy = {accuracy:.4f}")

    # 5) Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")

    # 6) Log accuracy
    log_accuracy_to_csv(accuracy)
    logger.info(f"Appended accuracy={accuracy:.4f} to {METRICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
