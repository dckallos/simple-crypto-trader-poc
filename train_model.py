# ==============================================================================
# FILE: train_model.py
# ==============================================================================
"""
train_model.py

1) Reads historical price data from the 'price_history' table in 'trades.db' for multiple pairs.
2) Constructs basic features and labels for each pair; combines them.
3) Trains a simple scikit-learn model (e.g., a RandomForestClassifier).
4) Saves the trained model to disk (by default: "trained_model.pkl").
5) Optionally logs the model's validation accuracy each time to "training_metrics.csv".

Usage:
    python train_model.py

Notes:
- In a real scenario, you'd likely create more sophisticated features
  (moving averages, indicators, etc.) and a more carefully chosen label
  (future return, etc.).
- We remove deprecated fillna(method="ffill"/"bfill") usage by using df.ffill()
  and df.bfill() instead.
- If you want to skip logging to a CSV, just remove or comment out the
  log_accuracy_to_csv(...) call.
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

# If you have a db.py with init_db or other DB utilities, you can import them, but
# it's not strictly necessary unless you want to ensure DB creation from here.
# from db import init_db

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DB_FILE = "trades.db"                # The same DB that main.py uses
MODEL_OUTPUT_PATH = "trained_model.pkl"
METRICS_OUTPUT_FILE = "training_metrics.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pairs from config.yaml (in the same folder)
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

# ------------------------------------------------------------------------------
# Step 2: Feature Engineering
# ------------------------------------------------------------------------------
def build_features_and_labels(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Given a DataFrame of price data, create simple features and a target label.

    For demonstration, we define a trivial label = whether the price 1 step
    in the future is higher than the current price. In reality, you'd do
    something more robust (like a future return threshold, multi-step ahead
    prediction, etc.).

    :param df: DataFrame with columns at least:
               [timestamp, last_price, bid_price, ask_price, volume]
    :return: (X, y)
             X => DataFrame of features
             y => Series (binary label: 1 => up, 0 => down or same)
    """
    if df.empty or "last_price" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Sort ascending by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Example features
    df["feature_price"] = df["last_price"]
    df["feature_ma_3"] = df["last_price"].rolling(window=3).mean()
    df["feature_spread"] = df["ask_price"] - df["bid_price"]

    # Forward/backward fill instead of deprecated fillna(method=...)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Label: next price > current price?
    df["future_price"] = df["last_price"].shift(-1)
    df["label_up"] = (df["future_price"] > df["last_price"]).astype(int)

    # Drop the last row (no future_price)
    df.dropna(subset=["future_price"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Our X, y
    feature_cols = ["feature_price", "feature_ma_3", "feature_spread"]
    X = df[feature_cols]
    y = df["label_up"]

    return X, y

# ------------------------------------------------------------------------------
# Step 3: Training Routine
# ------------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a random forest classifier on the given features and labels.

    :param X: Features DataFrame
    :param y: Labels Series
    :return: (model, accuracy) or (None, None) if insufficient data.
    """
    if X.empty or len(X) < 10:
        logger.warning("Not enough data to train a robust model!")
        return None, None

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc

# ------------------------------------------------------------------------------
# Step 4: (Optional) Log Accuracy Over Time
# ------------------------------------------------------------------------------
def log_accuracy_to_csv(accuracy: float):
    """
    Appends a new line with current timestamp and accuracy to METRICS_OUTPUT_FILE,
    so you can track how model performance changes over time.

    You can remove/comment this out if you don't want to track accuracy over time.
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If file doesn't exist, write header
    if not os.path.exists(METRICS_OUTPUT_FILE):
        with open(METRICS_OUTPUT_FILE, "w") as f:
            f.write("timestamp,accuracy\n")

    # Append a new line
    with open(METRICS_OUTPUT_FILE, "a") as f:
        f.write(f"{timestamp},{accuracy:.4f}\n")

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    # Step A) Gather data for all pairs
    all_X = []
    all_y = []

    for pair in TRADED_PAIRS:
        df = load_data_for_pair(DB_FILE, pair)
        if df.empty:
            logger.warning(f"No data found for pair={pair}. Skipping.")
            continue

        X_pair, y_pair = build_features_and_labels(df)
        if X_pair.empty:
            logger.warning(f"No valid features after building labels for {pair}. Skipping.")
            continue

        # Optionally store the pair name in features to differentiate pairs (e.g. one-hot)
        # X_pair["pair"] = pair

        all_X.append(X_pair)
        all_y.append(y_pair)

    if not all_X:
        logger.error("No data retrieved for any pairs. Exiting.")
        return

    # Combine all pairs into a single dataset
    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)

    logger.info(f"Combined dataset size: X={X_full.shape}, y={y_full.shape}")

    # Step B) Train the model
    model, accuracy = train_model(X_full, y_full)
    if model is None:
        logger.error("Model training returned None (insufficient data). Exiting.")
        return

    logger.info(f"Validation Accuracy = {accuracy:.4f}")

    # Step C) Save the model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")

    # Step D) Optional: log accuracy to CSV
    log_accuracy_to_csv(accuracy)
    logger.info(f"Appended accuracy={accuracy:.4f} to {METRICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
