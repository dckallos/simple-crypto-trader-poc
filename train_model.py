# ==============================================================================
# FILE: train_model.py
# ==============================================================================
"""
train_model.py

A minimal example of how to:
1. Read historical price data from the 'price_history' table in 'trades.db'.
2. Construct basic features and labels for a predictive model.
3. Train a simple scikit-learn model (e.g., a RandomForestClassifier).
4. Save the trained model to disk (e.g., using joblib).

You can run this script from the command line:
    python train_model.py

This example is simplified for demonstration. In a real scenario, you'll likely
create more sophisticated features (moving averages, indicators, etc.) and
a more carefully chosen label (future price movement, returns, etc.).
"""

import sqlite3
import os
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from db import init_db

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DB_FILE = "trades.db"  # The same DB that main.py uses
MODEL_OUTPUT_PATH = "trained_model.pkl"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Fetch Data from DB
# ------------------------------------------------------------------------------
def load_data_from_db(db_file=DB_FILE, pair="ETH/USD"):
    """
    Loads data from 'price_history' table for the specified pair.

    Returns a Pandas DataFrame with columns:
        timestamp, pair, bid_price, ask_price, last_price, volume
    If there's no data, returns an empty DataFrame.
    """
    if not os.path.exists(db_file):
        logger.error(f"Database file {db_file} not found.")
        return pd.DataFrame()  # empty

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
        logger.exception(f"Error reading from DB: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------------------
def build_features_and_labels(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Given a dataframe of price data, create simple features and a target label.

    For demonstration, we define a trivial "label" = whether the price 1 step in
    the future is higher than the current price. In reality, you'd do something
    more robust (like a future return threshold, multi-step ahead prediction, etc.)

    Returns:
        X: a DataFrame of features
        y: a Series of labels (1 => price up, 0 => price down or same)
    """
    if df.empty or "last_price" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Sort by timestamp to ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Example feature 1: "last_price" as the direct input
    df["feature_price"] = df["last_price"]

    # Example feature 2: rolling mean over a small window
    df["feature_ma_3"] = df["last_price"].rolling(window=3).mean()

    # Example feature 3: difference between bid/ask
    df["feature_spread"] = df["ask_price"] - df["bid_price"]

    # Fill NaNs from rolling computations
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Construct label: compare next price to current price
    # Shift last_price by -1 to get future price
    df["future_price"] = df["last_price"].shift(-1)
    df["label_up"] = (df["future_price"] > df["last_price"]).astype(int)

    # Drop the last row because it won't have a future price
    df = df.dropna(subset=["future_price"]).reset_index(drop=True)

    # Our feature set (X) and label (y)
    feature_cols = ["feature_price", "feature_ma_3", "feature_spread"]
    X = df[feature_cols]
    y = df["label_up"]

    return X, y


# ------------------------------------------------------------------------------
# Training Routine
# ------------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a random forest classifier on the given features and labels.

    :param X: Features DataFrame
    :param y: Label Series
    :return: Trained model
    """
    if X.empty or len(X) < 10:
        logger.warning("Not enough data to train a robust model!")
        return None

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Create and train the model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Validation Accuracy = {acc:.4f}")

    return model


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    # 1) Load data from DB
    df = load_data_from_db(DB_FILE, pair="ETH/USD")
    if df.empty:
        logger.error("No data retrieved from DB. Exiting.")
        return

    logger.info(f"Loaded {len(df)} rows of price data for 'ETH/USD'.")

    # 2) Build features and labels
    X, y = build_features_and_labels(df)
    if X.empty:
        logger.error("Feature DataFrame is empty. Exiting.")
        return

    logger.info(f"Feature shape: {X.shape}, Label shape: {y.shape}")

    # 3) Train model
    model = train_model(X, y)
    if model is None:
        logger.error("Model training returned None. Exiting.")
        return

    # 4) Save model to disk
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
