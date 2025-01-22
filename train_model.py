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
5) Optionally logs performance metrics (accuracy, F1, etc.) to "training_metrics.csv".

NEW ADDITION: An ADVANCED Backtest Logic
- Replaces the naive SHIFT_BARS approach with:
  (a) Position tracking (long/short/flat)
  (b) Slippage & fees
  (c) Stop-loss & take-profit
  (d) Partial exits
  (e) Advanced metrics (max drawdown, Sharpe, etc.)

Usage:
    python train_model.py
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
from typing import Tuple
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DB_FILE = "trades.db"                    # The same DB as main.py uses
MODEL_OUTPUT_PATH = "trained_model.pkl"  # Where we save the scikit model
METRICS_OUTPUT_FILE = "training_metrics.csv"

# If you want correlation with BTC, define the pair used in 'price_history' for BTC
BTC_PAIR = "XBT/USD"

# Load pairs from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
TRADED_PAIRS = config.get("traded_pairs", [])

# SHIFT_BARS and THRESHOLD_UP are used to define how we label "up"
SHIFT_BARS = 25
THRESHOLD_UP = 0.001  # +1%

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
    multiple daily snapshots, you can transform it to daily data: [lc_date, galaxy_score, alt_rank, etc.].
    Or if you store everything with a timestamp but only 1 snapshot daily, we do a daily grouping.

    We'll do something naive: just get the daily approach or last approach for the symbol.
    For a robust approach, you'd store a time series. For demonstration, we'll produce:
      [lc_date, galaxy_score, alt_rank, sentiment, ...].
    """
    conn = sqlite3.connect(db_file)
    try:
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

    SHIFT_BARS=3, require +1% for label_up=1, to handle short-term labeling.
    """

    import pandas as pd
    if df.empty or "last_price" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.weekday

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

    # Rolling vol & rolling volume
    df["rolling_vol_10"] = df["last_price"].rolling(10).std()
    df["rolling_volume_10"] = df["volume"].rolling(10).mean()

    # Merge correlation with BTC
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

    # Merge CryptoPanic aggregator
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

    # Merge LunarCrush aggregator
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

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Label: SHIFT_BARS=3, require +1% for "up"
    df["future_max"] = df["last_price"].rolling(SHIFT_BARS).max().shift(-SHIFT_BARS)
    df["pct_future_gain"] = (df["future_max"] - df["last_price"]) / (df["last_price"] + 1e-9)
    df["label_up"] = (df["pct_future_gain"] >= THRESHOLD_UP).astype(int)

    df.dropna(subset=["future_max"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Feature columns
    feature_cols = [
        "feature_price", "feature_ma_3", "feature_spread",
        "vol_change", "rsi", "macd_line", "macd_signal",
        "boll_upper", "boll_lower",
        "hour_of_day", "day_of_week",
        "rolling_vol_10", "rolling_volume_10"
    ]
    if "corr_with_btc" in df.columns:
        feature_cols.append("corr_with_btc")
    if "avg_sentiment" in df.columns:
        feature_cols.append("avg_sentiment")
    if "galaxy_score" in df.columns:
        feature_cols.append("galaxy_score")
    if "alt_rank" in df.columns:
        feature_cols.append("alt_rank")
    if "avg_lc_sent" in df.columns:
        feature_cols.append("avg_lc_sent")

    df.dropna(subset=feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[feature_cols]
    y = df["label_up"]
    return X, y

# ------------------------------------------------------------------------------
# Step 5: Training routine with class imbalance handling
# ------------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a RandomForest with class_weight balancing options.
    Uses TimeSeriesSplit + GridSearchCV with 'f1_macro' scoring.
    Then does a final forward-based hold-out for the last 20% of data.
    Logs accuracy, balanced accuracy, F1, confusion matrix, etc.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    if X.empty or len(X) < 10:
        logger.warning("Not enough data to train a robust model!")
        return None, None

    # 1) Inspect label distribution
    logger.info("Label distribution (0 vs 1):\n" + str(y.value_counts(normalize=True)))

    # 2) TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # 3) Hyperparam grid with class_weight
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'class_weight': [None, 'balanced']
    }

    base_rf = RandomForestClassifier(random_state=42)
    # We use 'f1_macro' to give both classes equal importance
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=tscv,
        n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    logger.info(f"Best params from GridSearchCV: {grid.best_params_}")
    logger.info(f"Best CV (f1_macro) score: {grid.best_score_:.4f}")

    # 4) Final forward-based hold-out (last 20%)
    hold_out_size = int(0.2 * len(X))
    X_train = X.iloc[:-hold_out_size]
    y_train = y.iloc[:-hold_out_size]
    X_test = X.iloc[-hold_out_size:]
    y_test = y.iloc[-hold_out_size:]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Evaluate on hold-out
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    logger.info(f"Final hold-out Accuracy = {acc:.4f}")
    logger.info(f"Final hold-out Balanced Accuracy = {bal_acc:.4f}")
    logger.info(f"Final hold-out F1 (macro) = {f1_macro:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix (final hold-out):\n{cm}")

    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")

    return best_model, f1_macro

def log_accuracy_to_csv(accuracy: float):
    """
    Optionally log the accuracy (or any other metric) to a CSV file over time.
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(METRICS_OUTPUT_FILE):
        with open(METRICS_OUTPUT_FILE, "w") as f:
            f.write("timestamp,accuracy\n")

    with open(METRICS_OUTPUT_FILE, "a") as f:
        f.write(f"{timestamp},{accuracy:.4f}\n")

# ------------------------------------------------------------------------------
# Step 6: ADVANCED Backtest Logic
# ------------------------------------------------------------------------------
def backtest_model(
    X: pd.DataFrame,
    model,
    buy_threshold=0.6,
    sell_threshold=0.4,
    stop_loss_pct=0.03,
    take_profit_pct=0.05,
    max_hold_bars=SHIFT_BARS,
    slippage_rate=0.0005,
    fee_rate=0.001
):
    """
    A more realistic backtest that simulates:
      - Long & Short entries based on probability (p_up > buy_threshold => go long,
        p_up < sell_threshold => go short).
      - Maintains a single position at a time (position_size).
      - Stop-loss & take-profit if price moves beyond thresholds.
      - Partial exit logic:
         if we reach half the take_profit early, we close 50% to lock in gains.
      - Slippage & fees on each trade.
      - Advanced metrics: total PnL, max drawdown, Sharpe ratio.

    :param X: DataFrame of features in chronological order.
              Must have "feature_price" for entry/exit prices.
    :param model: Trained scikit-learn model that supports predict_proba.
    :param buy_threshold: Probability above which we open a long position if flat.
    :param sell_threshold: Probability below which we open a short position if flat.
    :param stop_loss_pct: E.g. 0.03 => stop if trade is -3%.
    :param take_profit_pct: E.g. 0.05 => fully close if +5%.
    :param max_hold_bars: Force exit after these many bars if still open.
    :param slippage_rate: e.g. 0.0005 => 0.05%
    :param fee_rate: e.g. 0.001 => 0.1%

    :return:
      df_trades: each row is an entry/exit event or partial exit.
      advanced_stats: dict with final metrics (total_pnl, max_drawdown, sharpe, etc.)
    """
    import pandas as pd

    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba. We'll do predict => 0/1 only.")
        # We'll create a fallback function
        def prob_func(row_df):
            pred = model.predict(row_df)[0]
            return float(pred)
    else:
        def prob_func(row_df):
            # row_df is 1xN DF of features
            proba = model.predict_proba(row_df)[0]
            # Probability of "1"
            return proba[1] if len(proba) > 1 else float(model.classes_[0] == 1)

    # We assume X is sorted by time ascending. If not, sort it first.
    X = X.reset_index(drop=True)

    # Single "position" approach: 0=flat, >0=long, <0=short.
    position_size = 0
    entry_price = 0.0
    partial_exit_done = False  # track if we've done the 50% partial exit

    trades = []

    i = 0
    while i < len(X):
        row = X.iloc[i]
        row_df = row.to_frame().T  # 1-row DF with correct feature names
        current_price = row["feature_price"]

        if position_size == 0:
            # we're flat, see if we open long or short
            p_up = prob_func(row_df)
            if p_up > buy_threshold:
                # go LONG
                # Slippage on buy
                buy_price = current_price * (1 + slippage_rate)
                fee_buy = buy_price * fee_rate

                position_size = 1  # 1 means fully long
                entry_price = buy_price
                partial_exit_done = False

                trades.append({
                    "action": "BUY_OPEN",
                    "index": i,
                    "price": buy_price,
                    "fee": fee_buy,
                    "p_up": p_up
                })

            elif p_up < sell_threshold:
                # go SHORT
                sell_price = current_price * (1 - slippage_rate)
                fee_sell = sell_price * fee_rate

                position_size = -1  # -1 means fully short
                entry_price = sell_price
                partial_exit_done = False

                trades.append({
                    "action": "SELL_OPEN",
                    "index": i,
                    "price": sell_price,
                    "fee": fee_sell,
                    "p_up": p_up
                })

            # move to next bar
            i += 1
        else:
            # we have a position, check stop-loss, take-profit, partial exit, or time-based exit
            bar_held = 0
            # We'll iterate bars until we exit or we hit max_hold_bars
            exit_trade = False

            while i < len(X) and not exit_trade:
                row = X.iloc[i]
                row_df = row.to_frame().T
                current_price = row["feature_price"]

                # compute current gain/loss
                if position_size > 0:
                    # long
                    raw_return = (current_price - entry_price) / entry_price
                else:
                    # short
                    raw_return = (entry_price - current_price) / entry_price

                # Check partial exit logic: if we have > half the take_profit => close 50%
                # e.g. if we have +2.5% on a +5% target, do partial. (You can adapt logic)
                if not partial_exit_done and raw_return > 0 and raw_return >= (take_profit_pct / 2):
                    # close half
                    exit_price = current_price * (1 - slippage_rate) if position_size > 0 else current_price * (1 + slippage_rate)
                    fee_exit = exit_price * fee_rate * 0.5  # half position
                    realized_pnl = 0.5 * position_size * (exit_price - entry_price) if position_size > 0 \
                        else 0.5 * position_size * (entry_price - exit_price)
                    realized_pnl_percent = (realized_pnl / (entry_price * abs(0.5 * position_size))) * 100 if position_size != 0 else 0

                    trades.append({
                        "action": "PARTIAL_EXIT",
                        "index": i,
                        "price": exit_price,
                        "fee": fee_exit,
                        "pnl_percent": realized_pnl_percent
                    })
                    partial_exit_done = True
                    # we remain in half position => position_size stays same,
                    # but you could track partial size. For brevity, let's keep it at 1 or -1
                    # or you can do position_size = 0.5 => more advanced. We'll do simpler approach here.

                # check stop loss
                if raw_return <= -abs(stop_loss_pct):
                    # stop out
                    exit_price = current_price * (1 - slippage_rate) if position_size > 0 else current_price * (1 + slippage_rate)
                    fee_exit = exit_price * fee_rate
                    # realized
                    if position_size > 0:
                        realized_pnl = (exit_price - entry_price) - fee_exit
                        realized_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        realized_pnl = (entry_price - exit_price) - fee_exit
                        realized_pct = ((entry_price - exit_price) / entry_price) * 100

                    trades.append({
                        "action": "STOP_OUT",
                        "index": i,
                        "price": exit_price,
                        "fee": fee_exit,
                        "pnl_percent": realized_pct
                    })
                    position_size = 0
                    exit_trade = True
                    i += 1
                    break

                # check take profit
                if raw_return >= take_profit_pct:
                    # fully exit
                    exit_price = current_price * (1 - slippage_rate) if position_size > 0 else current_price * (1 + slippage_rate)
                    fee_exit = exit_price * fee_rate
                    if position_size > 0:
                        realized_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        realized_pct = ((entry_price - exit_price) / entry_price) * 100

                    trades.append({
                        "action": "TAKE_PROFIT",
                        "index": i,
                        "price": exit_price,
                        "fee": fee_exit,
                        "pnl_percent": realized_pct
                    })
                    position_size = 0
                    exit_trade = True
                    i += 1
                    break

                # check max_hold_bars
                if bar_held >= max_hold_bars:
                    # time-based exit
                    exit_price = current_price * (1 - slippage_rate) if position_size > 0 else current_price * (1 + slippage_rate)
                    fee_exit = exit_price * fee_rate
                    if position_size > 0:
                        realized_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        realized_pct = ((entry_price - exit_price) / entry_price) * 100

                    trades.append({
                        "action": "MAX_HOLD_EXIT",
                        "index": i,
                        "price": exit_price,
                        "fee": fee_exit,
                        "pnl_percent": realized_pct
                    })
                    position_size = 0
                    exit_trade = True
                    i += 1
                    break

                # Optionally: if we see a big reversal (p_up < sell_threshold for a long?), close or flip
                # We'll skip that for brevity.

                # else continue
                bar_held += 1
                i += 1

    # Summarize trades
    if not trades:
        logger.info("No trades triggered in advanced backtest.")
        df_trades = pd.DataFrame()
        stats = {}
        return df_trades, stats

    df_trades = pd.DataFrame(trades)

    # We compute cumulative PnL from each exit. Some rows might be partial, some might be open/close
    # We'll interpret any row with "pnl_percent" as a realized trade portion
    # accumulate them
    df_trades["cumulative_pnl"] = df_trades["pnl_percent"].fillna(0).cumsum()

    # Max drawdown
    peak = -999999
    max_drawdown = 0.0
    for val in df_trades["cumulative_pnl"]:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_drawdown:
            max_drawdown = dd

    # Sharpe ratio (approx) => we treat each trade as if it were "daily return"
    # Not strictly correct, but for demonstration:
    returns = df_trades["pnl_percent"].fillna(0) / 100.0
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = returns.mean() / returns.std()

    final_pnl = df_trades["pnl_percent"].fillna(0).sum()

    stats = {
        "total_pnl_%": final_pnl,
        "max_drawdown_%": max_drawdown,
        "sharpe": sharpe,
        "num_trades": len(df_trades)
    }

    logger.info(f"Advanced Backtest => total_pnl={final_pnl:.2f}%, max_dd={max_drawdown:.2f}%, sharpe={sharpe:.2f}, trades={len(df_trades)}")

    return df_trades, stats

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    """
    1) Load BTC data if desired for correlation.
    2) Load aggregator from cryptopanic daily if desired.
    3) For each pair in config.yaml, load 'price_history', build features, merges aggregator data.
    4) Combine everything into X_full, y_full, then do hyperparam search + final hold-out test.
    5) Run an advanced backtest on the combined dataset to see real "trading" results.
    6) Save the model + log final metric.
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

    # 4) Train the model with hyperparam search + final hold-out
    model, final_metric = train_model(X_full, y_full)
    if model is None:
        logger.error("Model training returned None. Exiting.")
        return

    # 5) Run advanced backtest
    logger.info("Running advanced backtest on the full dataset to see realistic 'right calls'.")
    df_trades, adv_stats = backtest_model(
        X_full, model,
        buy_threshold=0.6,
        sell_threshold=0.4,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        max_hold_bars=SHIFT_BARS,
        slippage_rate=0.0005,
        fee_rate=0.001
    )
    logger.info(f"Backtest stats: {adv_stats}")

    # 6) Save the model + log final metric
    log_accuracy_to_csv(final_metric)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
    logger.info(f"Appended f1_macro={final_metric:.4f} to {METRICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
