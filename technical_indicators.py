# In technical_indicators.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_technical_indicators(price_data: list) -> dict:
    """
    Computes various technical indicators from historical price data.

    Args:
        price_data (list): List of [timestamp, price] pairs.

    Returns:
        dict: Dictionary of indicator values. Returns None for indicators
              that cannot be computed due to insufficient data.
    """
    if not price_data or len(price_data) < 26:  # Need at least 26 points for MACD
        logger.warning("[Indicators] Insufficient price data for technical indicators.")
        return {
            "sma_10": None,
            "ema_10": None,
            "rsi_14": None,
            "macd": None,
            "macd_signal": None,
            "bollinger_upper": None,
            "bollinger_lower": None
        }

    try:
        df = pd.DataFrame(price_data, columns=["timestamp", "price"])
        df["price"] = df["price"].astype(float)

        # SMA and EMA
        sma_10 = df["price"].rolling(window=10).mean().iloc[-1]
        ema_10 = df["price"].ewm(span=10, adjust=False).mean().iloc[-1]

        # RSI
        delta = df["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else None

        # MACD
        ema_12 = df["price"].ewm(span=12, adjust=False).mean()
        ema_26 = df["price"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = macd.iloc[-1]
        macd_signal_value = macd_signal.iloc[-1]

        # Bollinger Bands
        sma_20 = df["price"].rolling(window=20).mean()
        std_20 = df["price"].rolling(window=20).std()
        bollinger_upper = sma_20 + (std_20 * 2)
        bollinger_lower = sma_20 - (std_20 * 2)
        bollinger_upper_value = bollinger_upper.iloc[-1]
        bollinger_lower_value = bollinger_lower.iloc[-1]

        return {
            "sma_10": sma_10,
            "ema_10": ema_10,
            "rsi_14": rsi_14,
            "macd": macd_value,
            "macd_signal": macd_signal_value,
            "bollinger_upper": bollinger_upper_value,
            "bollinger_lower": bollinger_lower_value
        }
    except Exception as e:
        logger.exception(f"[Indicators] Error computing technical indicators: {e}")
        return {
            "sma_10": None,
            "ema_10": None,
            "rsi_14": None,
            "macd": None,
            "macd_signal": None,
            "bollinger_upper": None,
            "bollinger_lower": None
        }