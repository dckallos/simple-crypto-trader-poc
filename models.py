#!/usr/bin/env python3
# =============================================================================
# FILE: models.py
# =============================================================================
"""
models.py

Defines all database tables for the cryptocurrency trading application using SQLAlchemy
with async support for PostgreSQL. This module replaces the SQLite table creation logic
previously scattered across db.py, ai_strategy.py, risk_manager.py, train_model.py,
and fetch_lunarcrush.py. Each model is production-ready, well-documented, and includes
appropriate constraints and indexes for high-performance trading operations.

Key Features:
- Async support via SQLAlchemy's AsyncAttrs for non-blocking database operations.
- Centralized schema definition for maintainability and future migrations with Alembic.
- Indexes on frequently queried columns (e.g., pair, timestamp) for sub-second responsiveness.
- Foreign key relationships (e.g., trades.lot_id -> holding_lots.id) for data integrity.
- JSONB columns for flexible storage of structured data (e.g., price changes, risk params).

Tables Defined:
1. Trades: Finalized trade executions with analytics.
2. PendingTrades: Ephemeral trade orders before execution.
3. PriceHistory: Real-time market price data.
4. LedgerEntries: Kraken ledger events (deposits, withdrawals, trades).
5. AISnapshots: AI inference context for machine learning.
6. HoldingLots: Open positions with cost basis and risk metrics.
7. StopLossEvents: Stop-loss trigger logs.
8. LunarCrushData: Snapshot data from LunarCrush for sentiment and volatility.
9. LunarCrushTimeseries: Historical time-series data from LunarCrush.
10. KrakenAssetPairs: Kraken pair metadata (e.g., min order size).
11. KrakenBalanceHistory: Periodic balance snapshots.
12. AIContext: Legacy GPT context (single row, optional).
13. AIDecisions: AI-driven trade decisions for logging.
14. AggregatorSummaries: Aggregated metrics for analysis.
15. AggregatorClassifierProbs: Local classifier probabilities.
16. AggregatorEmbeddings: PCA-based embeddings for ML.

Usage:
- Import into db_manager.py to initialize the database schema.
- Use with AsyncSession for async database operations in the application.

Dependencies:
- SQLAlchemy with asyncio support (install: `pip install sqlalchemy[asyncio] asyncpg`)
"""

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models with async support."""
    pass


# 1. Trades
class Trade(Base):
    """Stores finalized trade executions with analytics data."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    kraken_trade_id = Column(String)
    pair = Column(String, nullable=False)  # e.g., "ETH/USD"
    side = Column(String, nullable=False)  # "BUY" or "SELL"
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    order_id = Column(String)
    fee = Column(Float, default=0.0)
    realized_pnl = Column(Float)
    source = Column(String)  # e.g., "ai_strategy"
    rationale = Column(Text)
    lot_id = Column(Integer, ForeignKey("holding_lots.id"))
    ai_model = Column(String)  # e.g., "o1-mini"
    source_config = Column(JSON)  # JSON risk config
    ai_rationale = Column(Text)  # Full GPT rationale
    exchange_fill_time = Column(BigInteger)  # Actual exchange fill time
    trade_metrics_json = Column(JSON)  # Peak excursions, etc.

    __table_args__ = (
        Index("ix_trades_pair_timestamp", "pair", "timestamp"),
        Index("ix_trades_order_id", "order_id"),
    )


# 2. PendingTrades
class PendingTrade(Base):
    """Tracks ephemeral trade orders before execution or rejection."""
    __tablename__ = "pending_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(BigInteger, nullable=False)  # Epoch time
    pair = Column(String, nullable=False)
    side = Column(String, nullable=False)  # "BUY" or "SELL"
    requested_qty = Column(Float, nullable=False)
    status = Column(String, nullable=False)  # "pending", "open", "closed", "rejected"
    kraken_order_id = Column(String)
    reason = Column(Text)
    lot_id = Column(Integer, ForeignKey("holding_lots.id"))
    source = Column(String)  # e.g., "ai_strategy"
    rationale = Column(Text)

    __table_args__ = (
        Index("ix_pending_trades_pair_created_at", "pair", "created_at"),
        Index("ix_pending_trades_kraken_order_id", "kraken_order_id"),
    )


# 3. PriceHistory
class PriceHistory(Base):
    """Logs real-time market price data from Kraken’s public WebSocket."""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    pair = Column(String, nullable=False)
    bid_price = Column(Float)
    ask_price = Column(Float)
    last_price = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        Index("ix_price_history_pair_timestamp", "pair", "timestamp"),
    )


# 4. LedgerEntries
class LedgerEntry(Base):
    """Records all Kraken ledger events (deposits, withdrawals, trades)."""
    __tablename__ = "ledger_entries"

    ledger_id = Column(String, primary_key=True)  # e.g., "L4UESK-KG3EQ-UFO4T5"
    refid = Column(String)
    time = Column(Float, nullable=False)  # Epoch time with microseconds
    type = Column(String, nullable=False)
    subtype = Column(String)
    asset = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    balance = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_ledger_entries_asset_time", "asset", "time"),
    )


# 5. AISnapshots
class AISnapshot(Base):
    """Captures AI inference context for machine learning and analytics."""
    __tablename__ = "ai_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(BigInteger, nullable=False)  # Epoch time
    pair = Column(String, nullable=False)  # e.g., "ETH/USD" or "ALL"
    aggregator_interval_secs = Column(Integer, nullable=False)
    stop_loss_pct = Column(Float, nullable=False)
    take_profit_pct = Column(Float, nullable=False)
    daily_drawdown_limit = Column(Float, nullable=False)
    price_changes_json = Column(JSON, nullable=False)  # Vector of % changes
    last_price = Column(Float, nullable=False)
    lunarcrush_data_json = Column(JSON)  # Snapshot of LunarCrush metrics
    coin_volatility = Column(Float, nullable=False)
    is_market_bullish = Column(String, nullable=False)  # "YES", "NO", "MIXED"
    risk_estimate = Column(Float, nullable=False)
    notes = Column(Text)

    __table_args__ = (
        Index("ix_ai_snapshots_pair_created_at", "pair", "created_at"),
    )


# 6. HoldingLots
class HoldingLot(Base):
    """Manages open positions with cost basis and risk metrics."""
    __tablename__ = "holding_lots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String, nullable=False)
    purchase_price = Column(Float, nullable=False)
    initial_quantity = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    date_purchased = Column(BigInteger, nullable=False)  # Epoch time
    lot_status = Column(String, default="ACTIVE")  # "ACTIVE", "PENDING_SELL", "PARTIAL_SOLD", "CLOSED"
    origin_source = Column(String)  # e.g., "gpt", "manual"
    strategy_version = Column(String)  # e.g., "v1.0"
    risk_params_json = Column(JSON)  # JSON snapshot of risk configs
    time_closed = Column(BigInteger)  # Epoch time when closed
    peak_favorable_price = Column(Float)
    peak_adverse_price = Column(Float)

    __table_args__ = (
        Index("ix_holding_lots_pair_date_purchased", "pair", "date_purchased"),
    )


# 7. StopLossEvents
class StopLossEvent(Base):
    """Logs stop-loss triggers for auditing and analysis."""
    __tablename__ = "stop_loss_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lot_id = Column(Integer, ForeignKey("holding_lots.id"), nullable=False)
    triggered_at = Column(BigInteger, nullable=False)  # Epoch time
    current_price = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)
    risk_params_json = Column(JSON, nullable=False)  # Snapshot of risk params

    __table_args__ = (
        Index("ix_stop_loss_events_lot_id_triggered_at", "lot_id", "triggered_at"),
    )


# 8. LunarCrushData
class LunarCrushData(Base):
    """Stores LunarCrush snapshot data for market sentiment and volatility."""
    __tablename__ = "lunarcrush_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    lunarcrush_id = Column(Integer)
    symbol = Column(String, nullable=False)
    name = Column(String)
    price = Column(Float)
    price_btc = Column(Float)
    volume_24h = Column(Float)
    volatility = Column(Float)
    circulating_supply = Column(Float)
    max_supply = Column(Float)
    percent_change_1h = Column(Float)
    percent_change_24h = Column(Float)
    percent_change_7d = Column(Float)
    percent_change_30d = Column(Float)
    market_cap = Column(Float)
    market_cap_rank = Column(Integer)
    interactions_24h = Column(Float)
    social_volume_24h = Column(Float)
    social_dominance = Column(Float)
    market_dominance = Column(Float)
    market_dominance_prev = Column(Float)
    galaxy_score = Column(Float)
    galaxy_score_previous = Column(Float)
    alt_rank = Column(Integer)
    alt_rank_previous = Column(Integer)
    sentiment = Column(Float)
    categories = Column(Text)  # Comma-delimited
    topic = Column(Text)
    logo = Column(Text)
    blockchains = Column(Text)  # JSON string

    __table_args__ = (
        Index("ix_lunarcrush_data_symbol_timestamp", "symbol", "timestamp"),
    )


# 9. LunarCrushTimeseries
class LunarCrushTimeseries(Base):
    """Holds historical LunarCrush time-series data."""
    __tablename__ = "lunarcrush_timeseries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    open_price = Column(Float)
    close_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    market_dominance = Column(Float)
    circulating_supply = Column(Float)
    sentiment = Column(Float)
    spam = Column(Float)
    galaxy_score = Column(Float)
    volatility = Column(Float)
    alt_rank = Column(Integer)
    contributors_active = Column(Float)
    contributors_created = Column(Float)
    posts_active = Column(Float)
    posts_created = Column(Float)
    interactions = Column(Float)
    social_dominance = Column(Float)

    __table_args__ = (
        UniqueConstraint("coin_id", "timestamp", name="uq_lunarcrush_timeseries_coin_id_timestamp"),
        Index("ix_lunarcrush_timeseries_coin_id_timestamp", "coin_id", "timestamp"),
    )


# 10. KrakenAssetPairs
class KrakenAssetPair(Base):
    """Stores Kraken pair metadata (e.g., minimum order size)."""
    __tablename__ = "kraken_asset_pairs"

    pair_name = Column(String, primary_key=True)
    altname = Column(String)
    wsname = Column(String)
    aclass_base = Column(String)
    base = Column(String)
    aclass_quote = Column(String)
    quote = Column(String)
    lot = Column(String)
    cost_decimals = Column(Integer)
    pair_decimals = Column(Integer)
    lot_decimals = Column(Integer)
    lot_multiplier = Column(Integer)
    leverage_buy = Column(Text)  # JSON string
    leverage_sell = Column(Text)  # JSON string
    fees = Column(Text)  # JSON string
    fees_maker = Column(Text)  # JSON string
    fee_volume_currency = Column(String)
    margin_call = Column(Integer)
    margin_stop = Column(Integer)
    ordermin = Column(String)
    costmin = Column(String)
    tick_size = Column(String)
    status = Column(String)
    long_position_limit = Column(Integer)
    short_position_limit = Column(Integer)
    last_updated = Column(BigInteger)  # Epoch time

    __table_args__ = (
        Index("ix_kraken_asset_pairs_wsname", "wsname"),
    )


# 11. KrakenBalanceHistory
class KrakenBalanceHistory(Base):
    """Tracks periodic snapshots of Kraken balances."""
    __tablename__ = "kraken_balance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    asset = Column(String, nullable=False)
    balance = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_kraken_balance_history_asset_timestamp", "asset", "timestamp"),
    )


# 12. AIContext
class AIContext(Base):
    """Legacy table for storing GPT context (single row)."""
    __tablename__ = "ai_context"

    id = Column(Integer, primary_key=True)  # Fixed at 1
    context = Column(Text)

    __table_args__ = (
        UniqueConstraint("id", name="uq_ai_context_id"),
    )


# 13. AIDecisions
class AIDecision(Base):
    """Logs AI-driven trade decisions for analysis."""
    __tablename__ = "ai_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    pair = Column(String, nullable=False)
    action = Column(String, nullable=False)  # "BUY", "SELL", "HOLD"
    size = Column(Float, nullable=False)
    rationale = Column(Text)
    group_rationale = Column(Text)

    __table_args__ = (
        Index("ix_ai_decisions_pair_timestamp", "pair", "timestamp"),
    )


# 14. AggregatorSummaries
class AggregatorSummary(Base):
    """Aggregated metrics for analysis from lunarcrush_data."""
    __tablename__ = "aggregator_summaries"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    price_bucket = Column(String, nullable=False)  # "low", "medium", "high"
    galaxy_score = Column(Float, nullable=False)
    galaxy_score_previous = Column(Float, nullable=False)
    alt_rank = Column(Float, nullable=False)
    alt_rank_previous = Column(Float, nullable=False)
    market_dominance = Column(Float, nullable=False)
    dominance_bucket = Column(String, nullable=False)  # "dominant", "moderate"
    sentiment_label = Column(String, nullable=False)  # e.g., "strong_pos"
    tick_size = Column(Float)

    __table_args__ = (
        Index("ix_aggregator_summaries_symbol_timestamp", "symbol", "timestamp"),
    )


# 15. AggregatorClassifierProbs
class AggregatorClassifierProb(Base):
    """Stores probabilities from a local classifier."""
    __tablename__ = "aggregator_classifier_probs"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    prob_up = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_aggregator_classifier_probs_symbol_timestamp", "symbol", "timestamp"),
    )


# 16. AggregatorEmbeddings
class AggregatorEmbedding(Base):
    """Stores PCA-based embeddings for machine learning."""
    __tablename__ = "aggregator_embeddings"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # Epoch time
    comp1 = Column(Float, nullable=False)
    comp2 = Column(Float, nullable=False)
    comp3 = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_aggregator_embeddings_symbol_timestamp", "symbol", "timestamp"),
    )

class KrakenAssetNameLookup(Base):
    """
    Stores a lookup table mapping Kraken asset pairs to detailed asset information.
    Populated by build_coin_name_lookup_from_db in kraken_rest_manager.py, this table
    aggregates data from kraken_asset_pairs (wsname, base, pair_name) and Kraken's
    /AssetInfo endpoint (altname, decimals, etc.) for assets with quote='ZUSD'.

    Purpose:
    - Provides a consolidated view of asset metadata for efficient lookups.
    - Supports trading operations by storing formatting and precision details.

    Columns:
    - wsname: WebSocket name (e.g., "ETH/USD"), primary key.
    - base_asset: Base asset code (e.g., "XETH").
    - pair_name: Kraken pair name (e.g., "XETHZUSD").
    - alternative_name: Alternative name from kraken_asset_pairs (e.g., "ETH").
    - formatted_base_asset_name: Formatted name from /AssetInfo (e.g., "ETH").
    - aclass: Asset class (e.g., "currency").
    - decimals: Number of decimal places for the asset.
    - display_decimals: Number of decimals for display purposes.
    - collateral_value: Collateral value (nullable float).
    - status: Asset status (e.g., "enabled").

    Indexes:
    - ix_kraken_asset_name_lookup_base_asset: Optimizes lookups by base_asset.
    """
    __tablename__ = "kraken_asset_name_lookup"

    wsname = Column(String, primary_key=True)  # e.g., "ETH/USD"
    base_asset = Column(String, nullable=False)  # e.g., "XETH"
    pair_name = Column(String, nullable=False)  # e.g., "XETHZUSD"
    alternative_name = Column(String)  # e.g., "ETH" from kraken_asset_pairs
    formatted_base_asset_name = Column(String)  # e.g., "ETH" from /AssetInfo
    aclass = Column(String)  # e.g., "currency"
    decimals = Column(Integer)  # Number of decimal places
    display_decimals = Column(Integer)  # Display precision
    collateral_value = Column(Float, nullable=True)  # Nullable float
    status = Column(String)  # e.g., "enabled"

    __table_args__ = (
        Index("ix_kraken_asset_name_lookup_base_asset", "base_asset"),
    )


if __name__ == "__main__":
    # For testing schema creation
    from sqlalchemy.ext.asyncio import create_async_engine
    import asyncio


    async def init_db():
        from dotenv import load_dotenv
        import os
        load_dotenv()
        engine = create_async_engine(os.getenv("POSTGRES_DSN"))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()


    asyncio.run(init_db())