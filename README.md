Simple Trader (Deprecated Proof of Concept)
==========================================

Status: Deprecated proof-of-concept (POC) that explores the use of LLMs and real-time market data to drive programmatic trading decisions. This repository was created to learn about AI usage and limitations over programmatic API calls. It is not production-ready and must not be used for live trading without extensive review and hardening.


Overview
--------
Simple Trader is an async Python application that orchestrates:

- Postgres-backed data ingestion from Kraken WebSockets (ticker, order book, private events) and REST APIs
- A periodic Aggregator cycle that composes prompts and calls the OpenAI Chat Completions API to obtain trading recommendations
- A Risk Manager that validates recommendations and, when enabled, sends orders through Kraken’s private WebSocket v2
- A lightweight online-learning ML loop to incrementally update a simple trend model and feed its outputs into the prompt

The application’s orchestrator is `main.py`. It wires together configuration, database, REST/WS connectivity, the Aggregator (prompt + GPT loop), the Risk Manager, optional ML training, and long-running background tasks.


Orchestration: main.py
----------------------
`main.py` is the entry point. It performs:

1) Database and configuration bootstrap
   - Initialize `DBManager` and fix PK sequences where needed
   - Initialize `ConfigLoader` and install application logging via `setup_logging`

2) REST connectivity and data upsert
   - Start `KrakenRestManager` (with `HallMonitor` for REST throttling)
   - Run a startup “table upsert” sequence to populate reference and account tables:
     `kraken_assets`, `kraken_pairs`, `kraken_balances`, `kraken_trades`, `kraken_ledgers`, `kraken_orders`

3) Streaming ingestion and background tasks
   - Create a `price_queue` and start `price_consumer` to persist ticker tuples into `price_history`
   - Construct and initialize `RiskManager`
   - Build `AIStrategy` (delegates order logic to `RiskManager`)
   - Ensure an ML model exists (or train) via `ensure_model_trained`
   - Construct `Aggregator` with a reference to the predictor
   - Start `WSManager` to bring up public/private/order book WS feeds
   - Start long-running tasks: risk checks, scheduled incremental training, TradeSmith, CoinSmith (optional), etc.

4) Graceful shutdown
   - On `KeyboardInterrupt`, cancel tasks, stop WS, shut down REST and DB


The Aggregation Cycle (Prompt + GPT + Decision)
-----------------------------------------------
The Aggregator encapsulates the AI decision loop. Each cycle:

1) Refreshes account context
   - Calls `KrakenRestManager.fetch_and_store_ledger(...)`
   - Calls `KrakenRestManager.fetch_balance()`, normalizes balances (e.g., map base assets to `wsname`)

2) Prepares per-pair features
   - Loads minute-spaced prices from `price_history`
   - Computes technical indicators (SMA/EMA/RSI/MACD/Bollinger)
   - Optionally stores order book metrics from `OrderBookManager`
   - Optionally retrieves model probabilities via `PriceTrendPredictor`
   - Writes an `ai_snapshots` row to support ML and audit

3) Builds the prompt
   - Renders a Mustache template (`templates/*.mustache`) with balances, per‑pair items, and model hints

4) Calls OpenAI Chat Completions (via `GPTManager`)
   - If `is_function_calling_model` is enabled, uses a tool schema (`tools/aggregator_function.json`)
   - Otherwise, falls back to JSON parsing from the assistant’s content

5) Delegates decisions to `AIStrategy` and `RiskManager`
   - `AIStrategy` applies minimal checks, stores an `ai_decisions` audit row, and calls `RiskManager.handle_gpt_decision`
   - `RiskManager` applies risk controls, size clamps, min‑cost checks, and optionally submits orders to Kraken Private WS v2

OpenAI Integration (GPTManager)
-------------------------------
`GPTManager` sends the Aggregator’s prompt to the OpenAI Chat Completions API:

- Supports optional function calling (`aggregator_decision` tool) if enabled by config
- Otherwise parses structured JSON from the assistant’s content block
- Returns a canonical structure `{ decisions: [...], rationale, confidence }`
- Optionally logs prompts/responses under `./logs/<timestamp>/`

Important: This repo is a POC; the LLM outputs are unverified and must be treated as untrusted suggestions.


WebSockets and REST
-------------------
- `WSManager` orchestrates:
  - Public v2 ticker feed -> push into `price_queue` -> persisted by `price_consumer`
  - Private v2 (executions, balances) -> updates `pending_trades`, `trades`, `kraken_balance_history`; invokes RiskManager callbacks
  - v2 Order Book feed -> `OrderBookManager` computes metrics -> stored in `order_book_metrics`
  - Reactive runtime config toggles (start/stop feeds without restart)

- `KrakenRestManager` provides startup upsert and on-demand calls:
  - Public: `/0/public/Assets`, `/0/public/AssetPairs`
  - Private: balances, trade balance, ledger, open/closed orders, trades history, fees
  - Integrates `HallMonitor` for REST throttling


Risk and Trade Management
-------------------------
- `RiskManager`
  - Validates GPT decisions against min cost (`ordermin * latest price`) and available balances
  - Applies soft size adjustments and clamps per pair decimals
  - Records `pending_trades` and `holding_lots` and, when enabled, sends orders via Private WS v2 (advanced params for SL/TP)
  - Performs background stop‑loss / take‑profit checks if advanced OCO isn’t already managing exits

- `TradeSmith` (advanced automation)
  - Optional higher-level engine for entry/exit orchestration, dynamic OCO flips, and subscription optimization


Model Training (Optional)
-------------------------
- `model_trainer.py` implements a simple online-learning `PriceTrendPredictor` (River) over `ai_snapshots` (+ optional order book metrics) with horizons [1,2,3,5,10]
- `scheduled_training_task` periodically runs incremental training and reloads the predictor into the Aggregator


Data Model (Selected Tables)
----------------------------
- `price_history`: last/bid/ask/volume/vwap per pair (from public WS)
- `ai_snapshots`: Aggregator snapshot per pair (features, indicators, context)
- `order_book_metrics`: spread/imbalance/mid/vwap_mid (throttled writes)
- `pending_trades`, `trades`, `holding_lots`, `stop_loss_events`: order lifecycle and position tracking
- Kraken mirrors: `kraken_assets`, `kraken_pairs`, `kraken_balance_history`, `kraken_orders`, `kraken_ledgers`, `kraken_trades`
- `model_predictions`: logged probabilities for analysis


Configuration
-------------
Primary file: `config.yaml` (hot-reloaded by `ConfigLoader`). Notable keys:

- Feeds and runtime:
  - `enable_public_feed`, `enable_private_feed` (implicit via `place_live_orders`), `enable_order_book_feed`
  - `enable_minimal_order_book_feed`, `minimal_order_book_depth`, `order_book_depth`
- Model and GPT:
  - `openai_model`, `is_reasoning_model`, `reasoning_effort`, `is_function_calling_model`, `model_max_tokens`
- Aggregator and risk:
  - `quantity_of_price_history`, `stop_loss_percent`, `take_profit_percent`, `risk_manager_interval_seconds`
- TradeSmith:
  - `trade_smith_interval_seconds`, `trade_smith_cooldown_seconds`, volatility thresholds, etc.
- CoinSmith:
  - `coin_smith.*` (weights and windows)
- `traded_pairs`: list of pairs `e.g., ["ETH/USD", "SOL/USD", ...]`


Environment Variables (.env)
----------------------------
The following environment variables are typically required:

```
POSTGRES_DSN=postgresql+asyncpg://user:pass@host:5432/dbname
OPENAI_API_KEY=sk-...
KRAKEN_API_KEY=...
KRAKEN_SECRET_API_KEY=...
LUNARCRUSH_API_KEY=...
```


Running Locally (POC)
---------------------
1) Install dependencies (Python 3.10+ recommended). This project uses:
   - `asyncio`, `sqlalchemy[asyncio]`, `asyncpg`, `aiohttp`, `websockets`, `python-dotenv`, `httpx`, `openai`, `pandas`, `numpy`, `river`, `colorama`, `pyyaml` (and TA‑Lib for `coin_smith.py`, if used)

2) Provision Postgres and apply any migrations/schema you maintain (see `models.py`). Ensure `POSTGRES_DSN` is valid.

3) Provide `.env` with the variables above.

4) Review `config.yaml` to set pairs, feeds, and model options. For safe dry runs, disable `place_live_orders`.

5) Start the app:

```
python -m asyncio run main.py
```

Notes:
- This POC is not fully hardened. If you plan to actually place orders, test thoroughly on a paper/sandbox environment and review all risk controls.


Known Caveats (Refactor Deltas)
-------------------------------
This codebase contains areas that reflect in-progress refactors typical of a POC:

- `CoinSmith` constructor currently expects only `db_manager`, but some call sites pass additional args. Align the call signature before running.
- `TradeSmith` signature includes `ws_manager`, but certain initializations may omit or reorder parameters. Verify argument order/types.
- `aggregator_task` creation is present but commented out in `main.py`; enable if you want the loop to run on a schedule.

Treat these as guidance for local fixes; the architecture described above remains accurate.


Security, Compliance, and Risk
------------------------------
- This repository is a learning POC. It is not audited and is not intended for production trading.
- LLM outputs are unverified and may be incorrect or unsafe. Never act on LLM advice without human review and robust guardrails.
- Be mindful of API keys and secrets. Use least-privilege keys and never commit secrets to source control.
- Respect exchange rate limits and terms of service; `HallMonitor` provides basic throttling for REST calls.


Troubleshooting
---------------
- No data in `price_history`: ensure public WS feed is enabled and `price_consumer` is running.
- Orders not appearing: verify `place_live_orders` is true, Private WS is connected, and Open/Closed Orders endpoints are reachable.
- Model not updating: check `scheduled_training_task` frequency, `ai_snapshots` content, and predictor reload logs.
- Prompt empty or GPT errors: ensure `OPENAI_API_KEY` is set and the `openai_model` is valid for your account.


License and Attribution
-----------------------
Copyright © Contributors.
This project is provided “as is,” without warranty of any kind, for educational purposes only.


