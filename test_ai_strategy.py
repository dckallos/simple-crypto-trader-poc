import os
import sqlite3
import pytest
import time
from unittest.mock import MagicMock, patch
from pathlib import Path

from ai_strategy import AIStrategy
from db import init_db, load_gpt_context_from_db, save_gpt_context_to_db, DB_FILE
from risk_manager import RiskManagerDB

@pytest.fixture
def temp_db(tmp_path):
    """
    A pytest fixture to run tests against a temporary SQLite database file,
    leaving the main DB_FILE intact.
    """
    test_db = tmp_path / "test_trades.db"
    # Update the global DB_FILE reference so that ai_strategy + db + risk_manager
    # write to this test DB. In a real project, you may also want to patch DB_FILE
    # in each module that references it, to ensure it points to test_db.
    original_db_file = DB_FILE
    try:
        # monkeypatch the global
        import db
        db.DB_FILE = str(test_db)

        import risk_manager
        risk_manager.DB_FILE = str(test_db)

        yield str(test_db)
    finally:
        # revert
        db.DB_FILE = original_db_file
        risk_manager.DB_FILE = original_db_file


@pytest.fixture
def init_test_db(temp_db):
    """
    Initializes the schema in the test DB and returns the path.
    """
    init_db()  # This should create tables in temp_db
    return temp_db


@pytest.fixture
def ai_strategy_no_openai(init_test_db):
    """
    Returns an AIStrategy instance that does NOT use GPT, only fallback or scikit.
    """
    strat = AIStrategy(
        pairs=["ETH/USD"],
        use_openai=False,
        max_position_size=0.001,  # clamp
    )
    return strat


@pytest.fixture
def ai_strategy_openai(init_test_db):
    """
    Returns an AIStrategy instance that DOES use GPT logic. We'll mock the openai calls.
    """
    strat = AIStrategy(
        pairs=["ETH/USD"],
        use_openai=True,
        max_position_size=0.001,
    )
    return strat


class TestAIStrategy:
    def test_fallback_logic_no_price(self, ai_strategy_no_openai):
        """
        If price <= 0, strategy should produce ("HOLD", 0.0).
        """
        market_data = {
            "pair": "ETH/USD",
            "price": 0.0,
        }
        action, size = ai_strategy_no_openai.predict(market_data)
        assert action == "HOLD"
        assert size == 0.0

    def test_fallback_dummy_logic(self, ai_strategy_no_openai):
        """
        If price < 20000 => BUY 0.0005, else HOLD. This tests the dummy path
        because scikit is not loaded and use_openai=False.
        """
        market_data = {
            "pair": "ETH/USD",
            "price": 15000.0,  # less than 20000
        }
        action, size = ai_strategy_no_openai.predict(market_data)
        assert action == "BUY"
        assert size == 0.0005

        market_data2 = {
            "pair": "ETH/USD",
            "price": 25000.0,
        }
        action2, size2 = ai_strategy_no_openai.predict(market_data2)
        assert action2 == "HOLD"
        assert size2 == 0.0

    @patch("ai_strategy.joblib.load")
    def test_scikit_model_logic(self, mock_model_load, ai_strategy_no_openai):
        """
        Suppose we do have a scikit model. We can patch joblib.load to return
        a mock classifier that returns a certain probability.
        """
        # We'll pretend we created ai_strategy with a model_path => let's forcibly load it:
        mock_model = MagicMock()
        # mock predict_proba => returns [[0.3, 0.7]] => prob_up=0.7 => => BUY
        mock_model.predict_proba.return_value = [[0.3, 0.7]]

        mock_model_load.return_value = mock_model

        # We'll re-instantiate AIStrategy with a model path (for demonstration).
        # In a real scenario, you'd do that in a separate fixture if needed.
        strat = AIStrategy(
            pairs=["ETH/USD"],
            model_path="fake_model.pkl",
            use_openai=False,
            max_position_size=0.001,
        )

        # Because the DB is ephemeral, we need some minimal price_history data:
        _insert_dummy_price_history("ETH/USD", 21000.0)

        market_data = {
            "pair": "ETH/USD",
            "price": 21000.0,
            "cryptopanic_sentiment": 0.1
        }
        action, size = strat.predict(market_data)
        assert action == "BUY"
        # from scikit => "size_suggested=0.0005" if prob_up > 0.6
        # but also check we clamp at 0.001 if needed
        assert size == 0.0005

    @patch("ai_strategy.OpenAI")
    def test_gpt_fallback_if_no_choices(self, mock_openai_constructor, ai_strategy_openai):
        """
        If GPT returns no choices => fallback => dummy logic (since no scikit).
        """
        # mock the client
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[])
        mock_openai_constructor.return_value = mock_openai

        # We also need some minimal price_history or the fallback is "dummy"
        market_data = {"pair": "ETH/USD", "price": 14000.0}
        action, size = ai_strategy_openai.predict(market_data)
        assert action == "BUY"  # from dummy fallback
        assert size == 0.0005

    @patch("ai_strategy.OpenAI")
    def test_gpt_tool_call(self, mock_openai_constructor, ai_strategy_openai):
        """
        If GPT returns finish_reason='tool_calls' with a function => parse that function call.
        """
        mock_openai = MagicMock()

        # We'll simulate the model returning a 'tool_calls' finish_reason
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"

        # The .message has .tool_calls => one function call => "action":"BUY","size":0.0008
        mock_message = MagicMock()
        # We simulate the typed approach: message.tool_calls => [some tool call object]
        # We'll just mock out: tool_calls => an array of objects with .function
        # For simplicity, let's define them as we expect them:
        fn_obj = MagicMock()
        fn_obj.arguments = '{"action":"BUY","size":0.0008}'
        first_tool_call = MagicMock()
        first_tool_call.function = fn_obj
        mock_message.tool_calls = [first_tool_call]
        mock_choice.message = mock_message

        mock_completion.choices = [mock_choice]
        mock_openai.chat.completions.create.return_value = mock_completion

        mock_openai_constructor.return_value = mock_openai

        # Insert minimal price_history so we skip scikit
        _insert_dummy_price_history("ETH/USD", 10000.0)

        market_data = {"pair": "ETH/USD", "price": 10000.0}
        action, size = ai_strategy_openai.predict(market_data)
        # This should parse the function args => BUY, size=0.0008
        assert action == "BUY"
        assert abs(size - 0.0008) < 1e-9, f"Expected 0.0008, got {size}"

    def test_risk_controls_min_buy_amount(self, ai_strategy_no_openai):
        """
        If cost < min_buy => force hold
        """
        # We'll set risk controls
        rc = {
            "initial_spending_account": 10000.0,
            "purchase_upper_limit_percent": 0.01,
            "minimum_buy_amount": 50.0,
            "max_position_value": 5000.0
        }

        ai_strategy_no_openai.risk_controls = rc

        # fallback => if price < 20000 => we do BUY 0.0005 => cost= price*0.0005
        # If price=5000 => cost=5000*0.0005=2.5 < min_buy=50 => => HOLD
        market_data = {"pair": "ETH/USD", "price": 5000.0}
        action, size = ai_strategy_no_openai.predict(market_data)
        assert action == "HOLD"
        assert size == 0.0

    def test_risk_controls_purchase_upper_limit(self, ai_strategy_no_openai):
        """
        If cost > purchase_upper => clamp the size
        purchase_upper = 0.01 * 10000 => 100. So if cost=200 => clamp to 100
        """
        rc = {
            "initial_spending_account": 10000.0,
            "purchase_upper_limit_percent": 0.01,  # => 100
            "minimum_buy_amount": 20.0,
            "max_position_value": 5000.0
        }
        ai_strategy_no_openai.risk_controls = rc

        # fallback => price=30000 => we do BUY 0.0005 => cost=15 => which is > min(20)? It's actually < 20? Wait, 15 is < 20 => means hold
        # let's pick a bigger size scenario. We'll have to override the dummy logic? Actually let's do the same approach.
        # We'll do a forced scenario. We'll just pretend we do scikit => that sets size=0.01 => cost= price*0.01
        # We'll hack a function call:
        final_signal, final_size = ai_strategy_no_openai._post_validate_and_adjust(
            "BUY",
            size_suggested=0.01,  # attempt to buy 0.01
            current_price=30000.0
        )
        # cost=0.01*30000=300 => purchase_upper=100 => clamp => 100/30000 => ~0.0033
        assert final_signal == "BUY"
        assert abs(final_size - 0.0033) < 1e-4

    def test_gpt_context_persistence(self, init_test_db):
        """
        Tests that GPT context is stored/loaded in the 'ai_context' table.
        """
        # write
        save_gpt_context_to_db("Hello, this is my test context.")
        # read
        ctx = load_gpt_context_from_db()
        assert ctx == "Hello, this is my test context."

    def test_risk_manager_stop_loss(self, ai_strategy_no_openai):
        """
        If the sub-position is open and we have stop_loss_pct => it closes the position if price falls.
        """
        # We'll forcibly open a sub-position with side=long, entry=2000 => if price=1900 => -5% => triggers stop
        rm = ai_strategy_no_openai.risk_manager_db
        rm.stop_loss_pct = 0.05
        # open
        rm._insert_sub_position("ETH/USD", "long", 2000.0, 0.001)
        # now check with new price=1800 => that's a 10% drop => triggers stop
        rm.check_stop_loss_take_profit("ETH/USD", 1800.0)

        # verify closed
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT closed_at, realized_pnl FROM sub_positions WHERE pair=?",( "ETH/USD", ))
        row = c.fetchone()
        conn.close()
        assert row is not None, "No sub-position found"
        closed_at, realized_pnl = row
        assert closed_at is not None, "Should be closed"
        # realized => (exit_price - entry_price)*size => (1800-2000)*0.001=-0.2 => -$0.2
        assert abs(realized_pnl + 0.2) < 1e-6

# ------------------------------------------------------------------
# Helper functions for test data
# ------------------------------------------------------------------
def _insert_dummy_price_history(pair: str, price: float):
    """
    Insert a single row of 'price_history' with the given pair + last_price.
    Enough so that AIStrategy's scikit/dummy logic won't fail on empty DF.
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO price_history (timestamp, pair, bid_price, ask_price, last_price, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (int(time.time()), pair, price-5, price+5, price, 1000.0))
        conn.commit()
    finally:
        conn.close()
