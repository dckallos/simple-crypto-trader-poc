import os
import pytest
from ai_strategy import AIStrategy

@pytest.mark.integration
def test_aistrategy_real_gpt():
    """
    Creates a real AIStrategy with use_openai=True and tries a small prompt,
    verifying GPT tools or text fallback. This can cost tokens.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No real OPENAI_API_KEY => skipping integration test.")

    # Create strategy instance using real GPT
    strat = AIStrategy(
        pairs=["ETH/USD"],
        use_openai=True,
        max_position_size=0.0001,  # keep trades small
    )

    # Provide simple aggregator data
    market_data = {
        "pair": "ETH/USD",
        "price": 1500.0,
        "cryptopanic_sentiment": 0.3,
        "galaxy_score": 40.0,
        "alt_rank": 120
    }

    action, size = strat.predict(market_data)
    # We can't strictly guarantee "BUY" or "SELL" from GPT,
    # but we can check that we get a valid string in ("BUY","SELL","HOLD") and 0 <= size <= 0.0001
    assert action in ("BUY", "SELL", "HOLD"), f"Unexpected GPT action: {action}"
    assert 0.0 <= size <= 0.0001, f"Size out of range: {size}"

    print(f"AIStrategy with real GPT => action={action}, size={size}")
