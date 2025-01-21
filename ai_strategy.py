"""
ai_strategy.py

Contains the AIStrategy class, which encapsulates our ML/AI logic. In a
real application, you'd load a trained model and use it here to generate signals.
"""

class AIStrategy:
    """
    AIStrategy class stub. Replace the dummy predict() method with your
    real model's inference logic.
    """

    def __init__(self):
        """
        Initialize or load your model here. For example:
        self.model = load_my_model("model_checkpoint.pt")

        In this stub, we do nothing.
        """
        pass

    def predict(self, market_data: dict):
        """
        Uses AI logic to decide whether to BUY, SELL, or HOLD.

        :param market_data: A dict containing at least "price" and "timestamp".
        :return: (signal, size)
                 signal => one of ("BUY", "SELL", "HOLD")
                 size => float indicating how many units to trade
        """
        # DUMMY LOGIC: If price is below some threshold, BUY, otherwise HOLD.
        #  - In reality, you'd run the data through a neural net or other model.

        current_price = market_data.get("price", 0.0)
        if current_price < 20000:
            return ("BUY", 0.0005)
        else:
            return ("HOLD", 0.0)
