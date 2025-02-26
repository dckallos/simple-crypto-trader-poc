# In sentiment_analysis.py
import requests
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Initialize the sentiment analysis pipeline
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
except Exception as e:
    logger.exception("[Sentiment] Error initializing Hugging Face pipeline: {e}")
    sentiment_pipeline = None


def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of the given text using Hugging Face's transformers.

    Args:
        text (str): Text to analyze (e.g., news headline).

    Returns:
        dict: Dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence).
              Returns {'label': 'NEUTRAL', 'score': 0.0} on error.
    """
    if not sentiment_pipeline:
        logger.error("[Sentiment] Sentiment pipeline not initialized.")
        return {"label": "NEUTRAL", "score": 0.0}

    if not text or not isinstance(text, str):
        logger.warning("[Sentiment] Invalid text input provided.")
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        result = sentiment_pipeline(text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }
    except Exception as e:
        logger.exception(f"[Sentiment] Error analyzing sentiment for text: {text[:50]}...: {e}")
        return {"label": "NEUTRAL", "score": 0.0}


def fetch_crypto_news(terms: list, api_key: str) -> list:
    """
    Fetches recent news headlines for a list of terms using NewsAPI.
    The terms are combined with "OR" to capture articles mentioning any of the provided keywords.

    Args:
        terms (list): List of search terms (e.g., ["ETH/USD", "ETH", "Ethereum"]).
        api_key (str): NewsAPI key.

    Returns:
        list: List of news headlines (titles). Returns an empty list on error or if no terms/api_key provided.
    """
    # Check for missing API key
    if not api_key:
        logger.warning("[NewsAPI] API key not provided => unable to fetch news.")
        return []

    # Check for empty terms list
    if not terms:
        logger.warning("[NewsAPI] No terms provided for news query.")
        return []

    # Construct query by joining terms with " OR "
    query = " OR ".join(terms)  # e.g., "ETH/USD OR ETH OR Ethereum"
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "publishedAt",  # Sort by most recent
        "pageSize": 5  # Limit to 5 articles to respect free-tier rate limits
    }

    try:
        # Make the API request
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()  # Raise exception for HTTP errors
        articles = resp.json().get("articles", [])
        headlines = [article["title"] for article in articles if article.get("title")]
        logger.info(f"[NewsAPI] Fetched {len(headlines)} headlines for query: {query}")
        return headlines
    except requests.exceptions.RequestException as e:
        logger.exception(f"[NewsAPI] Error fetching news for query: {query}: {e}")
        return []


def compute_average_sentiment(texts: list) -> float:
    """
    Computes the average sentiment score from a list of texts.

    Args:
        texts (list): List of text strings (e.g., news headlines).

    Returns:
        float: Average sentiment score (positive scores > 0, negative scores < 0).
    """
    if not texts:
        logger.info("[Sentiment] No texts provided for sentiment analysis.")
        return 0.0

    scores = []
    for text in texts:
        sentiment = analyze_sentiment(text)
        if sentiment["label"] == "POSITIVE":
            scores.append(sentiment["score"])
        elif sentiment["label"] == "NEGATIVE":
            scores.append(-sentiment["score"])
        # NEUTRAL label scores are ignored (score = 0)

    average_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"[Sentiment] Computed average sentiment score: {average_score:.4f} from {len(scores)} texts.")
    return average_score