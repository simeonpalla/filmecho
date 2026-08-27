"""Filmecho agents package.

Exposes each agent so `adk web` / `adk run` can discover them directly
from this package if you want to test one in isolation, outside the full
orchestration pipeline.
"""

from agents.cast_agent import cast_agent
from agents.competitive_agent import competitive_agent
from agents.main_synthesis import main_synthesis_agent
from agents.marketing_agent import marketing_agent
from agents.news_cast_agent import news_cast_agent
from agents.sentiment_synthesis import sentiment_synthesis_agent
from agents.web_sentiment_agent import web_sentiment_agent

__all__ = [
    "web_sentiment_agent",
    "competitive_agent",
    "news_cast_agent",
    "cast_agent",
    "marketing_agent",
    "sentiment_synthesis_agent",
    "main_synthesis_agent",
]
