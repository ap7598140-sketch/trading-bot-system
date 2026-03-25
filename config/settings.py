"""
Global configuration for the AI Trading Bot System.
All bots import from here for consistent settings.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()


# ── Model assignments ──────────────────────────────────────────────────────────
class Models:
    HAIKU   = "claude-haiku-4-5-20251001"   # fast, cheap, high-frequency
    SONNET  = "claude-sonnet-4-6"            # live trading decisions
    OPUS    = "claude-opus-4-6"              # overnight deep analysis


# ── Alpaca ─────────────────────────────────────────────────────────────────────
class AlpacaConfig:
    API_KEY    = os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    PAPER      = os.getenv("PAPER_TRADING", "true").lower() == "true"


# ── Anthropic ──────────────────────────────────────────────────────────────────
class AnthropicConfig:
    API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
    MAX_TOKENS = 4096


# ── Redis (inter-bot message bus) ─────────────────────────────────────────────
class RedisConfig:
    HOST = os.getenv("REDIS_HOST", "localhost")
    PORT = int(os.getenv("REDIS_PORT", "6379"))
    DB   = int(os.getenv("REDIS_DB", "0"))

    # Channel names
    CHANNEL_MARKET_DATA   = "market_data"
    CHANNEL_SIGNALS       = "signals"
    CHANNEL_RISK          = "risk_updates"
    CHANNEL_ORDERS        = "orders"
    CHANNEL_ALERTS        = "alerts"
    CHANNEL_NEWS          = "news_sentiment"
    CHANNEL_OPTIONS_FLOW  = "options_flow"
    CHANNEL_MOMENTUM      = "momentum"
    CHANNEL_STRATEGY      = "strategy"
    CHANNEL_BACKTEST      = "backtest_results"
    CHANNEL_RESEARCH      = "research"
    CHANNEL_HEARTBEAT     = "heartbeat"


# ── Risk limits ────────────────────────────────────────────────────────────────
class RiskConfig:
    MAX_POSITION_SIZE   = float(os.getenv("MAX_POSITION_SIZE", "10000"))
    MAX_PORTFOLIO_RISK  = float(os.getenv("MAX_PORTFOLIO_RISK", "0.02"))  # 2% per trade
    MAX_DAILY_LOSS      = 0.05    # 5% daily loss limit
    MAX_OPEN_POSITIONS  = 10
    STOP_LOSS_PCT       = 0.02    # 2% stop loss
    TAKE_PROFIT_PCT     = 0.04    # 4% take profit
    MAX_SECTOR_EXPOSURE = 0.30    # 30% in any one sector


# ── Trading universe ───────────────────────────────────────────────────────────
class UniverseConfig:
    WATCHLIST = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "SPY",  "QQQ",  "AMD",
        "NFLX", "AAPL", "JPM",  "BAC",  "GS",
    ]
    SCAN_INTERVAL_SECONDS = 300  # how often momentum scanner runs (5 min)
    DATA_REFRESH_SECONDS  = 120  # how often data agent refreshes (2 min)


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR   = os.path.join(BASE_DIR, "logs")
DATA_DIR  = os.path.join(BASE_DIR, "data")

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ── Alerts ─────────────────────────────────────────────────────────────────────
class AlertConfig:
    EMAIL           = os.getenv("ALERT_EMAIL", "")
    SLACK_WEBHOOK   = os.getenv("SLACK_WEBHOOK_URL", "")
    NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "")
