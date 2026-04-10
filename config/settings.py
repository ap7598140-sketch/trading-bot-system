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
    MAX_POSITION_SIZE       = float(os.getenv("MAX_POSITION_SIZE", "10000"))
    MAX_PORTFOLIO_RISK      = float(os.getenv("MAX_PORTFOLIO_RISK", "0.025"))
    MAX_DAILY_LOSS          = 0.05    # 5% daily loss limit
    MAX_OPEN_POSITIONS      = 10      # max simultaneous open positions
    MAX_DAILY_TRADES        = 20      # max orders submitted per day
    MAX_SINGLE_POSITION_USD = 1000.0  # hard $1,000 cap per individual trade
    CONFIDENCE_THRESHOLD    = 0.55    # minimum confidence to approve a trade
    STOP_LOSS_PCT           = 0.01    # 1% stop loss
    TAKE_PROFIT_PCT         = 0.02    # 2% take profit  →  2:1 reward/risk
    MAX_SECTOR_EXPOSURE     = 0.30    # 30% in any one sector


# ── Trading universe ───────────────────────────────────────────────────────────
class UniverseConfig:
    # Core watchlist (continuously monitored)
    WATCHLIST = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "SPY",  "QQQ",  "AMD",
        "NFLX", "JPM",  "BAC",  "GS",   "COIN",
    ]

    # Expanded universe for the 9am morning scan (top volume/gap candidates)
    SCAN_UNIVERSE = [
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        # Semiconductors
        "AMD", "INTC", "MU", "QCOM", "ARM", "SMCI",
        # Software / Cloud
        "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR", "PANW",
        # Finance
        "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "SCHW", "COIN",
        # Energy
        "XOM", "CVX", "COP", "OXY",
        # Healthcare / Pharma / Biotech
        "LLY", "UNH", "PFE", "ABBV", "MRK", "AMGN", "MRNA", "BIIB",
        # Consumer
        "COST", "WMT", "TGT", "NKE", "SBUX", "MCD",
        # Streaming / Media
        "NFLX", "DIS", "SPOT",
        # EV / Autos
        "F", "GM", "RIVN",
        # Crypto-linked
        "HOOD", "MSTR",
        # Sector ETFs (for sector-strength check)
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        # Broad market ETFs
        "SPY", "QQQ", "IWM",
    ]

    # Sector ETF → member symbols mapping (for sector filter in bot05)
    SECTOR_ETFS = {
        "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM", "ADBE", "ORCL", "NOW", "QCOM"],
        "XLF": ["JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "SCHW", "COIN"],
        "XLE": ["XOM", "CVX", "COP", "OXY", "SLB"],
        "XLV": ["LLY", "UNH", "PFE", "ABBV", "MRK", "AMGN", "MRNA"],
        "XLI": ["GE", "HON", "CAT", "DE", "UPS", "FDX"],
        "XLY": ["AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD", "COST"],
    }

    SCAN_INTERVAL_SECONDS = 600  # how often momentum scanner runs (10 min)
    DATA_REFRESH_SECONDS  = 300  # how often data agent refreshes (5 min)


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR   = os.path.join(BASE_DIR, "logs")
DATA_DIR  = os.path.join(BASE_DIR, "data")

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ── Alerts ─────────────────────────────────────────────────────────────────────
class AlertConfig:
    EMAIL              = os.getenv("ALERT_EMAIL", "")
    SLACK_WEBHOOK      = os.getenv("SLACK_WEBHOOK_URL", "")
    NEWS_API_KEY       = os.getenv("NEWS_API_KEY", "")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
