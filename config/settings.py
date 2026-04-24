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
    CHANNEL_CONTROL       = "system_control"   # Telegram controller → Bot 8


# ── Risk limits ────────────────────────────────────────────────────────────────
class RiskConfig:
    MAX_POSITION_SIZE        = float(os.getenv("MAX_POSITION_SIZE", "10000"))
    MAX_PORTFOLIO_RISK       = float(os.getenv("MAX_PORTFOLIO_RISK", "0.025"))
    MAX_DAILY_LOSS           = 0.05    # 5% daily loss limit (portfolio %)
    MAX_OPEN_POSITIONS       = 10
    MAX_DAILY_TRADES         = 30

    # ── Position sizing by grade ─────────────────────────────────────────────
    GRADE_A_POSITION_USD     = 3000.0  # Grade A signal → $3,000 position
    GRADE_B_POSITION_USD     = 2000.0  # Grade B signal → $2,000 position
    MIN_SINGLE_POSITION_USD  = 1500.0  # floor after regime scaling
    MAX_SINGLE_POSITION_USD  = 3000.0  # ceiling (Grade A)

    # ── Stop / take-profit ────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD     = 0.80
    STOP_LOSS_PCT            = 0.015   # 1.5% → ~$45 on $3k, ~$30 on $2k
    TAKE_PROFIT_PCT          = 0.03    # 3%   → 2:1 RR

    # ── Partial close + trailing stop ─────────────────────────────────────────
    PARTIAL_CLOSE_TRIGGER_PCT     = 0.02    # sell 50% of position at +2%
    PARTIAL_CLOSE_PCT             = 0.50    # fraction to close at first target
    TRAILING_STOP_USD             = 15.0    # $15 trailing stop on remainder
    TRAILING_ACTIVATE_PROFIT_USD  = 40.0    # activate trailing stop when up $40

    # ── Compound winners ──────────────────────────────────────────────────────
    COMPOUND_TRIGGER_PROFIT_USD   = 50.0    # add 50% more when up $50

    # ── Daily caps ────────────────────────────────────────────────────────────
    MAX_TRADE_LOSS_USD       = 75.0    # max loss per individual trade
    DAILY_LOSS_LIMIT_USD     = 150.0   # halt all trading at -$150/day
    DAILY_PROFIT_TARGET_USD  = 100.0   # stop new trades at +$100/day (lock it in)

    # ── Signal quality ────────────────────────────────────────────────────────
    MIN_STOCK_SCORE          = 70      # premarket score required (1-100)
    MIN_WINNER_CRITERIA      = 4       # of 6 criteria that must pass
    VOLUME_CONFIRMATION_X    = 2.0     # require 2× average volume
    MIN_GRADE_A_PROFIT_USD   = 75.0    # minimum target $ profit for Grade A
    MIN_GRADE_B_PROFIT_USD   = 50.0    # minimum target $ profit for Grade B
    MIN_REWARD_RISK_RATIO    = 2.0     # always 2:1
    MAX_SECTOR_EXPOSURE      = 0.30


# ── Trading windows ────────────────────────────────────────────────────────────
class TradingWindowConfig:
    """Only open new positions during high-momentum windows (EST)."""
    MORNING_OPEN    = (9,  45)   # 9:45am
    MORNING_CLOSE   = (10, 30)   # 10:30am
    AFTERNOON_OPEN  = (14,  0)   # 2:00pm
    AFTERNOON_CLOSE = (15,  0)   # 3:00pm
    SKIP_START      = (11,  0)   # dead zone start
    SKIP_END        = (13,  0)   # dead zone end
    EOD_CLOSE       = (15, 50)   # hard close time
    TIMEZONE        = "America/New_York"

    # String forms for display / comparison convenience
    MORNING_START   = "09:45"
    MORNING_END     = "10:30"
    AFTERNOON_START = "14:00"
    AFTERNOON_END   = "15:00"
    SKIP_START_STR  = "11:00"
    SKIP_END_STR    = "13:00"
    EOD_CLOSE_STR   = "15:50"


# ── Trading universe ───────────────────────────────────────────────────────────
class UniverseConfig:
    # Inverse ETFs for bear/crash regime
    BEAR_ETFS = ["SQQQ", "SPXS", "SOXS"]

    # Core 10 always-on fast movers (+ inverse ETFs kept for bear regime)
    WATCHLIST = [
        "NVDA", "TSLA", "META", "AMD", "GOOGL",
        "AMZN", "AAPL", "MSFT", "SPY",  "QQQ",
        "SQQQ", "SPXS", "SOXS",
    ]

    # ── ~300-symbol scan universe (filters price $10-$500 at runtime) ──────────
    SCAN_UNIVERSE = [
        # ── Mega-cap tech ──────────────────────────────────────────────────────
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        # ── Semiconductors ────────────────────────────────────────────────────
        "AMD", "INTC", "MU", "QCOM", "ARM", "SMCI", "MRVL", "KLAC",
        "LRCX", "AMAT", "NXPI", "ON", "TXN", "ADI", "MPWR", "WOLF",
        # ── Software / Cloud / Cybersecurity ─────────────────────────────────
        "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR", "PANW", "CRWD",
        "ZS", "FTNT", "S", "NET", "DDOG", "OKTA", "HUBS", "MDB",
        "TEAM", "GTLB", "PATH", "U", "RBLX", "COIN",
        # ── Finance / Fintech ─────────────────────────────────────────────────
        "V", "MA", "SCHW", "PYPL", "SQ", "AFRM", "SOFI", "NU",
        "IBKR", "ICE", "CME", "MELI", "HOOD", "XP",
        # ── Consumer / Retail ─────────────────────────────────────────────────
        "COST", "WMT", "TGT", "NKE", "SBUX", "MCD", "HD", "LOW",
        "SHOP", "ETSY", "W", "DG", "DLTR", "ROSS", "TJX", "CHWY",
        # ── Streaming / Media / Entertainment ────────────────────────────────
        "NFLX", "DIS", "SPOT", "WBD", "CMCSA", "PARA",
        # ── Healthcare / Pharma ───────────────────────────────────────────────
        "LLY", "UNH", "PFE", "ABBV", "MRK", "AMGN", "MRNA", "BIIB",
        "REGN", "GILD", "VRTX", "BMY", "CVS", "CI", "HUM", "ELV",
        # ── Biotech ───────────────────────────────────────────────────────────
        "BNTX", "SGEN", "INCY", "EXAS", "RARE", "SRPT", "ALNY",
        # ── Energy ────────────────────────────────────────────────────────────
        "XOM", "CVX", "COP", "OXY", "EOG", "DVN", "FANG", "MPC",
        "PSX", "VLO", "SLB", "HAL", "BKR",
        # ── Industrials / Defense ─────────────────────────────────────────────
        "CAT", "DE", "HON", "GE", "ETN", "ROK", "EMR", "ITW",
        "LMT", "RTX", "NOC", "GD", "BA",
        # ── Transportation ────────────────────────────────────────────────────
        "UNP", "CSX", "NSC", "UPS", "FDX", "DAL", "AAL", "UAL", "LUV",
        # ── EV / Autos ────────────────────────────────────────────────────────
        "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI",
        # ── Travel / Hotels ───────────────────────────────────────────────────
        "ABNB", "BKNG", "EXPE", "MAR", "HLT", "CCL", "RCL", "NCLH",
        # ── Crypto-linked ─────────────────────────────────────────────────────
        "MSTR", "RIOT", "MARA", "CLSK",
        # ── Chinese / Intl tech ───────────────────────────────────────────────
        "BABA", "JD", "PDD", "BIDU", "SE",
        # ── Real Estate / REITs ───────────────────────────────────────────────
        "AMT", "EQIX", "PLD", "SPG", "CCI", "SBAC", "DLR",
        # ── Telecom ───────────────────────────────────────────────────────────
        "T", "VZ", "TMUS",
        # ── Utilities ─────────────────────────────────────────────────────────
        "NEE", "DUK", "SO", "AES",
        # ── Materials ─────────────────────────────────────────────────────────
        "FCX", "NEM", "GOLD", "AA", "NUE", "CLF",
        # ── Sector ETFs (for sector-strength check) ───────────────────────────
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        # ── Broad market ETFs ─────────────────────────────────────────────────
        "SPY", "QQQ", "IWM", "ARKK",
        # ── Inverse ETFs (bear/crash regime) ─────────────────────────────────
        "SQQQ", "SPXS", "SOXS",
    ]

    # How many stocks to actively trade from the daily 26-stock watchlist
    MAX_DAILY_ACTIVE_TRADES = 10

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


# ── Circuit Breaker thresholds ────────────────────────────────────────────────
class CircuitBreakerConfig:
    DAILY_HALVE_PCT    = 0.02   # down 2% today   → halve position sizes
    DAILY_CLOSE_PCT    = 0.03   # down 3% today   → close all positions
    WEEKLY_QUARTER_PCT = 0.05   # down 5% week    → quarter position sizes
    PEAK_LOCKOUT_PCT   = 0.10   # down 10% peak   → write LOCKFILE.lock
    PEAK_EMERGENCY_PCT = 0.15   # down 15% peak   → emergency exit + alert


# ── HMM Regime Detection ──────────────────────────────────────────────────────
class RegimeConfig:
    TRAIN_DAYS              = 504   # ~2 years of daily bars for HMM training
    REGIME_LOOKBACK_DAYS    = 504   # alias used by regime_detector
    REGIME_MIN_STATES       = 3
    REGIME_MAX_STATES       = 7
    STABILITY_BARS          = 3     # consecutive bars required before regime acts
    REGIME_STABILITY_BARS   = 3     # alias
    FLICKER_WINDOW          = 20    # look-back bars for flicker detection
    REGIME_FLICKER_WINDOW   = 20    # alias
    FLICKER_THRESH          = 4     # changes in FLICKER_WINDOW → uncertain
    REGIME_FLICKER_THRESHOLD = 4    # alias
    RETRAIN_HOURS           = 24    # retrain every 24 h (nightly)
    # Allocation scales per regime (multiplied against position sizing)
    ALLOC_EUPHORIA  = 0.70
    ALLOC_BULL      = 0.95
    ALLOC_NEUTRAL   = 0.60
    ALLOC_BEAR      = 0.50
    ALLOC_CRASH     = 0.00
    ALLOC_UNCERTAIN = 0.50  # multiplier applied on top of regime scale when flickering


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
