# AI Trading Bot System – 12 Bots

A multi-agent AI trading system built on the Anthropic Claude API and Alpaca paper trading.

## Architecture

```
                        ┌─────────────────────┐
                        │  Bot 8: Master       │
                        │  Commander (Sonnet)  │ ← System leader & health monitor
                        └──────────┬──────────┘
                                   │ Redis Message Bus
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  ┌───────▼───────┐      ┌─────────▼──────┐      ┌────────▼───────┐
  │ Bot 4: Data   │      │ Bot 5: Strategy│      │ Bot 6: Risk    │
  │ Agent (Haiku) │      │ Agent (Sonnet) │      │ Agent (Sonnet) │
  └───────┬───────┘      └─────────┬──────┘      └────────┬───────┘
          │                        │                       │
  ┌───────▼──────────────┐         │              ┌────────▼───────┐
  │ Market Data Feed     │         │              │ Bot 7: Execute │
  │ + Indicators         │         │              │ Agent (Haiku)  │
  └──────────────────────┘         │              └────────────────┘
                                   │
  ┌────────────────────────────────▼───────────────────────────┐
  │ Signal Producers (all feed into Strategy Agent)            │
  │  Bot 1: News Sentiment (Haiku)                             │
  │  Bot 2: Options Flow   (Sonnet)                            │
  │  Bot 3: Momentum Scanner (Haiku)                           │
  └────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ Overnight Deep Analysis (Opus 4.6)                         │
  │  Bot 10: Backtesting Bot   – simulates strategies          │
  │  Bot 11: Strategy Builder  – generates new strategies      │
  │  Bot 12: Research Bot      – market regime & intelligence  │
  └────────────────────────────────────────────────────────────┘

  Bot 9: Alert Bot (Haiku) – unified notification gateway
```

## Model Assignment

| Model | Bots | Reason |
|-------|------|--------|
| Haiku 4.5 | 1, 3, 4, 7, 9 | Fast, cheap, high-frequency tasks |
| Sonnet 4.6 | 2, 5, 6, 8 | Live trading decisions, balanced cost/quality |
| Opus 4.6 | 10, 11, 12 | Overnight deep analysis, maximum capability |

## Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start Redis
redis-server

# 4. Run the system
python scripts/run_system.py

# 5. Open dashboard (separate terminal)
python scripts/dashboard.py
```

## Project Structure

```
trading-bot-system/
├── bots/
│   ├── bot01_news_sentiment.py    # Haiku 4.5
│   ├── bot02_options_flow.py      # Sonnet 4.6
│   ├── bot03_momentum_scanner.py  # Haiku 4.5
│   ├── bot04_data_agent.py        # Haiku 4.5  ← starts first
│   ├── bot05_strategy_agent.py    # Sonnet 4.6
│   ├── bot06_risk_agent.py        # Sonnet 4.6
│   ├── bot07_execution_agent.py   # Haiku 4.5
│   ├── bot08_master_commander.py  # Sonnet 4.6 ← system leader
│   ├── bot09_alert_bot.py         # Haiku 4.5
│   ├── bot10_backtesting_bot.py   # Opus 4.6
│   ├── bot11_strategy_builder.py  # Opus 4.6
│   └── bot12_research_bot.py      # Opus 4.6
├── config/
│   └── settings.py                # All config in one place
├── shared/
│   ├── base_bot.py                # BaseBot abstract class
│   ├── message_bus.py             # Redis pub/sub wrapper
│   └── alpaca_client.py           # Alpaca SDK wrapper
├── scripts/
│   ├── run_system.py              # Launch all 12 bots
│   └── dashboard.py               # Terminal dashboard
├── logs/                          # Per-bot log files
├── data/                          # Local data cache
└── tests/
```

## Data Flow

```
Alpaca → Bot 4 (Data Agent) → Redis CHANNEL_MARKET_DATA
                                       ↓
                         Bot 3 (Momentum) + Bot 2 (Options)
                                       ↓
News API → Bot 1 (News) → CHANNEL_NEWS
                                       ↓
                         Bot 5 (Strategy Agent) → CHANNEL_STRATEGY
                                       ↓
                         Bot 6 (Risk Agent) → CHANNEL_ORDERS
                                       ↓
                         Bot 7 (Execution) → Alpaca orders
                                       ↓
                         Bot 9 (Alerts) → Slack/Email
```

## Risk Controls

- Max position size: $10,000 per trade
- Max portfolio risk: 2% per trade
- Stop loss: 2% per position
- Take profit: 4% per position
- Daily loss limit: 5% (auto-halt)
- Max open positions: 10
- All trading halted outside market hours

## Running Individual Bots

```bash
# Run a single bot for testing
python bots/bot04_data_agent.py
python bots/bot08_master_commander.py
```
