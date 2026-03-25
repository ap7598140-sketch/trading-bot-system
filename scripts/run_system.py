"""
System launcher – starts all 12 bots as concurrent asyncio tasks.
Run: python scripts/run_system.py
"""

import asyncio
import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from bots.bot01_news_sentiment   import NewsSentimentBot
from bots.bot02_options_flow     import OptionsFlowBot
from bots.bot03_momentum_scanner import MomentumScanner
from bots.bot04_data_agent       import DataAgent
from bots.bot05_strategy_agent   import StrategyAgent
from bots.bot06_risk_agent       import RiskAgent
from bots.bot07_execution_agent  import ExecutionAgent
from bots.bot08_master_commander import MasterCommander
from bots.bot09_alert_bot        import AlertBot
from bots.bot10_backtesting_bot  import BacktestingBot
from bots.bot11_strategy_builder import StrategyBuilder
from bots.bot12_research_bot     import ResearchBot


BOTS = [
    DataAgent,           # Bot 4  – must start first (data backbone)
    MasterCommander,     # Bot 8  – system leader
    RiskAgent,           # Bot 6  – risk gatekeeper (before strategy/execution)
    StrategyAgent,       # Bot 5
    ExecutionAgent,      # Bot 7
    NewsSentimentBot,    # Bot 1
    OptionsFlowBot,      # Bot 2
    MomentumScanner,     # Bot 3
    AlertBot,            # Bot 9
    BacktestingBot,      # Bot 10
    StrategyBuilder,     # Bot 11
    ResearchBot,         # Bot 12
]


async def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | "
                      "<cyan>{extra[bot]}</cyan> | {message}",
               filter=lambda r: True)
    logger.add("logs/system.log", rotation="100 MB", retention="7 days")

    logger.info("=" * 60)
    logger.info("  AI Trading Bot System – 12 Bots")
    logger.info("  Paper trading: Alpaca")
    logger.info("=" * 60)

    # Instantiate all bots
    instances = [BotClass() for BotClass in BOTS]

    # Create tasks
    tasks = [asyncio.create_task(bot.start(), name=bot.name) for bot in instances]

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()

    def shutdown():
        logger.warning("Shutdown signal received")
        for task in tasks:
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("All bots stopped")


if __name__ == "__main__":
    asyncio.run(main())
