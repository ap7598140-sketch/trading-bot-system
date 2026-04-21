"""
System launcher – starts all 13 bots as supervised asyncio tasks.
Run: python scripts/run_system.py

Features:
  • System lock file — prevents multiple instances from running simultaneously
  • Per-bot supervisor — automatically restarts any bot that crashes (5 s delay)
  • Telegram crash alerts — notifies operator on every crash + restart
  • Graceful shutdown — SIGINT/SIGTERM cancels all supervisors cleanly
"""

import asyncio
import os
import signal
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from loguru import logger

from bots.bot01_news_sentiment      import NewsSentimentBot
from bots.bot02_options_flow        import OptionsFlowBot
from bots.bot03_momentum_scanner    import MomentumScanner
from bots.bot04_data_agent          import DataAgent
from bots.bot05_strategy_agent      import StrategyAgent
from bots.bot06_risk_agent          import RiskAgent
from bots.bot07_execution_agent     import ExecutionAgent
from bots.bot08_master_commander    import MasterCommander
from bots.bot09_alert_bot           import AlertBot
from bots.bot10_backtesting_bot     import BacktestingBot
from bots.bot11_strategy_builder    import StrategyBuilder
from bots.bot12_research_bot        import ResearchBot
from bots.bot13_telegram_controller import TelegramController
from config import BASE_DIR


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
    TelegramController,  # Bot 13 – operator control via Telegram
]

RESTART_DELAY = 5       # seconds before restarting a crashed bot
SYSTEM_LOCK   = os.path.join(BASE_DIR, "system.lock")


# ── System lock ────────────────────────────────────────────────────────────────

def _acquire_lock():
    """
    Write PID to system.lock.  Exit immediately if another instance is running.
    Removes stale lock files left by crashed processes.
    """
    if os.path.exists(SYSTEM_LOCK):
        try:
            pid = int(open(SYSTEM_LOCK).read().strip())
            os.kill(pid, 0)   # raises if process doesn't exist
            logger.error(
                f"System already running (PID {pid}). "
                f"Stop it first, or delete {SYSTEM_LOCK} to force-start."
            )
            sys.exit(1)
        except (ProcessLookupError, ValueError, OSError):
            logger.warning(f"Stale lock file found (dead PID) — removing and continuing")
            os.remove(SYSTEM_LOCK)

    with open(SYSTEM_LOCK, "w") as fh:
        fh.write(str(os.getpid()))
    logger.info(f"System lock acquired (PID {os.getpid()}) → {SYSTEM_LOCK}")


def _release_lock():
    try:
        if os.path.exists(SYSTEM_LOCK):
            pid = int(open(SYSTEM_LOCK).read().strip())
            if pid == os.getpid():
                os.remove(SYSTEM_LOCK)
                logger.info("System lock released")
    except Exception:
        pass


# ── Telegram helper ────────────────────────────────────────────────────────────

async def _telegram(text: str):
    """Fire-and-forget Telegram message — never raises."""
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            )
    except Exception:
        pass


# ── Per-bot supervisor ─────────────────────────────────────────────────────────

async def _supervise(BotClass, stop_event: asyncio.Event):
    """
    Run BotClass continuously.  On crash: log + Telegram alert + wait 5 s + restart.
    Exits cleanly when stop_event is set or the task is cancelled.
    """
    name = BotClass.__name__

    while not stop_event.is_set():
        bot = BotClass()
        try:
            await bot.start()
            # start() returned normally → graceful shutdown requested
            break

        except asyncio.CancelledError:
            break

        except Exception as exc:
            if stop_event.is_set():
                break   # shutdown in progress — don't restart
            err_msg = str(exc)[:300]
            logger.error(f"[{name}] CRASHED: {err_msg}")
            await _telegram(
                f"🔴 <b>{name} crashed</b>\n"
                f"<code>{err_msg}</code>\n"
                f"Restarting in {RESTART_DELAY} s..."
            )
            # Wait with stop_event check so shutdown isn't delayed
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=RESTART_DELAY
                )
                break   # stop was requested during the wait
            except asyncio.TimeoutError:
                pass    # normal — restart delay elapsed

            logger.warning(f"[{name}] restarting now")
            await _telegram(f"🟡 <b>{name} restarting...</b>")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | "
               "<cyan>{extra[bot]}</cyan> | {message}",
        filter=lambda r: "bot" in r["extra"],
    )
    logger.add("logs/system.log", rotation="100 MB", retention="7 days")

    logger.info("=" * 60)
    logger.info("  AI Trading Bot System – 13 Bots")
    logger.info("  Paper trading: Alpaca")
    logger.info("  Crash recovery: ENABLED (restart delay = 5 s)")
    logger.info("=" * 60)

    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(_supervise(BotClass, stop_event), name=BotClass.__name__)
        for BotClass in BOTS
    ]

    loop = asyncio.get_event_loop()

    def _shutdown():
        logger.warning("Shutdown signal received — stopping all bots")
        stop_event.set()
        for task in tasks:
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await _telegram(
        f"🟢 <b>Trading System Started</b>\n"
        f"{len(BOTS)} bots launching | crash recovery enabled"
    )

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as exc:
        logger.error(f"System error: {exc}")
    finally:
        _release_lock()
        logger.info("All bots stopped")
        await _telegram("⚫ <b>Trading System Stopped</b>")


if __name__ == "__main__":
    _acquire_lock()
    try:
        asyncio.run(main())
    finally:
        _release_lock()
