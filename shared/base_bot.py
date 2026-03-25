"""
BaseBot: abstract base class every bot inherits from.
Provides lifecycle management, logging, heartbeat, and message bus wiring.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from loguru import logger
import sys
import os

from config import RedisConfig, LOG_DIR
from shared.message_bus import MessageBus


class BaseBot(ABC):
    """
    Every bot inherits from BaseBot.
    Subclasses must implement:
        - async setup()   → one-time initialisation
        - async run()     → main loop
        - async cleanup() → graceful shutdown
    """

    def __init__(self, bot_id: int, name: str, model: str):
        self.bot_id   = bot_id
        self.name     = name
        self.model    = model
        self.running  = False
        self.bus      = MessageBus()   # each bot owns its own connection

        # Configure per-bot log file
        log_file = os.path.join(LOG_DIR, f"bot{bot_id:02d}_{name.lower().replace(' ', '_')}.log")
        logger.add(log_file, rotation="100 MB", retention="7 days",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   filter=lambda r: True)
        self.logger = logger.bind(bot=self.name)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        """Connect to bus, run setup, start heartbeat, then run main loop."""
        self.running = True
        await self.bus.connect()
        self.logger.info(f"Bot {self.bot_id} '{self.name}' starting | model={self.model}")
        await self.setup()
        asyncio.create_task(self._heartbeat())
        try:
            await self.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Unhandled exception: {e}")
        finally:
            await self.stop()

    async def stop(self):
        self.running = False
        self.logger.info(f"Bot {self.bot_id} '{self.name}' shutting down")
        await self.cleanup()
        await self.bus.disconnect()

    @abstractmethod
    async def setup(self): ...

    @abstractmethod
    async def run(self): ...

    @abstractmethod
    async def cleanup(self): ...

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    async def _heartbeat(self):
        while self.running:
            await self.bus.publish(RedisConfig.CHANNEL_HEARTBEAT, {
                "bot_id":    self.bot_id,
                "name":      self.name,
                "status":    "alive",
                "timestamp": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(30)

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def publish(self, channel: str, data: dict):
        data.setdefault("source_bot", self.bot_id)
        data.setdefault("timestamp",  datetime.utcnow().isoformat())
        await self.bus.publish(channel, data)

    async def save_state(self, key: str, value, ttl: int = 300):
        await self.bus.set_state(f"bot{self.bot_id}:{key}", value, ttl)

    async def load_state(self, key: str):
        return await self.bus.get_state(f"bot{self.bot_id}:{key}")

    def log(self, msg: str, level: str = "info"):
        getattr(self.logger, level)(msg)
