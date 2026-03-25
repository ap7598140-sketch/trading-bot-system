"""
Redis-based message bus for inter-bot communication.
All bots publish/subscribe through this module.

Each bot gets its own MessageBus instance (created in BaseBot.__init__).
publish/state ops use self._redis; listen() creates its own dedicated
pubsub connection so concurrent subscribers never share a socket.
"""

import json
import asyncio
import redis.asyncio as aioredis
from typing import Callable, Any
from loguru import logger
from config import RedisConfig


def _redis_url() -> str:
    return f"redis://{RedisConfig.HOST}:{RedisConfig.PORT}/{RedisConfig.DB}"


class MessageBus:
    """Async Redis pub/sub message bus. One instance per bot."""

    def __init__(self):
        self._redis: aioredis.Redis | None = None          # publish + state ops
        self._subscriptions: dict[str, list[Callable]] = {}

    async def connect(self):
        self._redis = await aioredis.from_url(
            _redis_url(), decode_responses=True,
        )
        logger.info("MessageBus connected to Redis")

    async def disconnect(self):
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    # ── Publish ────────────────────────────────────────────────────────────────

    async def publish(self, channel: str, data: dict):
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        await self._redis.publish(channel, json.dumps(data))

    # ── Subscribe / listen ────────────────────────────────────────────────────

    async def subscribe(self, channel: str, handler: Callable):
        """Register a handler for a channel (call before listen())."""
        self._subscriptions.setdefault(channel, []).append(handler)

    async def listen(self):
        """
        Start listening on all subscribed channels.
        Creates a *dedicated* Redis connection + pubsub so this bot's
        socket is never shared with another coroutine.
        """
        if not self._subscriptions:
            return

        # Dedicated connection for blocking pubsub reads
        pubsub_conn = await aioredis.from_url(
            _redis_url(), decode_responses=True,
        )
        pubsub = pubsub_conn.pubsub()
        channels = list(self._subscriptions.keys())
        await pubsub.subscribe(*channels)
        logger.info(f"MessageBus listening on: {channels}")

        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                channel = message["channel"]
                try:
                    data = json.loads(message["data"])
                except json.JSONDecodeError:
                    data = message["data"]
                for handler in self._subscriptions.get(channel, []):
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Handler error on {channel}: {e}")
        finally:
            await pubsub.aclose()
            await pubsub_conn.aclose()

    # ── State ──────────────────────────────────────────────────────────────────

    async def set_state(self, key: str, value: Any, ttl: int = 300):
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        await self._redis.set(key, json.dumps(value), ex=ttl)

    async def get_state(self, key: str) -> Any:
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        raw = await self._redis.get(key)
        return json.loads(raw) if raw is not None else None

    async def get_all_states(self, pattern: str) -> dict:
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        keys = await self._redis.keys(pattern)
        result = {}
        for key in keys:
            result[key] = await self.get_state(key)
        return result
