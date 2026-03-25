"""
Redis-based message bus for inter-bot communication.
All bots publish/subscribe through this module.

Design: each bot owns one MessageBus instance (created in BaseBot.__init__).
Two completely separate Redis connections are created in connect():
  - self._redis       : connection pool for publish / get_state / set_state
  - self._pubsub_conn : single dedicated client for pubsub only
                        (health_check_interval=0 prevents background reads
                         from racing with the blocking pubsub.listen() loop)
listen() uses self._pubsub which was set up in connect() — it never creates
connections lazily, so there is no race between task start and socket setup.
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
    """Async Redis pub/sub message bus.  One instance per bot."""

    def __init__(self):
        self._redis: aioredis.Redis | None = None          # publish + state
        self._pubsub_conn: aioredis.Redis | None = None    # dedicated pubsub
        self._pubsub = None
        self._subscriptions: dict[str, list[Callable]] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self):
        # Pool for publish / get / set  (concurrent-safe, pooled)
        self._redis = await aioredis.from_url(
            _redis_url(),
            decode_responses=True,
        )
        # Dedicated single client for pubsub reads only.
        # health_check_interval=0 disables background pings that would race
        # with the blocking pubsub.listen() async generator.
        self._pubsub_conn = await aioredis.from_url(
            _redis_url(),
            decode_responses=True,
            health_check_interval=0,
        )
        self._pubsub = self._pubsub_conn.pubsub(ignore_subscribe_messages=True)
        logger.info("MessageBus connected to Redis")

    async def disconnect(self):
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe()
                await self._pubsub.aclose()
            except Exception:
                pass
            self._pubsub = None
        if self._pubsub_conn:
            try:
                await self._pubsub_conn.aclose()
            except Exception:
                pass
            self._pubsub_conn = None
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

    # ── Publish ────────────────────────────────────────────────────────────────

    async def publish(self, channel: str, data: dict):
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        await self._redis.publish(channel, json.dumps(data))

    # ── Subscribe / listen ────────────────────────────────────────────────────

    async def subscribe(self, channel: str, handler: Callable):
        """Register a handler for a channel.  Call before listen()."""
        self._subscriptions.setdefault(channel, []).append(handler)

    async def listen(self):
        """
        Listen on all subscribed channels using the dedicated pubsub connection
        created in connect().  Only one coroutine ever reads from this socket.
        """
        if not self._pubsub or not self._subscriptions:
            return

        channels = list(self._subscriptions.keys())
        await self._pubsub.subscribe(*channels)
        logger.info(f"MessageBus listening on: {channels}")

        try:
            async for message in self._pubsub.listen():
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
        except asyncio.CancelledError:
            pass

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
