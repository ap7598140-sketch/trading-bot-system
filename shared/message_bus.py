"""
Redis-based message bus for inter-bot communication.
All bots publish/subscribe through this module.

Design: each bot owns one MessageBus instance (created in BaseBot.__init__).
Two completely separate Redis connections are created in connect():
  - self._redis       : connection pool for publish / get_state / set_state
                        health_check_interval=30 keeps it alive during quiet periods
  - self._pubsub_conn : single dedicated client for pubsub only
                        health_check_interval=0 — background pings must not race
                        with the blocking pubsub.listen() async generator; OS-level
                        TCP keepalive handles the socket instead

Both connections use socket_keepalive=True so the OS sends TCP keepalive probes
every 60 seconds, preventing NAT/firewall timeout mid-session.

listen() fully recreates the pubsub connection on every error (not just
unsubscribe/resubscribe) so a dead underlying socket is never reused.
"""

import json
import asyncio
import socket as _socket

import redis.asyncio as aioredis
from typing import Callable, Any
from loguru import logger
from config import RedisConfig


def _redis_url() -> str:
    return f"redis://{RedisConfig.HOST}:{RedisConfig.PORT}/{RedisConfig.DB}"


# OS-level TCP keepalive probe settings (Linux).  Guards against silent
# NAT/firewall drops that leave sockets open but unresponsive.
_KEEPALIVE_OPTS: dict[int, int] = {}
for _attr, _val in [("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)]:
    if hasattr(_socket, _attr):
        _KEEPALIVE_OPTS[getattr(_socket, _attr)] = _val


def _pool_kwargs() -> dict:
    """Shared connection kwargs for both pool and pubsub clients."""
    return {
        "decode_responses":        True,
        "socket_keepalive":        True,
        "socket_keepalive_options": _KEEPALIVE_OPTS,
        "socket_connect_timeout":  10,
        "socket_timeout":          30,
        "retry_on_timeout":        True,
    }


class MessageBus:
    """Async Redis pub/sub message bus.  One instance per bot."""

    def __init__(self):
        self._redis: aioredis.Redis | None       = None   # publish + state
        self._pubsub_conn: aioredis.Redis | None = None   # dedicated pubsub
        self._pubsub                             = None
        self._subscriptions: dict[str, list[Callable]] = {}
        self._keepalive_task: asyncio.Task | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self):
        # Pool for publish / get / set  (concurrent-safe, pooled).
        # health_check_interval=30: redis-py pings the pool every 30 s so idle
        # connections are refreshed before commands fail.
        self._redis = await aioredis.from_url(
            _redis_url(),
            health_check_interval=30,
            **_pool_kwargs(),
        )
        await self._new_pubsub_conn()
        logger.info("MessageBus connected to Redis")

        # Background keepalive ping on the pubsub socket every 30 s.
        # Supplements OS TCP keepalive for application-layer visibility.
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def disconnect(self):
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        await self._close_pubsub()
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

        On any connection error:
          1. Close and recreate the pubsub connection entirely (never reuse a
             dead socket — unsubscribe alone is not enough).
          2. Re-subscribe to all channels.
          3. Resume listening.
        Only asyncio.CancelledError stops the loop.
        """
        if not self._pubsub or not self._subscriptions:
            return

        channels = list(self._subscriptions.keys())

        while True:
            try:
                await self._pubsub.subscribe(*channels)
                logger.info(f"MessageBus subscribed on: {channels}")

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
                return
            except Exception as e:
                logger.error(
                    f"MessageBus listen() error: {e} — "
                    f"pub/sub dropped, reconnecting in 5s (channels={channels})"
                )
                await asyncio.sleep(5)
                # Fully recreate — a dead socket cannot be recovered by unsubscribe
                await self._close_pubsub()
                await self._new_pubsub_conn()
                logger.info("MessageBus pubsub connection recreated — resubscribing")

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

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _new_pubsub_conn(self):
        """Create a fresh pubsub Redis client and pubsub object."""
        # health_check_interval=0: background pings must not race with
        # the async generator in listen().  OS TCP keepalive handles the socket.
        self._pubsub_conn = await aioredis.from_url(
            _redis_url(),
            health_check_interval=0,
            **_pool_kwargs(),
        )
        self._pubsub = self._pubsub_conn.pubsub(ignore_subscribe_messages=True)

    async def _close_pubsub(self):
        """Tear down the pubsub object and its backing connection."""
        if self._pubsub:
            try:
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

    async def _keepalive_loop(self):
        """Ping the pool connection every 30 s — surfaces dead connections early."""
        while True:
            await asyncio.sleep(30)
            if self._redis:
                try:
                    await self._redis.ping()
                except Exception as e:
                    logger.warning(f"MessageBus keepalive ping failed: {e}")
