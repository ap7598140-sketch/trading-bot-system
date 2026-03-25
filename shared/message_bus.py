"""
Redis-based message bus for inter-bot communication.
All bots publish/subscribe through this module.
"""

import json
import asyncio
import redis.asyncio as aioredis
from typing import Callable, Any
from loguru import logger
from config import RedisConfig


class MessageBus:
    """Async Redis pub/sub message bus."""

    def __init__(self):
        self._redis: aioredis.Redis | None = None
        self._pubsub = None
        self._subscriptions: dict[str, list[Callable]] = {}

    async def connect(self):
        self._redis = await aioredis.from_url(
            f"redis://{RedisConfig.HOST}:{RedisConfig.PORT}/{RedisConfig.DB}",
            decode_responses=True,
        )
        self._pubsub = self._redis.pubsub()
        logger.info("MessageBus connected to Redis")

    async def disconnect(self):
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.aclose()

    async def publish(self, channel: str, data: dict):
        """Publish a message to a channel."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        payload = json.dumps(data)
        await self._redis.publish(channel, payload)

    async def subscribe(self, channel: str, handler: Callable):
        """Subscribe to a channel with an async handler."""
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(handler)

    async def listen(self):
        """Start listening for messages on all subscribed channels."""
        if not self._pubsub or not self._subscriptions:
            return
        channels = list(self._subscriptions.keys())
        await self._pubsub.subscribe(*channels)
        logger.info(f"MessageBus listening on: {channels}")
        async for message in self._pubsub.listen():
            if message["type"] == "message":
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

    async def set_state(self, key: str, value: Any, ttl: int = 300):
        """Store bot state in Redis with optional TTL (seconds)."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        await self._redis.set(key, json.dumps(value), ex=ttl)

    async def get_state(self, key: str) -> Any:
        """Retrieve bot state from Redis."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def get_all_states(self, pattern: str) -> dict:
        """Get all state keys matching a pattern."""
        if not self._redis:
            raise RuntimeError("MessageBus not connected")
        keys = await self._redis.keys(pattern)
        result = {}
        for key in keys:
            result[key] = await self.get_state(key)
        return result


# Singleton instance
bus = MessageBus()
