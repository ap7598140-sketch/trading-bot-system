"""
LLM Router — model dispatch with selective caching.

Rules:
  • Haiku by default; Sonnet for final trade/risk decisions (unrestricted)
  • In-memory response cache ONLY for Haiku news/data calls (TTL 10 min)
    Never cache trade decisions or price-based Sonnet responses
  • Minified JSON helper saves tokens without losing information
  • compress_system_state() — system health summary for bot08 (no prices)

Usage:
    router = LLMRouter(client, save_fn=self.save_state, get_fn=self.bus.get_state)
    # Trade decision — no cache, full context
    text = await router.call(messages, prefer="sonnet", max_tokens=600)
    # News/data — cache OK
    text = await router.call(messages, prefer="haiku", max_tokens=300, cache_key=k)
"""

import asyncio
import hashlib
import json
import time
from typing import Callable, Optional

import anthropic

from config import Models


NEWS_CACHE_TTL = 600   # 10 min — only for news/data Haiku calls


class LLMRouter:
    """
    Thin dispatch layer — route and optionally cache.
    Sonnet calls are never cached and never rate-limited.
    Haiku calls may be cached when the caller passes a cache_key.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        save_fn: Optional[Callable] = None,   # unused — kept for API compatibility
        get_fn:  Optional[Callable] = None,   # unused — kept for API compatibility
    ):
        self._client = client
        self._cache: dict[str, tuple[str, float]] = {}   # key → (text, expiry_mono)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def call(
        self,
        messages: list[dict],
        *,
        prefer:     str = "haiku",   # "haiku" | "sonnet"
        max_tokens: int = 300,
        cache_key:  str = "",        # only honoured for Haiku calls
    ) -> str:
        """
        Execute an LLM call.
        - Sonnet: always fires a fresh API call; cache_key is ignored.
        - Haiku: uses in-memory cache when cache_key is provided (TTL 10 min).
        Never raises — returns "" on error.
        """
        model = Models.SONNET if prefer == "sonnet" else Models.HAIKU

        # Cache only applies to Haiku; trade decisions (Sonnet) are always live
        if model == Models.HAIKU and cache_key:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return hit

        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.messages.create(
                    model=model, max_tokens=max_tokens, messages=messages
                ),
            )
            text = resp.content[0].text.strip()
        except Exception:
            return ""

        if model == Models.HAIKU and cache_key:
            self._cache_set(cache_key, text)
        return text

    # ── In-memory cache (Haiku only) ───────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry and time.monotonic() < entry[1]:
            return entry[0]
        self._cache.pop(key, None)
        return None

    def _cache_set(self, key: str, text: str):
        self._cache[key] = (text, time.monotonic() + NEWS_CACHE_TTL)
        if len(self._cache) > 200:
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def cache_key(*parts) -> str:
        """Stable MD5 hash of args — use as cache_key for Haiku calls."""
        raw = "|".join(
            json.dumps(p, sort_keys=True) if not isinstance(p, str) else p
            for p in parts
        )
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def j(d) -> str:
        """Minified JSON — no indent, no spaces — saves ~40% tokens vs indent=2."""
        return json.dumps(d, separators=(",", ":"))

    @staticmethod
    def compress_system_state(state: dict) -> str:
        """
        Compress system state to a single descriptive line (~50 tokens).
        Used in Master Commander health-check prompt. Contains no prices.
        """
        cb  = state.get("circuit_breaker", {})
        reg = state.get("regime", {})
        st  = state.get("strategy", {})
        nws = state.get("news", {})
        rsk = state.get("risk", {})
        pnl = state.get("daily_pnl_pct", 0)
        return (
            f"sess={state.get('market_session','?')} "
            f"pv={int(state.get('portfolio_value', 0))} "
            f"pnl={pnl:+.1f}% "
            f"cb=L{cb.get('level', 0)}({cb.get('action', 'ok')}) "
            f"regime={reg.get('name', '?')} "
            f"setups={st.get('setup_count', 0)} bias={st.get('market_bias', '?')} "
            f"bull={nws.get('bullish', 0)} bear={nws.get('bearish', 0)} "
            f"closs={rsk.get('consecutive_losses', 0)} "
            f"dead_bots={len(state.get('dead_bots', []))}"
        )
