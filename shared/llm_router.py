"""
LLM Router — cost-optimized AI dispatch.

Features:
  • Haiku by default; Sonnet only for final trade/risk decisions
  • Per-instance in-memory response cache (TTL 5 min, keyed by prompt hash)
  • System-wide Sonnet daily counter stored in Redis (max 20/bot/day → fallback Haiku)
  • Static prompt compression helpers — never send raw price history, indent=0 JSON

Usage:
    router = LLMRouter(client, save_fn=self.save_state, get_fn=self.bus.get_state)
    text   = await router.call(messages, prefer="sonnet", max_tokens=300, cache_key=k)
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import anthropic

from config import Models


CACHE_TTL          = 300   # seconds — same prompt reused for 5 min
SONNET_DAILY_LIMIT = 20    # max Sonnet calls per bot per day (hard ceiling)


class LLMRouter:
    """
    Thin dispatch layer — route, cache, count.
    One instance per bot; share nothing across processes (counter backed by Redis).
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        save_fn: Optional[Callable] = None,   # async(key, data, ttl=N)
        get_fn:  Optional[Callable] = None,   # async(key) → dict|None
    ):
        self._client = client
        self._save   = save_fn
        self._get    = get_fn
        self._cache: dict[str, tuple[str, float]] = {}   # key → (text, expiry_mono)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def call(
        self,
        messages: list[dict],
        *,
        prefer:     str = "haiku",   # "haiku" | "sonnet"
        max_tokens: int = 300,
        cache_key:  str = "",
    ) -> str:
        """
        Execute an LLM call with caching and model routing.
        Never raises — returns "" on error.
        """
        # Cache hit
        if cache_key:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return hit

        model = await self._select_model(prefer)

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

        if cache_key:
            self._cache_set(cache_key, text)
        return text

    # ── Model selection ────────────────────────────────────────────────────────

    async def _select_model(self, prefer: str) -> str:
        if prefer != "sonnet":
            return Models.HAIKU
        count = await self._sonnet_count()
        if count >= SONNET_DAILY_LIMIT:
            return Models.HAIKU          # limit hit — degrade silently
        await self._increment_sonnet(count)
        return Models.SONNET

    async def _sonnet_count(self) -> int:
        if not self._get:
            return 0
        try:
            data = await self._get("llm_sonnet_count")
            if data and data.get("date") == _today():
                return int(data.get("count", 0))
        except Exception:
            pass
        return 0

    async def _increment_sonnet(self, current: int):
        if not self._save:
            return
        try:
            await self._save("llm_sonnet_count", {
                "date":  _today(),
                "count": current + 1,
            }, ttl=86400)
        except Exception:
            pass

    # ── In-memory cache ────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry and time.monotonic() < entry[1]:
            return entry[0]
        self._cache.pop(key, None)
        return None

    def _cache_set(self, key: str, text: str):
        self._cache[key] = (text, time.monotonic() + CACHE_TTL)
        if len(self._cache) > 200:
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}

    # ── Static prompt helpers ──────────────────────────────────────────────────

    @staticmethod
    def cache_key(*parts) -> str:
        """Stable MD5 hash of args — use as cache_key for repeated prompts."""
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
    def compress_signals(signals: dict) -> dict:
        """
        Compress multi-signal dict to bare summary stats (~60-80 tokens as JSON).
        Strips: full text, descriptions, raw price series, all None fields.
        Keeps: symbol, direction, score, grade, rsi, pct_b, vol_ratio, sentiment counts.
        """
        # Top 5 momentum — abbreviated keys
        mom = [
            {
                "s":  s.get("symbol"),
                "d":  (s.get("direction") or "?")[0],   # l/b/n
                "sc": round(float(s.get("score", 0.5)), 2),
                "g":  s.get("grade", "?"),
                "r":  s.get("rsi_14"),
            }
            for s in signals.get("momentum", [])[:5]
            if s.get("symbol")
        ]
        # News → sentiment counts per symbol (no text)
        news: dict[str, str] = {}
        for sym, items in list(signals.get("news", {}).items())[:6]:
            b = sum(1 for i in items if i.get("sentiment") == "bullish")
            n = sum(1 for i in items if i.get("sentiment") == "bearish")
            if b or n:
                cat = (items[0].get("catalyst") or "")[:12] if items else ""
                news[sym] = f"{b}b{n}n {cat}".strip()
        # Options top 3
        opts = [
            {"s": o.get("sym"), "b": (o.get("bias") or "?")[0],
             "c": round(float(o.get("conf") or 0), 2)}
            for o in signals.get("options", [])[:3]
            if o.get("sym")
        ]
        # Price context: only rsi, pct_b, vol_ratio
        px = {
            sym: {"r": ctx.get("rsi_14"), "b": ctx.get("pct_b"), "v": ctx.get("vol_ratio")}
            for sym, ctx in list(signals.get("price_context", {}).items())[:5]
        }
        return {"m": mom, "nws": news, "opts": opts, "px": px}

    @staticmethod
    def compress_system_state(state: dict) -> str:
        """
        Compress system state to a single descriptive line (~50 tokens).
        Used in Master Commander prompt instead of the full JSON blob.
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


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
