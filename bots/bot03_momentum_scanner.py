"""
Bot 3 – Momentum Scanner
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : Listens to market data from Bot 4 and runs a multi-timeframe
         momentum scoring system every scan cycle. Identifies:
           • Trend direction (SMA alignment)
           • Momentum strength (RSI + MACD)
           • Breakout candidates (price vs Bollinger Bands)
           • Volume confirmation
         Publishes scored momentum signals to CHANNEL_MOMENTUM.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import pytz

from config import Models, RedisConfig, UniverseConfig, AnthropicConfig
from shared.base_bot import BaseBot
from shared.llm_router import LLMRouter

MARKET_TZ = pytz.timezone("America/New_York")


SCAN_INTERVAL  = UniverseConfig.SCAN_INTERVAL_SECONDS
MIN_SCORE      = 0.60    # minimum momentum score to publish a signal


class MomentumScanner(BaseBot):
    """
    Bot 3 – Momentum Scanner
    Consumes CHANNEL_MARKET_DATA, scores momentum, emits signals.
    """

    BOT_ID = 3
    NAME   = "Momentum Scanner"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.client  = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self._router = LLMRouter(self.client)
        # Latest data packet per symbol (written by _on_market_data)
        self._latest: dict[str, dict] = {}
        # Daily watchlist — set at 9am, watched all day, updated next morning
        self._daily_watchlist: set[str] = set()
        self._watchlist_set: bool = False   # True once 9am task has run today

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA, self._on_market_data)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._morning_watchlist_task())
        self.log("Momentum Scanner starting")

    async def run(self):
        # Run scan on interval
        while self.running:
            await asyncio.sleep(SCAN_INTERVAL)
            try:
                await self._scan_cycle()
            except Exception as e:
                self.log(f"Scan cycle error: {e}", "error")

    async def cleanup(self):
        self.log("Momentum Scanner stopped")

    # ── Morning watchlist (9am, once per day) ─────────────────────────────────

    async def _morning_watchlist_task(self):
        """At 9am, read bot04's morning scan and lock in the top 20 watchlist for the day."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self._watchlist_set = False
            await self._build_daily_watchlist()

    async def _build_daily_watchlist(self):
        """
        Read the morning scan results from bot04 (or fall back to scoring
        all available market data). Set self._daily_watchlist to top 20 symbols.
        """
        # Try to get bot04's morning scan picks first
        try:
            scan = await self.bus.get_state("bot4:morning_scan") or {}
            opps = scan.get("opportunities", [])
            if opps:
                top_syms = [o["symbol"] for o in opps[:20] if o.get("symbol")]
                self._daily_watchlist = set(top_syms)
                self._watchlist_set   = True
                self.log(
                    f"Daily watchlist set from morning scan: {sorted(self._daily_watchlist)}"
                )
                return
        except Exception as e:
            self.log(f"Morning scan read error: {e}", "warning")

        # Fallback: score whatever market data we already have, take top 20
        if self._latest:
            scored = []
            for sym, data in self._latest.items():
                result = self._score_symbol(data)
                scored.append((sym, result["score"]))
            scored.sort(key=lambda x: x[1], reverse=True)
            self._daily_watchlist = {s[0] for s in scored[:20]}
            self._watchlist_set   = True
            self.log(
                f"Daily watchlist set from live scores (fallback): "
                f"{sorted(self._daily_watchlist)}"
            )
        else:
            # No data yet — use full watchlist as default
            self._daily_watchlist = set(UniverseConfig.WATCHLIST)
            self.log("Daily watchlist defaulted to WATCHLIST (no scan data yet)")

    # ── Market data ingestion ──────────────────────────────────────────────────

    async def _on_market_data(self, data: dict):
        sym = data.get("symbol")
        if sym:
            self._latest[sym] = data

    # ── Momentum scoring ───────────────────────────────────────────────────────

    def _score_symbol(self, data: dict) -> dict:
        """
        Rule-based momentum score (0-1) with component breakdown.
        Score is the arithmetic mean of component scores.
        """
        ind   = data.get("indicators", {})
        price = data.get("price", 0)

        components = {}

        # 1. Trend: SMA alignment (200 > 50 > 20 = bullish)
        sma20  = ind.get("sma_20")
        sma50  = ind.get("sma_50")
        sma200 = ind.get("sma_200")
        if sma20 and sma50 and sma200:
            if price > sma20 > sma50 > sma200:
                components["trend"] = 1.0
                trend_dir = "bullish"
            elif price < sma20 < sma50 < sma200:
                components["trend"] = 0.0
                trend_dir = "bearish"
            elif price > sma20 > sma50:
                components["trend"] = 0.75
                trend_dir = "bullish"
            elif price < sma20 < sma50:
                components["trend"] = 0.25
                trend_dir = "bearish"
            else:
                components["trend"] = 0.5
                trend_dir = "mixed"
        else:
            trend_dir = "unknown"

        # 2. RSI momentum
        rsi14 = ind.get("rsi_14")
        rsi7  = ind.get("rsi_7")
        if rsi14 is not None:
            if 50 <= rsi14 <= 70:
                components["rsi"] = 0.8    # bullish momentum, not yet overbought
            elif rsi14 > 70:
                components["rsi"] = 0.6    # overbought – caution
            elif 30 <= rsi14 < 50:
                components["rsi"] = 0.3    # bearish momentum
            else:
                components["rsi"] = 0.2    # oversold (potential reversal)
            # RSI acceleration: 7-period trending up
            if rsi7 and rsi14:
                if rsi7 > rsi14:
                    components["rsi"] = min(1.0, components["rsi"] + 0.1)
                elif rsi7 < rsi14:
                    components["rsi"] = max(0.0, components["rsi"] - 0.1)

        # 3. MACD
        macd_data = ind.get("macd", {}) or {}
        macd_hist  = macd_data.get("histogram")
        macd_line  = macd_data.get("macd")
        signal_line = macd_data.get("signal")
        if macd_hist is not None and macd_line is not None:
            if macd_hist > 0 and macd_line > signal_line:
                components["macd"] = 0.85
            elif macd_hist > 0:
                components["macd"] = 0.65
            elif macd_hist < 0 and macd_line < signal_line:
                components["macd"] = 0.15
            else:
                components["macd"] = 0.35

        # 4. Bollinger Band position
        bb   = ind.get("bollinger", {}) or {}
        pctb = bb.get("pct_b")
        if pctb is not None:
            if 0.5 <= pctb <= 0.8:
                components["bollinger"] = 0.8    # upper half, not yet breakout
            elif pctb > 1.0:
                components["bollinger"] = 0.5    # breakout (possible continuation or reversal)
            elif 0.2 <= pctb < 0.5:
                components["bollinger"] = 0.3
            else:
                components["bollinger"] = 0.1    # near/below lower band

        # 5. Volume confirmation
        vol_ratio = ind.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio >= 2.0:
                components["volume"] = 1.0
            elif vol_ratio >= 1.5:
                components["volume"] = 0.8
            elif vol_ratio >= 1.0:
                components["volume"] = 0.6
            elif vol_ratio >= 0.7:
                components["volume"] = 0.4
            else:
                components["volume"] = 0.2

        # 6. Price vs VWAP
        vwap = ind.get("vwap")
        if vwap and price:
            if price > vwap * 1.005:
                components["vwap"] = 0.75
            elif price > vwap:
                components["vwap"] = 0.6
            elif price < vwap * 0.995:
                components["vwap"] = 0.25
            else:
                components["vwap"] = 0.4

        if not components:
            return {"score": 0.5, "components": {}, "direction": "unknown", "grade": "D"}

        score = round(sum(components.values()) / len(components), 3)

        # Direction from components
        if score >= 0.65:
            direction = "bullish"
        elif score <= 0.35:
            direction = "bearish"
        else:
            direction = "neutral"

        # Grade
        if score >= 0.80:   grade = "A"
        elif score >= 0.65: grade = "B"
        elif score >= 0.50: grade = "C"
        elif score >= 0.35: grade = "D"
        else:               grade = "F"

        return {
            "score":      score,
            "components": components,
            "direction":  direction,
            "grade":      grade,
            "trend_dir":  trend_dir,
        }

    # ── AI narrative ───────────────────────────────────────────────────────────

    async def _ai_narrative(self, top_signals: list[dict]) -> dict[str, str]:
        """One-sentence momentum narrative per symbol. Haiku, cached 5 min."""
        if not top_signals:
            return {}
        # Compress to key metrics only — no full signal objects
        items = [
            {"s": s["symbol"], "d": s["direction"][0], "g": s["grade"],
             "r": s.get("rsi_14"), "v": s.get("volume_ratio")}
            for s in top_signals
        ]
        ck     = LLMRouter.cache_key(items)
        prompt = (
            f"1-sentence momentum narrative per symbol.\n"
            f"{LLMRouter.j(items)}\n"
            "JSON:{\"narratives\":{\"SYM\":\"\"}}"
        )
        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="haiku", max_tokens=300, cache_key=ck,
        )
        try:
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw).get("narratives", {})
        except Exception as e:
            self.log(f"AI narrative error: {e}", "warning")
            return {}

    # ── Scan cycle ─────────────────────────────────────────────────────────────

    async def _scan_cycle(self):
        if not self._latest:
            self.log("No market data yet, skipping scan")
            return

        # Only score symbols on the daily watchlist (set at 9am)
        # Fall back to full universe if watchlist not yet populated
        scan_universe = (
            {sym: data for sym, data in self._latest.items()
             if sym in self._daily_watchlist}
            if self._daily_watchlist else self._latest
        )

        scored = []
        for sym, data in scan_universe.items():
            result = self._score_symbol(data)
            if result["score"] >= MIN_SCORE or result["score"] <= (1 - MIN_SCORE):
                ind = data.get("indicators", {})
                scored.append({
                    "symbol":       sym,
                    "price":        data.get("price"),
                    "score":        result["score"],
                    "grade":        result["grade"],
                    "direction":    result["direction"],
                    "trend_dir":    result.get("trend_dir", "unknown"),
                    "components":   result["components"],
                    "rsi_14":       ind.get("rsi_14"),
                    "pct_b":        (ind.get("bollinger") or {}).get("pct_b"),
                    "volume_ratio": ind.get("volume_ratio"),
                    "macd_hist":    (ind.get("macd") or {}).get("histogram"),
                    "ai_flag":      data.get("ai_flag"),
                })

        if not scored:
            self.log("No high-momentum signals this cycle")
            return

        # Sort: highest bullish first, then highest bearish
        bullish = sorted([s for s in scored if s["direction"] == "bullish"],
                         key=lambda x: x["score"], reverse=True)
        bearish = sorted([s for s in scored if s["direction"] == "bearish"],
                         key=lambda x: x["score"])   # lowest score = most bearish

        top_signals = bullish[:5] + bearish[:5]

        # Get AI narratives
        narratives = await self._ai_narrative(top_signals)

        # Publish each signal
        published = 0
        for sig in top_signals:
            sig["narrative"] = narratives.get(sig["symbol"], "")
            await self.publish(RedisConfig.CHANNEL_MOMENTUM, {
                "type": "momentum_signal",
                **sig,
            })
            published += 1

        # Leaderboard state for Master Commander
        leaderboard = {
            "type":      "momentum_leaderboard",
            "top_bullish": [{"sym": s["symbol"], "score": s["score"], "grade": s["grade"]}
                            for s in bullish[:3]],
            "top_bearish": [{"sym": s["symbol"], "score": s["score"], "grade": s["grade"]}
                            for s in bearish[:3]],
            "scanned":   len(self._latest),
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.save_state("leaderboard", leaderboard, ttl=120)
        self.log(
            f"Scan complete | watchlist={len(scan_universe)}/{len(self._latest)} symbols | "
            f"{len(bullish)} bullish | {len(bearish)} bearish | published {published}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = MomentumScanner()
    asyncio.run(bot.start())
