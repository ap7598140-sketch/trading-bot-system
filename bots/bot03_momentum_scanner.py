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
from datetime import datetime
from typing import Optional

import anthropic

from config import Models, RedisConfig, UniverseConfig, AnthropicConfig
from shared.base_bot import BaseBot


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
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        # Latest data packet per symbol (written by _on_market_data)
        self._latest: dict[str, dict] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA, self._on_market_data)
        asyncio.create_task(self.bus.listen())
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
        """Generate brief momentum narratives for top signals using Haiku."""
        if not top_signals:
            return {}

        items = [
            {
                "sym":   s["symbol"],
                "score": s["score"],
                "grade": s["grade"],
                "dir":   s["direction"],
                "rsi":   s.get("rsi_14"),
                "trend": s.get("trend_dir"),
                "bb":    s.get("pct_b"),
                "vol":   s.get("volume_ratio"),
            }
            for s in top_signals
        ]

        prompt = (
            "You are a momentum trader. Write a 1-sentence momentum narrative for each symbol.\n\n"
            f"Signals: {json.dumps(items)}\n\n"
            "Respond ONLY with JSON: "
            "{\"narratives\": {\"AAPL\": \"Strong uptrend with volume confirmation...\"}}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            return parsed.get("narratives", {})
        except Exception as e:
            self.log(f"AI narrative error: {e}", "warning")
            return {}

    # ── Scan cycle ─────────────────────────────────────────────────────────────

    async def _scan_cycle(self):
        if not self._latest:
            self.log("No market data yet, skipping scan")
            return

        scored = []
        for sym, data in self._latest.items():
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
            f"Scan complete | {len(self._latest)} symbols | "
            f"{len(bullish)} bullish | {len(bearish)} bearish | "
            f"published {published}"
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
