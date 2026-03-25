"""
Bot 4 – Data Agent
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : The market data backbone.
         • Polls Alpaca every DATA_REFRESH_SECONDS for quotes & OHLCV bars
         • Computes basic technical indicators (RSI, MACD, BBands, ATR, VWAP)
         • Publishes enriched market_data packets to the Redis bus
         • All downstream bots consume from here – nothing talks to Alpaca
           for price data except this bot
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

from alpaca.data.timeframe import TimeFrame

from config import Models, RedisConfig, UniverseConfig, AnthropicConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


# ── Technical indicator helpers ────────────────────────────────────────────────

def calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    s = pd.Series(closes)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def calc_macd(closes: list[float],
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> dict[str, Optional[float]]:
    if len(closes) < slow + signal:
        return {"macd": None, "signal": None, "histogram": None}
    s        = pd.Series(closes)
    ema_fast = s.ewm(span=fast,   adjust=False).mean()
    ema_slow = s.ewm(span=slow,   adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    hist     = macd - sig
    return {
        "macd":      round(float(macd.iloc[-1]), 4),
        "signal":    round(float(sig.iloc[-1]),  4),
        "histogram": round(float(hist.iloc[-1]), 4),
    }


def calc_bollinger(closes: list[float], period: int = 20, std_dev: float = 2.0
                   ) -> dict[str, Optional[float]]:
    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None, "pct_b": None}
    s      = pd.Series(closes)
    mid    = s.rolling(period).mean()
    std    = s.rolling(period).std()
    upper  = mid + std_dev * std
    lower  = mid - std_dev * std
    last_close = closes[-1]
    band_width = upper.iloc[-1] - lower.iloc[-1]
    pct_b = (last_close - lower.iloc[-1]) / band_width if band_width != 0 else 0.5
    return {
        "upper":  round(float(upper.iloc[-1]),  2),
        "middle": round(float(mid.iloc[-1]),    2),
        "lower":  round(float(lower.iloc[-1]),  2),
        "pct_b":  round(float(pct_b),           4),
    }


def calc_atr(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return round(float(atr.iloc[-1]), 4)


def calc_vwap(bars: list[dict]) -> Optional[float]:
    if not bars:
        return None
    total_pv = sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"] for b in bars)
    total_v  = sum(b["volume"] for b in bars)
    return round(total_pv / total_v, 2) if total_v else None


def calc_sma(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return round(float(np.mean(closes[-period:])), 4)


def calc_ema(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    s = pd.Series(closes)
    return round(float(s.ewm(span=period, adjust=False).mean().iloc[-1]), 4)


# ── Data Agent ─────────────────────────────────────────────────────────────────

class DataAgent(BaseBot):
    """
    Bot 4 – Data Agent
    Runs on Haiku 4.5 for cheap, fast AI-assisted anomaly flagging.
    """

    BOT_ID = 4
    NAME   = "Data Agent"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.alpaca  = AlpacaClient()
        self.client  = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.symbols = list(dict.fromkeys(UniverseConfig.WATCHLIST))   # deduplicated
        self._bar_cache: dict[str, list[dict]] = {}   # rolling bar history

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log(f"Data Agent starting | watching {len(self.symbols)} symbols")
        await self._warm_cache()

    async def run(self):
        while self.running:
            try:
                await self._refresh_cycle()
            except Exception as e:
                self.log(f"Refresh error: {e}", "error")
            await asyncio.sleep(UniverseConfig.DATA_REFRESH_SECONDS)

    async def cleanup(self):
        self.log("Data Agent stopped")

    # ── Cache warm-up ──────────────────────────────────────────────────────────

    async def _warm_cache(self):
        """Load 60 days of daily bars so indicators have enough history."""
        self.log("Warming bar cache…")
        start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        try:
            bars = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.alpaca.get_bars(self.symbols, TimeFrame.Day, start),
            )
            for sym, bar_list in bars.items():
                self._bar_cache[sym] = bar_list
            self.log(f"Cache warm | {len(self._bar_cache)} symbols loaded")
        except Exception as e:
            self.log(f"Cache warm failed: {e}", "warning")

    # ── Refresh cycle ──────────────────────────────────────────────────────────

    async def _refresh_cycle(self):
        loop = asyncio.get_event_loop()

        # 1. Latest quotes (real-time bid/ask)
        quotes = await loop.run_in_executor(
            None, lambda: self.alpaca.get_latest_quotes(self.symbols)
        )

        # 2. Today's 1-min bars for intraday indicators
        today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        intraday = await loop.run_in_executor(
            None,
            lambda: self.alpaca.get_bars(self.symbols, TimeFrame.Minute, today_start),
        )

        # 3. Build enriched packets per symbol
        packets = []
        for sym in self.symbols:
            try:
                packet = self._build_packet(sym, quotes.get(sym, {}), intraday.get(sym, []))
                packets.append(packet)
            except Exception as e:
                self.log(f"Packet build failed for {sym}: {e}", "warning")

        # 4. AI anomaly scan (Haiku – cheap)
        anomalies = await self._ai_anomaly_scan(packets)

        # 5. Publish to bus
        for packet in packets:
            sym = packet["symbol"]
            if sym in anomalies:
                packet["ai_flag"] = anomalies[sym]
            await self.publish(RedisConfig.CHANNEL_MARKET_DATA, packet)

        # 6. Publish summary state for Master Commander
        summary = {
            "type":      "market_data_summary",
            "symbol_count": len(packets),
            "timestamp": datetime.utcnow().isoformat(),
            "flagged":   list(anomalies.keys()),
        }
        await self.save_state("latest_summary", summary, ttl=120)
        self.log(f"Published {len(packets)} packets | {len(anomalies)} flagged")

    # ── Packet builder ─────────────────────────────────────────────────────────

    def _build_packet(self, symbol: str, quote: dict, intraday_bars: list[dict]) -> dict:
        # Merge intraday bars with daily cache for indicator calc
        daily_bars  = self._bar_cache.get(symbol, [])
        all_bars    = daily_bars + intraday_bars          # daily + today's intraday

        closes  = [b["close"]  for b in all_bars]
        highs   = [b["high"]   for b in all_bars]
        lows    = [b["low"]    for b in all_bars]
        volumes = [b["volume"] for b in all_bars]

        # Update rolling intraday cache
        if intraday_bars:
            self._bar_cache[symbol] = (daily_bars + intraday_bars)[-200:]

        current_price = quote.get("ask", closes[-1] if closes else 0)

        indicators = {
            "rsi_14":    calc_rsi(closes, 14),
            "rsi_7":     calc_rsi(closes, 7),
            "macd":      calc_macd(closes),
            "bollinger": calc_bollinger(closes),
            "atr_14":    calc_atr(highs, lows, closes, 14),
            "vwap":      calc_vwap(intraday_bars),
            "sma_20":    calc_sma(closes, 20),
            "sma_50":    calc_sma(closes, 50),
            "sma_200":   calc_sma(closes, 200),
            "ema_9":     calc_ema(closes, 9),
            "ema_21":    calc_ema(closes, 21),
            "volume_avg_20": round(float(np.mean(volumes[-20:])), 0) if len(volumes) >= 20 else None,
            "volume_ratio":  round(volumes[-1] / np.mean(volumes[-20:]), 2)
                             if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else None,
        }

        return {
            "type":          "market_data",
            "symbol":        symbol,
            "price":         current_price,
            "bid":           quote.get("bid"),
            "ask":           quote.get("ask"),
            "spread":        round(quote.get("ask", 0) - quote.get("bid", 0), 4)
                             if quote.get("ask") and quote.get("bid") else None,
            "indicators":    indicators,
            "bar_count":     len(all_bars),
            "timestamp":     datetime.utcnow().isoformat(),
        }

    # ── AI anomaly scan ────────────────────────────────────────────────────────

    async def _ai_anomaly_scan(self, packets: list[dict]) -> dict[str, dict]:
        """
        Use Haiku to flag unusual conditions across all symbols in one call.
        Returns {symbol: {flag, reason, severity}} for flagged symbols only.
        """
        if not packets:
            return {}

        # Compact snapshot for the prompt
        snapshot = []
        for p in packets:
            ind = p.get("indicators", {})
            macd = ind.get("macd", {}) or {}
            bb   = ind.get("bollinger", {}) or {}
            snapshot.append({
                "sym":         p["symbol"],
                "price":       p["price"],
                "rsi":         ind.get("rsi_14"),
                "macd_hist":   macd.get("histogram"),
                "pct_b":       bb.get("pct_b"),
                "vol_ratio":   ind.get("volume_ratio"),
                "atr":         ind.get("atr_14"),
                "spread":      p.get("spread"),
            })

        prompt = (
            "You are a market data anomaly detector. Scan the following symbol snapshots "
            "and identify any with unusual conditions: extreme RSI (>80 or <20), "
            "MACD histogram spikes, Bollinger band breakouts (pct_b >1.0 or <0.0), "
            "volume ratio >3x or <0.1x, or abnormally wide spreads.\n\n"
            f"Snapshots: {json.dumps(snapshot)}\n\n"
            "Respond ONLY with a JSON object: "
            "{\"flagged\": [{\"symbol\": \"AAPL\", \"flag\": \"RSI_OVERBOUGHT\", "
            "\"reason\": \"RSI 82\", \"severity\": \"high\"}]}"
            " Return {\"flagged\": []} if nothing is unusual."
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
            # Extract JSON even if model adds markdown fences
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed  = json.loads(raw)
            flagged = parsed.get("flagged", [])
            return {item["symbol"]: item for item in flagged}
        except Exception as e:
            self.log(f"AI anomaly scan error: {e}", "warning")
            return {}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    agent = DataAgent()
    asyncio.run(agent.start())
