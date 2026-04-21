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
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

from alpaca.data.timeframe import TimeFrame

import pytz
from config import Models, RedisConfig, UniverseConfig, AnthropicConfig, RiskConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient
from shared.llm_router import LLMRouter

MARKET_TZ = pytz.timezone("America/New_York")


# ── JSON cleaning helper ───────────────────────────────────────────────────────

def _extract_json_object(text: str) -> str:
    """
    Walk the string character-by-character to extract the outermost {...} block,
    correctly handling nested braces and quoted strings.
    Falls back to the original text if no object is found.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string:
            i += 2          # skip escaped character
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return text[start:]     # unterminated — return what we have


def _clean_llm_json(raw: str) -> str:
    """
    Progressively clean common LLM JSON malformations so json.loads can parse.
    Order matters: strip fences → extract object → remove comments →
    fix Python literals → fix trailing commas.
    """
    # 1. Strip markdown code fences
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE).strip()
    # 2. Extract the outermost {...} using bracket-tracking
    raw = _extract_json_object(raw)
    # 3. Remove // single-line comments
    raw = re.sub(r"//[^\n\"]*(?=\"|\n|$)", "", raw)
    # 4. Remove /* block comments */
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    # 5. Replace Python/JS literals
    raw = re.sub(r"\bNone\b",  "null",  raw)
    raw = re.sub(r"\bTrue\b",  "true",  raw)
    raw = re.sub(r"\bFalse\b", "false", raw)
    # 6. Remove trailing commas before ] or } (run twice for nested cases)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return raw.strip()


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
        self._router = LLMRouter(self.client)
        self.symbols = list(dict.fromkeys(UniverseConfig.WATCHLIST))   # deduplicated
        self._bar_cache: dict[str, list[dict]] = {}   # rolling bar history
        self._last_ai_scan: datetime | None = None   # throttle AI scan to 10-min intervals
        self._morning_scan_done: bool = False   # runs once per day at 9am
        self._premarket_scan_done: bool = False  # runs once per day at 8am

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log(f"Data Agent starting | watching {len(self.symbols)} symbols")
        asyncio.create_task(self._premarket_scan_task())
        asyncio.create_task(self._morning_scan_task())
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

    # ── Connection retry helper ────────────────────────────────────────────────

    async def _retry(self, fn, label: str, retries: int = 3, delay: float = 5.0):
        """Run a sync callable in executor; retry up to `retries` times on connection errors."""
        loop = asyncio.get_event_loop()
        for attempt in range(1, retries + 1):
            try:
                return await loop.run_in_executor(None, fn)
            except (ConnectionResetError, TimeoutError, OSError) as e:
                if attempt < retries:
                    self.log(f"{label} connection error (attempt {attempt}/{retries}): {e} — retrying in {delay}s", "warning")
                    await asyncio.sleep(delay)
                else:
                    self.log(f"{label} failed after {retries} attempts: {e}", "warning")
                    raise
            except Exception:
                raise   # non-connection errors bubble up immediately

    # ── 8am Pre-market scan (runs ONCE per day) ───────────────────────────────

    async def _premarket_scan_task(self):
        """Sleep until 8:00am EST, score all universe symbols 1-100, publish watchlist."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=8, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            while target.weekday() >= 5:
                target += timedelta(days=1)
            self.log(f"Premarket scan scheduled in {(target - now_et).total_seconds()/60:.1f} min")
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self._premarket_scan_done = False
            await self._run_premarket_scan()
            self._premarket_scan_done = True

    def _score_symbol(self, sym: str) -> int:
        """
        Score a symbol 1-100 using cached daily bars.

        Points breakdown (max 100):
          25 – Recent price momentum (yesterday vs prior close)
          20 – Volume strength vs 20-day average
          15 – RSI setup (sweet spot 45-70)
          15 – MACD bullish histogram
          15 – Gap-up indicator
          10 – Proximity to recent 90-day high
        """
        bars = self._bar_cache.get(sym, [])
        if len(bars) < 5:
            return 0

        closes  = [b["close"]  for b in bars]
        highs   = [b["high"]   for b in bars]
        volumes = [b["volume"] for b in bars]

        score = 0

        # 1. Price momentum (25 pts)
        if len(closes) >= 2 and closes[-2] > 0:
            pct = (closes[-1] - closes[-2]) / closes[-2] * 100
            if pct > 5:    score += 25
            elif pct > 2:  score += 15
            elif pct > 0:  score += 5
            elif pct < -2: score -= 10

        # 2. Volume strength (20 pts)
        if len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            if avg_vol > 0:
                vr = volumes[-1] / avg_vol
                if vr > 3:    score += 20
                elif vr > 2:  score += 15
                elif vr > 1.5: score += 10

        # 3. RSI setup (15 pts)
        rsi = calc_rsi(closes, 14)
        if rsi is not None:
            if 45 <= rsi <= 70:   score += 15
            elif 35 <= rsi < 45:  score += 8   # oversold recovery
            elif rsi > 80:        score -= 10  # overbought

        # 4. MACD bullish (15 pts)
        macd = calc_macd(closes)
        if macd.get("histogram") is not None:
            if macd["histogram"] > 0:
                score += 15
            elif (macd.get("macd") is not None and macd.get("signal") is not None
                  and macd["macd"] > macd["signal"]):
                score += 7   # crossing up

        # 5. Gap up (15 pts)
        if len(bars) >= 2 and bars[-2]["close"] > 0:
            gap = (bars[-1]["open"] - bars[-2]["close"]) / bars[-2]["close"] * 100
            if gap > 3:    score += 15
            elif gap > 2:  score += 10
            elif gap > 1:  score += 5

        # 6. Near recent high (10 pts)
        if len(highs) >= 10:
            recent_high = max(highs[-min(63, len(highs)):])  # ~3-month high
            if recent_high > 0 and closes[-1] >= recent_high * 0.98:
                score += 10
            elif recent_high > 0 and closes[-1] >= recent_high * 0.95:
                score += 5

        return max(0, min(score, 100))

    async def _run_premarket_scan(self):
        """
        Score every symbol in SCAN_UNIVERSE + BEAR_ETFS.
        Save scored watchlist to Redis for bot05 and bot08 to consume.
        Publish top picks to market_data channel.
        """
        all_syms = list(dict.fromkeys(
            UniverseConfig.SCAN_UNIVERSE + UniverseConfig.BEAR_ETFS
        ))

        # Refresh bar cache for accurate scores
        start = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            from alpaca.data.timeframe import TimeFrame
            bars = await self._retry(
                lambda: self.alpaca.get_bars(all_syms, TimeFrame.Day, start),
                "premarket_scan_bars",
            )
            for sym, bar_list in bars.items():
                if bar_list:
                    existing = self._bar_cache.get(sym, [])
                    self._bar_cache[sym] = (existing + bar_list)[-200:]
        except Exception as e:
            self.log(f"Premarket scan bar refresh failed (using cached): {e}", "warning")

        # Score every symbol
        scored = []
        for sym in all_syms:
            s = self._score_symbol(sym)
            bars = self._bar_cache.get(sym, [])
            price = round(bars[-1]["close"], 2) if bars else 0
            if s > 0:
                scored.append({"symbol": sym, "score": s, "price": price})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_picks = [s for s in scored if s["score"] >= RiskConfig.MIN_STOCK_SCORE]

        await self.save_state("premarket_scan", {
            "scored":    scored[:40],
            "top_picks": top_picks,
            "threshold": RiskConfig.MIN_STOCK_SCORE,
            "timestamp": datetime.utcnow().isoformat(),
        }, ttl=3600 * 8)

        self.log(
            f"Premarket scan: {len(scored)} symbols scored | "
            f"{len(top_picks)} qualify (score ≥ {RiskConfig.MIN_STOCK_SCORE}) | "
            f"top3={[s['symbol'] for s in top_picks[:3]]}"
        )

        # Publish top picks so downstream bots react immediately
        for pick in top_picks[:15]:
            await self.publish(RedisConfig.CHANNEL_MARKET_DATA, {
                "type":      "premarket_pick",
                "symbol":    pick["symbol"],
                "score":     pick["score"],
                "price":     pick["price"],
                "timestamp": datetime.utcnow().isoformat(),
            })

    # ── 9am Morning scan (runs ONCE per day) ──────────────────────────────────

    async def _morning_scan_task(self):
        """Sleep until 9:00am EST, run one market-wide scan, then wait until next day."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            self.log(f"Morning scan scheduled in {(target - now_et).total_seconds()/60:.1f} min")
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self._morning_scan_done = False
            await self._run_morning_scan()
            self._morning_scan_done = True

    async def _run_morning_scan(self):
        """
        One-shot 9am scan using Alpaca FREE data only:
          1. Fetch 2-day bars for the full SCAN_UNIVERSE
          2. Rank by volume, % change, and gap
          3. Claude Haiku picks TOP 20 best opportunities
          4. Save results to Redis for all other bots to read
        """
        scan_symbols = list(dict.fromkeys(UniverseConfig.SCAN_UNIVERSE))
        self.log(f"Morning scan starting | {len(scan_symbols)} symbols in universe")

        start = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            bars = await self._retry(
                lambda: self.alpaca.get_bars(scan_symbols, TimeFrame.Day, start),
                "morning_scan_bars",
            )
        except Exception as e:
            self.log(f"Morning scan bars failed: {e}", "warning")
            return

        # Compute per-symbol metrics (no AI cost here — pure math)
        candidates = []
        for sym, bar_list in bars.items():
            if len(bar_list) < 2:
                continue
            prev = bar_list[-2]
            curr = bar_list[-1]
            if prev["close"] <= 0:
                continue
            pct_change = (curr["close"] - prev["close"]) / prev["close"] * 100
            gap_pct    = (curr["open"]  - prev["close"]) / prev["close"] * 100
            vol_ratio  = curr["volume"] / max(prev["volume"], 1)
            candidates.append({
                "symbol":     sym,
                "price":      round(curr["close"], 2),
                "pct_change": round(pct_change, 2),
                "gap_pct":    round(gap_pct, 2),
                "volume":     int(curr["volume"]),
                "vol_ratio":  round(vol_ratio, 2),
            })

        # Sort by combined signal strength: |% change| + vol_ratio + |gap|
        candidates.sort(
            key=lambda x: abs(x["pct_change"]) + x["vol_ratio"] + abs(x["gap_pct"]),
            reverse=True,
        )
        top_raw = candidates[:30]   # send top 30 to AI to choose best 20

        # Flag gap stocks (≥2% gap up or down)
        gap_ups   = [c["symbol"] for c in candidates if c["gap_pct"] >= 2.0]
        gap_downs = [c["symbol"] for c in candidates if c["gap_pct"] <= -2.0]
        self.log(
            f"Morning scan raw: {len(candidates)} symbols | "
            f"gap up {len(gap_ups)} | gap down {len(gap_downs)}"
        )

        # Claude AI picks top 20 opportunities (ONE call, Haiku — cheap)
        opportunities = await self._ai_morning_analysis(top_raw, gap_ups, gap_downs)

        # Save to Redis — all bots read this for the day
        await self.save_state("morning_scan", {
            "opportunities": opportunities,
            "gap_ups":        gap_ups[:10],
            "gap_downs":      gap_downs[:10],
            "top_by_volume":  sorted(candidates, key=lambda x: x["volume"], reverse=True)[:10],
            "scanned":        len(candidates),
            "timestamp":      datetime.utcnow().isoformat(),
        }, ttl=3600 * 8)   # keep for 8 hours

        # Publish each top pick as a flagged market_data event
        for opp in opportunities:
            await self.publish(RedisConfig.CHANNEL_MARKET_DATA, {
                "type":    "morning_pick",
                "symbol":  opp.get("symbol"),
                "direction": opp.get("direction", "long"),
                "reason":  opp.get("reason", ""),
                "priority": opp.get("priority", "medium"),
                "timestamp": datetime.utcnow().isoformat(),
            })

        self.log(f"Morning scan complete | {len(opportunities)} AI picks published")

    async def _ai_morning_analysis(self, candidates: list[dict],
                                   gap_ups: list[str], gap_downs: list[str]) -> list[dict]:
        """ONE Haiku call — picks top 20 opportunities. Compressed + cached."""
        # Minify candidate data: key abbreviations, no indent
        mini = [{"s": c["symbol"], "pct": c["pct_change"],
                 "vr": c["vol_ratio"], "gap": c["gap_pct"]}
                for c in candidates[:25]]
        ck = LLMRouter.cache_key(mini, gap_ups[:10], gap_downs[:10])
        prompt = (
            f"Top 20 trades from 9am scan. Long or short. JSON only.\n"
            f"Cands:{LLMRouter.j(mini)}\n"
            f"GapUp:{gap_ups[:8]} GapDn:{gap_downs[:8]}\n"
            "JSON:{\"opportunities\":[{\"symbol\":\"\",\"direction\":\"long\","
            "\"reason\":\"\",\"priority\":\"high\"}]}"
        )
        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="haiku", max_tokens=800, cache_key=ck,
        )
        try:
            parsed = json.loads(_clean_llm_json(raw))
            return parsed.get("opportunities", [])[:20]
        except Exception as e:
            self.log(f"Morning AI analysis error: {e}", "warning")
            return [{"symbol": c["symbol"], "direction": "long",
                     "reason": f"pct={c['pct_change']:+.1f}% vr={c['vol_ratio']:.1f}x",
                     "priority": "medium"} for c in candidates[:20]]

    # ── Cache warm-up ──────────────────────────────────────────────────────────

    async def _warm_cache(self):
        """Load 60 days of daily bars so indicators have enough history."""
        self.log("Warming bar cache…")
        start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        try:
            bars = await self._retry(
                lambda: self.alpaca.get_bars(self.symbols, TimeFrame.Day, start),
                "get_bars(warm_cache)",
            )
            for sym, bar_list in bars.items():
                self._bar_cache[sym] = bar_list
            self.log(f"Cache warm | {len(self._bar_cache)} symbols loaded")
        except Exception as e:
            self.log(f"Cache warm failed: {e}", "warning")

    # ── Refresh cycle ──────────────────────────────────────────────────────────

    async def _refresh_cycle(self):
        # 1. Latest quotes (real-time bid/ask)
        today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        try:
            quotes = await self._retry(
                lambda: self.alpaca.get_latest_quotes(self.symbols),
                "get_latest_quotes",
            )
        except Exception as e:
            self.log(f"Quotes fetch failed, skipping cycle: {e}", "warning")
            return

        # 2. Today's 1-min bars for intraday indicators
        try:
            intraday = await self._retry(
                lambda: self.alpaca.get_bars(self.symbols, TimeFrame.Minute, today_start),
                "get_bars(intraday)",
            )
        except Exception as e:
            self.log(f"Intraday bars fetch failed, skipping cycle: {e}", "warning")
            return

        # 3. Build enriched packets per symbol
        packets = []
        for sym in self.symbols:
            try:
                packet = self._build_packet(sym, quotes.get(sym, {}), intraday.get(sym, []))
                packets.append(packet)
            except Exception as e:
                self.log(f"Packet build failed for {sym}: {e}", "warning")

        # 4. AI anomaly scan — throttled to once every 10 minutes
        now = datetime.utcnow()
        if self._last_ai_scan is None or (now - self._last_ai_scan).total_seconds() >= 600:
            anomalies = await self._ai_anomaly_scan(packets)
            self._last_ai_scan = now
        else:
            anomalies = {}

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
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            try:
                parsed = json.loads(_clean_llm_json(raw))
            except json.JSONDecodeError:
                return {}
            flagged = parsed.get("flagged", [])
            return {item["symbol"]: item for item in flagged if "symbol" in item}
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
