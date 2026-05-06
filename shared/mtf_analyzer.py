"""
Multi-Timeframe Analyzer — MTF trend, FVG, and ASP setup detection.

Evaluates 4 timeframes per symbol:
  Daily (20 days), 4H (48 bars), 1H (24 bars), 15M (96 bars)

Scoring (6 criteria):
  1. Daily trend bullish
  2. 4H trend bullish
  3. 1H trend bullish
  4. 15M trend bullish
  5. Active bullish FVG (price touching gap zone)
  6. ASP setup confirmed (Attack / Support / Protection)

Grades: A+(6), A(5), B(4), C(3), D(<3)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from shared.alpaca_client import AlpacaClient


# Timeframe definitions
_TF_DAY = TimeFrame.Day
_TF_4H  = TimeFrame(4,  TimeFrameUnit.Hour)
_TF_1H  = TimeFrame.Hour
_TF_15M = TimeFrame(15, TimeFrameUnit.Minute)

# Calendar days to look back per timeframe
_LOOKBACK_DAYS = {
    "day": 30,   # ~20 trading days
    "4h":  20,   # ~48 4H bars
    "1h":   7,   # ~24 1H bars
    "15m":  3,   # ~96 15M bars
}

_FETCH_TIMEOUT = 15.0   # seconds per timeframe fetch


class MultiTimeframeAnalyzer:
    """
    Fetch and score multi-timeframe data for a batch of symbols.
    All Alpaca calls run in an executor — no blocking of the event loop.
    """

    def __init__(self, alpaca: AlpacaClient):
        self._alpaca = alpaca

    # ── Public API ─────────────────────────────────────────────────────────────

    async def analyze_batch(
        self,
        symbols: list[str],
        current_prices: Optional[dict[str, float]] = None,
    ) -> dict[str, dict]:
        """
        Analyze all symbols concurrently. Returns:
          {symbol: {trend_daily, trend_4h, trend_1h, trend_15m,
                    fvg_active, asp_setup, criteria_count, grade}}
        Never raises — returns empty analysis on any per-symbol error.
        """
        if not symbols:
            return {}
        prices = current_prices or {}
        tasks  = [self._analyze_symbol(sym, prices.get(sym)) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            sym: (r if not isinstance(r, Exception) else self._empty_analysis(str(r)))
            for sym, r in zip(symbols, results)
        }

    # ── Per-symbol analysis ────────────────────────────────────────────────────

    async def _analyze_symbol(self, symbol: str, price: Optional[float]) -> dict:
        try:
            day_bars, bars_4h, bars_1h, bars_15m = await asyncio.gather(
                self._fetch_tf(symbol, _TF_DAY, _LOOKBACK_DAYS["day"]),
                self._fetch_tf(symbol, _TF_4H,  _LOOKBACK_DAYS["4h"]),
                self._fetch_tf(symbol, _TF_1H,  _LOOKBACK_DAYS["1h"]),
                self._fetch_tf(symbol, _TF_15M, _LOOKBACK_DAYS["15m"]),
            )
        except Exception as e:
            return self._empty_analysis(f"fetch_error:{e}")

        if not price and day_bars:
            price = day_bars[-1]["close"]

        trend_daily = self._trend(day_bars)
        trend_4h    = self._trend(bars_4h)
        trend_1h    = self._trend(bars_1h)
        trend_15m   = self._trend(bars_15m)

        fvgs_1h    = self._detect_fvgs(bars_1h)
        fvg_active = self._active_fvgs(fvgs_1h, price)
        asp_setup  = self._asp_setup(bars_1h, bars_15m, price)

        criteria = [
            trend_daily == "bullish",
            trend_4h    == "bullish",
            trend_1h    == "bullish",
            trend_15m   == "bullish",
            fvg_active,
            asp_setup,
        ]
        n = sum(criteria)

        return {
            "trend_daily":    trend_daily,
            "trend_4h":       trend_4h,
            "trend_1h":       trend_1h,
            "trend_15m":      trend_15m,
            "fvg_active":     fvg_active,
            "asp_setup":      asp_setup,
            "criteria_count": n,
            "grade":          self._grade_from_count(n),
        }

    # ── Data fetching ─────────────────────────────────────────────────────────

    async def _fetch_tf(
        self, symbol: str, tf: TimeFrame, lookback_days: int
    ) -> list[dict]:
        start = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        loop = asyncio.get_event_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._alpaca.get_bars([symbol], tf, start),
                ),
                timeout=_FETCH_TIMEOUT,
            )
            return raw.get(symbol, [])
        except Exception:
            return []

    # ── Trend detection ────────────────────────────────────────────────────────

    @staticmethod
    def _trend(bars: list[dict]) -> str:
        """
        SMA20 position + 5-bar momentum → bullish / neutral / bearish.
        Returns neutral when fewer than 10 bars are available.
        """
        if len(bars) < 10:
            return "neutral"

        closes     = [b["close"] for b in bars]
        sma_window = min(20, len(closes))
        sma20      = sum(closes[-sma_window:]) / sma_window
        above_sma  = closes[-1] > sma20

        recent       = sum(closes[-5:]) / 5
        prior        = sum(closes[-10:-5]) / 5
        momentum_up  = recent > prior

        if above_sma and momentum_up:
            return "bullish"
        if not above_sma and not momentum_up:
            return "bearish"
        return "neutral"

    # ── FVG detection ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_fvgs(bars: list[dict]) -> list[dict]:
        """
        Bullish FVG: candle[i+2].low > candle[i].high (gap in the last 20 bars).
        Returns list of {low, high} zone dicts.
        """
        check = bars[-20:] if len(bars) > 20 else bars
        fvgs  = []
        for i in range(len(check) - 2):
            c0, c2 = check[i], check[i + 2]
            if c2["low"] > c0["high"]:
                fvgs.append({"low": c0["high"], "high": c2["low"]})
        return fvgs

    @staticmethod
    def _active_fvgs(fvgs: list[dict], price: Optional[float]) -> bool:
        """
        True if price is within 0.5% of any bullish FVG zone
        (price returning to the gap = potential support + continuation).
        """
        if not fvgs or not price:
            return False
        for fvg in fvgs:
            if fvg["low"] * 0.995 <= price <= fvg["high"] * 1.005:
                return True
        return False

    # ── ASP setup detection ───────────────────────────────────────────────────

    @staticmethod
    def _asp_setup(
        bars_1h: list[dict], bars_15m: list[dict], price: Optional[float]
    ) -> bool:
        """
        ASP (Attack / Support / Protection):
          Attack   — price broke above a recent 1H swing high
          Support  — level is now holding (brief 15M pullback, price stays above)
          Protection — stop zone defined implicitly below the swing high

        Detection:
          1. Swing high = max of prior 8 1H bars (excluding latest)
          2. Current price > swing high  → Attack confirmed
          3. Last 4 15M bars: at least one bearish candle (pullback) but all
             close above the swing high (support holds)
        """
        if not bars_1h or not bars_15m or not price:
            return False

        look = bars_1h[-9:-1] if len(bars_1h) >= 9 else bars_1h[:-1]
        if not look:
            return False
        swing_high = max(b["high"] for b in look)

        if price <= swing_high:
            return False

        recent = bars_15m[-4:] if len(bars_15m) >= 4 else bars_15m
        if not recent:
            return False

        had_pullback = any(b["close"] < b["open"] for b in recent)
        held_support = all(b["close"] > swing_high for b in recent)
        return had_pullback and held_support

    # ── Grade ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _grade_from_count(n: int) -> str:
        if n == 6:  return "A+"
        if n == 5:  return "A"
        if n == 4:  return "B"
        if n == 3:  return "C"
        return "D"

    @staticmethod
    def _empty_analysis(reason: str = "") -> dict:
        return {
            "trend_daily":    "neutral",
            "trend_4h":       "neutral",
            "trend_1h":       "neutral",
            "trend_15m":      "neutral",
            "fvg_active":     False,
            "asp_setup":      False,
            "criteria_count": 0,
            "grade":          "D",
            "error":          reason,
        }
