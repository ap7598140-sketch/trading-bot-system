"""
Bot 5 – Strategy Agent
Strategy: 8am Range + Key Level Sweep + FVG Entry

Core flow:
  9:00am    Capture the 8am 1H candle (high/low) for all symbols.
            Build key levels: prev day/week H/L + nearest round numbers.
  Every 60s during windows — scan 1M bars for sweep pattern:
            Price breaks outside range → touches key level → closes back inside
            → Find FVG entry zone on 1M chart
            → Confirm direction with SPY/QQQ/IWM
            → Publish trade setup

Trading windows:
  Window 1: 9:35am – 11:30am EST
  Window 2: 2:00pm  – 3:30pm  EST

Position sizing:
  Indices (SPY/QQQ/IWM/DIA): $3,000
  Leveraged ETFs (TQQQ/SPXS…): $2,000
  Sector ETFs (XLK/XLF…): $1,500
  Individual stocks: $2,000

Bearish setups always use the corresponding inverse ETF (never short directly).
"""

import asyncio
import math
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Optional

import pytz

from alpaca.data.timeframe import TimeFrame

from config import Models, RedisConfig, RiskConfig, TradingWindowConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


MARKET_TZ = pytz.timezone("America/New_York")

# ── Symbol universe ─────────────────────────────────────────────────────────────

SYMBOLS_INDICES   = ["SPY", "QQQ", "IWM", "DIA"]
SYMBOLS_LEVERAGED = ["TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS"]
SYMBOLS_SECTORS   = ["XLK", "XLF", "XLE", "XLV", "XLI"]
SYMBOLS_STOCKS    = [
    "NVDA", "AAPL", "TSLA", "META", "AMD", "GOOGL",
    "AMZN", "MSFT", "PLTR", "CRWD", "RDDT", "SHOP",
    "COIN", "SOFI", "DASH", "UBER",
]
ALL_SYMBOLS  = SYMBOLS_INDICES + SYMBOLS_LEVERAGED + SYMBOLS_SECTORS + SYMBOLS_STOCKS
# Scan priority: indices first (they set the confirmed direction)
SCAN_ORDER   = SYMBOLS_INDICES + SYMBOLS_LEVERAGED + SYMBOLS_SECTORS + SYMBOLS_STOCKS

# ── Position sizing by category ────────────────────────────────────────────────

POSITION_SIZE: dict[str, int] = {
    **{s: 3000 for s in SYMBOLS_INDICES},
    **{s: 2000 for s in SYMBOLS_LEVERAGED},
    **{s: 1500 for s in SYMBOLS_SECTORS},
    **{s: 2000 for s in SYMBOLS_STOCKS},
}

# ── Inverse ETF for bearish setups ─────────────────────────────────────────────

INVERSE_ETF: dict[str, str] = {
    "SPY":  "SPXS",  "QQQ":  "SQQQ",  "IWM": "SPXS",  "DIA": "SPXS",
    "SPXL": "SPXS",  "TQQQ": "SQQQ",  "SOXL": "SOXS",
    "XLK":  "SQQQ",  "XLF":  "SPXS",  "XLE": "SPXS",
    "XLV":  "SPXS",  "XLI":  "SPXS",
}

# ── Round-number increments for key level detection ────────────────────────────

ROUND_INC: dict[str, float] = {
    "SPY": 5, "QQQ": 10, "IWM": 5, "DIA": 100,
}
DEFAULT_ROUND_INC = 10.0

# ── Strategy scanning windows (separate from execution window) ─────────────────

_W1_OPEN  = dt_time(9,  35)
_W1_CLOSE = dt_time(11, 30)
_W2_OPEN  = dt_time(14,  0)
_W2_CLOSE = dt_time(15, 30)


class StrategyAgent(BaseBot):
    """
    Bot 5 – Strategy Agent
    Implements the 8am Range + Key Level Sweep + FVG Entry strategy.
    Rule-based — no AI call required for trade decisions.
    """

    BOT_ID = 5
    NAME   = "Strategy Agent"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.alpaca = AlpacaClient()

        # Real-time market data (price fallback from bot02 pub/sub)
        self._market_data: dict[str, dict] = {}

        # 8am range and key levels — populated at 9:00am, cleared each morning
        self._eight_am_ranges: dict[str, dict] = {}   # {sym: {high, low, range}}
        self._key_levels:      dict[str, dict] = {}   # {sym: {kl_above, kl_below, …}}

        # Daily trade tracking — cleared each morning
        self._first_sweep_used: set[str]       = set()   # symbol already used first sweep
        self._directions_used:  dict[str, set] = {}      # {sym: {"long"} or {"short"}}
        self._trade_count:      dict[str, int] = {}      # {sym: n} capped at 2

        # Index sweep direction observed this scan pass (for confirmation logic)
        self._index_direction: dict[str, str] = {}       # {sym: "long"/"short"}

        # Risk / regime state from bot08
        self._portfolio_value: float = 0.0
        self._current_regime:  str   = "neutral"
        self._regime_scale:    float = 1.0
        self._cb_scale:        float = 1.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA, self._on_market_data)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._morning_range_task())
        await self._refresh_portfolio()

        # If bot starts mid-day (past 9am) capture ranges immediately
        now_et = datetime.now(MARKET_TZ)
        if now_et.hour >= 9:
            asyncio.create_task(self._capture_8am_ranges())

        self.log(
            "Strategy Agent starting | 8am Range + KL Sweep + FVG | "
            f"windows: {_W1_OPEN.strftime('%H:%M')}–{_W1_CLOSE.strftime('%H:%M')} "
            f"and {_W2_OPEN.strftime('%H:%M')}–{_W2_CLOSE.strftime('%H:%M')} ET"
        )

    async def run(self):
        """Sweep scan loop: fires every 60s, active only during strategy windows."""
        while self.running:
            await asyncio.sleep(60)
            if not self._in_strategy_window():
                continue
            if not self._eight_am_ranges:
                self.log("Waiting for 8am ranges — scan skipped", "warning")
                continue
            await self._refresh_scales()
            try:
                await self._sweep_scan()
            except Exception as e:
                self.log(f"Sweep scan error: {e}", "error")

    async def cleanup(self):
        self.log("Strategy Agent stopped")

    # ── Price feed (fallback) ──────────────────────────────────────────────────

    async def _on_market_data(self, data: dict):
        sym = data.get("symbol")
        if sym:
            self._market_data[sym] = data

    # ── 9:00am range capture task ─────────────────────────────────────────────

    async def _morning_range_task(self):
        """Sleep until 9:00am ET each day, then capture 8am ranges."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now_et).total_seconds())
            try:
                await self._capture_8am_ranges()
            except Exception as e:
                self.log(f"Morning range capture error: {e}", "error")

    async def _capture_8am_ranges(self):
        """
        Fetch 1H and daily bars for all symbols.
        Extract the 8am 1H candle for the range.
        Build key levels from daily bars.
        """
        self.log("Capturing 8am ranges and key levels for all symbols…")
        self._eight_am_ranges.clear()
        self._key_levels.clear()
        self._first_sweep_used.clear()
        self._directions_used.clear()
        self._trade_count.clear()
        self._index_direction.clear()

        loop  = asyncio.get_event_loop()
        start = (datetime.now(MARKET_TZ) - timedelta(days=12)).strftime("%Y-%m-%d")

        # Fetch in batches of 10 to stay within Alpaca rate limits
        for i in range(0, len(ALL_SYMBOLS), 10):
            batch = ALL_SYMBOLS[i:i + 10]
            try:
                bars_1h, bars_day = await asyncio.gather(
                    asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda b=batch: self.alpaca.get_bars(b, TimeFrame.Hour, start)
                        ),
                        timeout=30.0,
                    ),
                    asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda b=batch: self.alpaca.get_bars(b, TimeFrame.Day, start)
                        ),
                        timeout=30.0,
                    ),
                )
            except Exception as e:
                self.log(f"Range fetch error (batch {batch[:3]}…): {e}", "warning")
                continue

            for sym in batch:
                self._extract_8am_range(sym, bars_1h.get(sym, []))
                self._build_key_levels(sym, bars_day.get(sym, []))

        captured = len(self._eight_am_ranges)
        spy_rng  = self._eight_am_ranges.get("SPY", {})
        self.log(
            f"8am ranges: {captured}/{len(ALL_SYMBOLS)} symbols captured | "
            f"SPY 8am range: ${spy_rng.get('low', 0):.2f} – ${spy_rng.get('high', 0):.2f}"
        )
        await self._refresh_portfolio()

    def _extract_8am_range(self, sym: str, bars_1h: list[dict]):
        """Find today's 8am 1H candle and record its high/low."""
        if not bars_1h:
            return
        today = datetime.now(MARKET_TZ).date()
        for bar in reversed(bars_1h):
            try:
                dt = datetime.fromisoformat(bar["timestamp"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_et = dt.astimezone(MARKET_TZ)
                if dt_et.date() == today and dt_et.hour == 8:
                    self._eight_am_ranges[sym] = {
                        "high":  bar["high"],
                        "low":   bar["low"],
                        "range": bar["high"] - bar["low"],
                    }
                    return
            except Exception:
                continue

    def _build_key_levels(self, sym: str, bars_day: list[dict]):
        """
        Build key levels from daily bars:
          kl_above — nearest level above 8am_high (prev day high / prev week high / round)
          kl_below — nearest level below 8am_low  (prev day low  / prev week low  / round)
        """
        if not bars_day:
            return
        r = self._eight_am_ranges.get(sym)
        if not r:
            return

        prev_bar  = bars_day[-2] if len(bars_day) >= 2 else bars_day[-1]
        prev_high = prev_bar["high"]
        prev_low  = prev_bar["low"]

        # Previous week H/L — last 5 bars before this week
        today_et   = datetime.now(MARKET_TZ).date()
        week_start = today_et - timedelta(days=today_et.weekday())
        prior_week = [b for b in bars_day if self._bar_date(b) < week_start]
        pw_bars    = prior_week[-5:] if prior_week else [prev_bar]
        pw_high    = max(b["high"] for b in pw_bars)
        pw_low     = min(b["low"]  for b in pw_bars)

        rng_high = r["high"]
        rng_low  = r["low"]

        round_above = self._nearest_round(sym, rng_high, "above")
        round_below = self._nearest_round(sym, rng_low,  "below")

        above_candidates = sorted(
            [l for l in [prev_high, pw_high, round_above] if l > rng_high]
        )
        below_candidates = sorted(
            [l for l in [prev_low, pw_low, round_below] if l < rng_low],
            reverse=True,
        )

        self._key_levels[sym] = {
            "kl_above":       above_candidates[0] if above_candidates else rng_high * 1.005,
            "kl_below":       below_candidates[0] if below_candidates else rng_low  * 0.995,
            "prev_day_high":  prev_high,
            "prev_day_low":   prev_low,
            "prev_week_high": pw_high,
            "prev_week_low":  pw_low,
        }

    @staticmethod
    def _bar_date(bar: dict):
        try:
            return datetime.fromisoformat(bar["timestamp"]).date()
        except Exception:
            return datetime.min.date()

    def _nearest_round(self, sym: str, price: float, direction: str) -> float:
        inc = ROUND_INC.get(sym, DEFAULT_ROUND_INC)
        if direction == "above":
            return math.ceil(price / inc) * inc
        return math.floor(price / inc) * inc

    # ── Strategy window check ──────────────────────────────────────────────────

    @staticmethod
    def _in_strategy_window() -> bool:
        t = datetime.now(MARKET_TZ).time()
        return _W1_OPEN <= t <= _W1_CLOSE or _W2_OPEN <= t <= _W2_CLOSE

    # ── Daily P&L gate ─────────────────────────────────────────────────────────

    async def _pnl_allows_trading(self) -> bool:
        try:
            dashboard = await self.bus.get_state("bot8:dashboard") or {}
            pnl = float(dashboard.get("daily_pnl", 0))
            if pnl <= -RiskConfig.DAILY_LOSS_LIMIT_USD:
                self.log(f"Daily loss limit hit (${pnl:.2f}) — scan blocked", "warning")
                return False
            if pnl >= RiskConfig.DAILY_PROFIT_TARGET_USD:
                self.log(f"Daily profit target hit (+${pnl:.2f}) — scan blocked", "warning")
                return False
        except Exception:
            pass
        return True

    # ── Main sweep scan ────────────────────────────────────────────────────────

    async def _sweep_scan(self):
        """
        Fetch 1M bars for all symbols, then check each in priority order.
        Indices are evaluated first — their sweep direction is used to confirm
        (or veto) setups on leveraged ETFs, sectors, and stocks.
        """
        if self._cb_scale == 0.0:
            return
        if not await self._pnl_allows_trading():
            return

        loop  = asyncio.get_event_loop()
        start = (datetime.now(timezone.utc) - timedelta(minutes=40)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            bars_1m_all = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.alpaca.get_bars(SCAN_ORDER, TimeFrame.Minute, start)
                ),
                timeout=30.0,
            )
        except Exception as e:
            self.log(f"1M bar fetch error: {e}", "warning")
            return

        # Phase 1: indices — records their direction in self._index_direction
        for sym in SYMBOLS_INDICES:
            await self._check_symbol(sym, bars_1m_all.get(sym, []))

        # Phase 2: rest of universe — uses index_direction for confirmation
        for sym in SYMBOLS_LEVERAGED + SYMBOLS_SECTORS + SYMBOLS_STOCKS:
            await self._check_symbol(sym, bars_1m_all.get(sym, []))

    # ── Per-symbol evaluation ─────────────────────────────────────────────────

    async def _check_symbol(self, sym: str, bars_1m: list[dict]):
        r  = self._eight_am_ranges.get(sym)
        kl = self._key_levels.get(sym)
        if not r or not kl or not bars_1m:
            return
        if self._trade_count.get(sym, 0) >= 2:
            return

        for direction in ("long", "short"):
            if direction in self._directions_used.get(sym, set()):
                continue

            sweep = self._detect_sweep(bars_1m, r, kl, direction)
            if not sweep:
                continue

            fvg = self._find_fvg_1m(bars_1m, direction)
            if not fvg:
                self.log(
                    f"{sym} {direction}: sweep confirmed but no FVG found — skip",
                    "warning",
                )
                continue

            confidence, conf_label = self._index_confidence(sym, direction)
            if confidence == 0.0:
                self.log(
                    f"{sym} {direction}: index confirmation disagrees — skip",
                    "warning",
                )
                continue

            await self._publish_setup(sym, direction, r, kl, sweep, fvg,
                                      confidence, conf_label)
            return   # one setup per symbol per scan

    # ── Sweep detection ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_sweep(
        bars_1m: list[dict],
        r: dict,
        kl: dict,
        direction: str,
    ) -> Optional[dict]:
        """
        Bullish (direction='long'):
          1. A candle's low went below 8am_low (swept support)
          2. That low is within 0.5% of kl_below (touched the key level)
          3. A subsequent candle (within 3 bars) closed back above 8am_low

        Bearish (direction='short'):
          Mirror pattern above 8am_high and kl_above.

        Only looks at the last 15 1M candles (~15 minutes).
        Returns sweep info dict or None.
        """
        if len(bars_1m) < 4:
            return None

        recent   = bars_1m[-15:]
        rng_high = r["high"]
        rng_low  = r["low"]
        kl_above = kl["kl_above"]
        kl_below = kl["kl_below"]

        if direction == "long":
            for i, bar in enumerate(recent[:-1]):
                if bar["low"] >= rng_low:
                    continue
                swept_low = bar["low"]
                if abs(swept_low - kl_below) / kl_below > 0.005:
                    continue
                recovery = recent[i + 1: i + 4]
                if any(b["close"] > rng_low for b in recovery):
                    return {"direction": "long",
                            "sweep_level": swept_low,
                            "kl_touched":  kl_below}
        else:
            for i, bar in enumerate(recent[:-1]):
                if bar["high"] <= rng_high:
                    continue
                swept_high = bar["high"]
                if abs(swept_high - kl_above) / kl_above > 0.005:
                    continue
                recovery = recent[i + 1: i + 4]
                if any(b["close"] < rng_high for b in recovery):
                    return {"direction": "short",
                            "sweep_level": swept_high,
                            "kl_touched":  kl_above}

        return None

    # ── FVG detection on 1M ────────────────────────────────────────────────────

    @staticmethod
    def _find_fvg_1m(bars_1m: list[dict], direction: str) -> Optional[dict]:
        """
        Bullish FVG: candle[i+2].low > candle[i].high
        Bearish FVG: candle[i+2].high < candle[i].low
        Searches the last 10 1M bars; returns the most recent valid gap.
        """
        check = bars_1m[-10:]
        fvgs  = []
        for i in range(len(check) - 2):
            c0, c2 = check[i], check[i + 2]
            if direction == "long" and c2["low"] > c0["high"]:
                low, high = c0["high"], c2["low"]
                fvgs.append({"low": low, "high": high, "mid": (low + high) / 2})
            elif direction == "short" and c2["high"] < c0["low"]:
                low, high = c2["high"], c0["low"]
                fvgs.append({"low": low, "high": high, "mid": (low + high) / 2})
        return fvgs[-1] if fvgs else None

    # ── Index confirmation ─────────────────────────────────────────────────────

    def _index_confidence(self, sym: str, direction: str) -> tuple[float, str]:
        """
        Returns (confidence, label).

        Index symbols confirm themselves (no external check needed).
        For all other symbols:
          SPY + QQQ + IWM all agree  → 0.90  VERY HIGH
          Any 2 of 3 agree           → 0.85  HIGH
          Only 1 agrees              → 0.75  MEDIUM
          Any index opposes          → 0.00  DISAGREE (skip)
          No index data yet          → 0.65  UNCONFIRMED
        """
        if sym in SYMBOLS_INDICES:
            self._index_direction[sym] = direction
            return 0.75, "INDEX"

        matching = sum(
            1 for s in ("SPY", "QQQ", "IWM")
            if self._index_direction.get(s) == direction
        )
        opposing = sum(
            1 for s in ("SPY", "QQQ", "IWM")
            if self._index_direction.get(s) not in (None, "", direction)
        )

        if opposing:
            return 0.0, "DISAGREE"
        if matching == 3:
            return 0.90, "VERY HIGH"
        if matching == 2:
            return 0.85, "HIGH"
        if matching == 1:
            return 0.75, "MEDIUM"
        return 0.65, "UNCONFIRMED"

    # ── Setup publishing ───────────────────────────────────────────────────────

    async def _publish_setup(
        self,
        sym: str,
        direction: str,
        r: dict,
        kl: dict,
        sweep: dict,
        fvg: dict,
        confidence: float,
        conf_label: str,
    ):
        # Bearish → buy the inverse ETF instead
        trade_sym = sym if direction == "long" else INVERSE_ETF.get(sym, "SPXS")
        entry     = round(fvg["mid"], 4)

        if direction == "long":
            stop = round(sweep["sweep_level"] * 0.999, 4)   # just below sweep low
            t1   = round(r["high"], 4)                        # 8am high (close 50%)
            t2   = round(kl["prev_day_high"], 4)              # prev day high (trail rest)
        else:
            # Inverse ETF — use percentage-based targets
            stop = round(entry * 0.990, 4)                    # 1% below entry
            t1   = round(entry * 1.020, 4)                    # +2%
            t2   = round(entry * 1.040, 4)                    # +4%

        scale        = max(0.0, min(self._regime_scale * self._cb_scale, 1.0))
        position_usd = round(POSITION_SIZE.get(sym, 2000) * scale, 2)
        position_usd = max(position_usd, 500.0)

        grade    = "A" if confidence >= 0.85 else "B"
        category = (
            "index"    if sym in SYMBOLS_INDICES   else
            "leveraged" if sym in SYMBOLS_LEVERAGED else
            "sector"   if sym in SYMBOLS_SECTORS   else
            "stock"
        )

        # Record usage so we don't re-enter same direction or exceed daily limit
        self._first_sweep_used.add(sym)
        self._directions_used.setdefault(sym, set()).add(direction)
        self._trade_count[sym] = self._trade_count.get(sym, 0) + 1
        if sym in SYMBOLS_INDICES:
            self._index_direction[sym] = direction

        setup = {
            "type":              "trade_setup",
            "symbol":            trade_sym,
            "source_symbol":     sym,
            "direction":         "long",      # always long (inverse ETF for bear)
            "entry_price":       entry,
            "stop_loss":         stop,
            "take_profit":       t1,
            "take_profit_2":     t2,
            "position_size_usd": position_usd,
            "confidence":        confidence,
            "grade":             grade,
            "winner_score":      6,           # sweep+FVG+index = rule-based validated
            "category":          category,
            "market_bias":       "bullish" if direction == "long" else "bearish",
            "timeframe":         "1m",
            "catalyst": (
                f"{'Bullish' if direction == 'long' else 'Bearish'} KL sweep "
                f"({'8am low' if direction == 'long' else '8am high'}) + FVG entry"
            ),
            "thesis": (
                f"{sym} swept {'below' if direction == 'long' else 'above'} "
                f"8am {'low' if direction == 'long' else 'high'} "
                f"(${r['low'] if direction == 'long' else r['high']:.2f}), "
                f"touched KL ${sweep['kl_touched']:.2f}, closed back inside. "
                f"FVG ${fvg['low']:.2f}–${fvg['high']:.2f}. "
                f"Index confirmation: {conf_label}"
            ),
            "sweep_level":        sweep["sweep_level"],
            "kl_touched":         sweep["kl_touched"],
            "fvg_zone":           [fvg["low"], fvg["high"]],
            "eight_am_high":      r["high"],
            "eight_am_low":       r["low"],
            "prev_day_high":      kl["prev_day_high"],
            "prev_day_low":       kl["prev_day_low"],
            "index_confirmation": conf_label,
        }

        # Telegram alert
        cat_emoji = "🔵" if category == "index" else "⚡" if category == "leveraged" else "📊"
        dir_label = "LONG" if direction == "long" else f"SHORT → buy {trade_sym}"
        alert = (
            f"{cat_emoji} <b>{category.upper()} SETUP: {sym}</b>\n\n"
            f"Direction: {dir_label}\n"
            f"Setup: {'Bullish' if direction == 'long' else 'Bearish'} KL Sweep + FVG\n"
            f"8am Range: ${r['low']:.2f} – ${r['high']:.2f}\n"
            f"Sweep {'Low' if direction == 'long' else 'High'}: ${sweep['sweep_level']:.2f}\n"
            f"KL Touched: ${sweep['kl_touched']:.2f}\n"
            f"FVG Zone: ${fvg['low']:.2f} – ${fvg['high']:.2f}\n"
            f"Entry: ${entry:.2f}\n"
            f"Stop: ${stop:.2f}\n"
            f"Target 1: ${t1:.2f} "
            f"{'(8am high — close 50%)' if direction == 'long' else '(+2%)'}\n"
            f"Target 2: ${t2:.2f} "
            f"{'(prev day high — trail rest)' if direction == 'long' else '(+4%)'}\n"
            f"Confidence: {conf_label} ({confidence:.0%})\n"
            f"Size: ${position_usd:,.0f}"
        )
        setup["telegram_alert"] = alert

        await self.publish(RedisConfig.CHANNEL_STRATEGY, setup)
        self.log(
            f"SETUP: {trade_sym} long (source={sym} {direction}) | "
            f"entry=${entry:.2f} sl=${stop:.2f} t1=${t1:.2f} t2=${t2:.2f} | "
            f"conf={conf_label} ({confidence:.0%}) | size=${position_usd:,.0f}"
        )

    # ── Portfolio / risk scale helpers ─────────────────────────────────────────

    async def _refresh_portfolio(self):
        try:
            loop    = asyncio.get_event_loop()
            account = await loop.run_in_executor(None, self.alpaca.get_account)
            self._portfolio_value = float(account["portfolio_value"])
            self.log(f"Portfolio value: ${self._portfolio_value:,.2f}")
        except Exception as e:
            self.log(f"Portfolio refresh error: {e}", "warning")

    async def _refresh_scales(self):
        try:
            s = await self.bus.get_state("bot8:regime") or {}
            self._regime_scale   = float(s.get("scale", 1.0))
            self._current_regime = s.get("regime", "neutral")
            if self._current_regime == "crash":
                self._regime_scale = 0.0
        except Exception:
            pass
        try:
            s = await self.bus.get_state("bot8:cb_state") or {}
            self._cb_scale = float(s.get("position_scale", 1.0))
            if not s.get("trading_allowed", True):
                self._cb_scale = 0.0
        except Exception:
            pass
        try:
            g = await self.bus.get_state("bot8:market_gate") or {}
            if not g.get("allow_new_trades", True):
                self._cb_scale = 0.0
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = StrategyAgent()
    asyncio.run(bot.start())
