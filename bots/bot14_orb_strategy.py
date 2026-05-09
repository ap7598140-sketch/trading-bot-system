"""
Bot 14 – 8am ORB Midpoint Strategy
Strategy: Opening Range Breakout + Midpoint Rejection Entry

Daily flow:
  8:15am  Capture the 8:00–8:15am 15M candle for SPY (high/low/mid).
          Assess market regime (SPY vs 20MA + VIX) → set position size.
          → Telegram: range + regime

  9:30am+ Watch every 60s for first breakout above high or below low.
          → Telegram: breakout detected, waiting for midpoint return

  After breakout: watch every 60s for rejection candle at midpoint.
    Bullish rejection: low ≤ mid AND close > mid AND green candle → buy SPY
    Bearish rejection: high ≥ mid AND close < mid AND red candle  → buy SPXS

  On entry: place market order, set stop/target, → Telegram

  In trade: check every 60s for target / stop / 3:45pm EOD close.
          → Telegram on exit

Rules:
  - One trade per day max
  - Stop looking for new setups after 2pm
  - Never hold overnight (hard EOD close at 3:45pm)
  - Long = SPY, Short = SPXS (inverse ETF, never short directly)
  - Skip entirely when VIX > 25 or SPY down > 2%

Position sizing by regime:
  STRONG BULL  (SPY>20MA, VIX<15, range>$2) → $5,000
  NORMAL       (VIX ≤ 20, range ≥ $1)       → $2,500
  WEAK/CHOPPY  (VIX ≤ 25)                   → $1,000
  HIGH VOL     (VIX > 25)                   → $0 (skip)
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone, date as dt_date, time as dt_time
from typing import Optional
import pytz

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import Models, RedisConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


MARKET_TZ = pytz.timezone("America/New_York")

# ── Timing constants ───────────────────────────────────────────────────────────
_RANGE_CAPTURE = dt_time(8,  15)   # fetch 8am candle when this passes
_MARKET_OPEN   = dt_time(9,  30)   # start scanning for breakout
_SEARCH_CUTOFF = dt_time(14,  0)   # stop looking for new setups
_EOD_CLOSE     = dt_time(15, 45)   # hard close everything

# ── State labels ───────────────────────────────────────────────────────────────
_IDLE      = "IDLE"        # waiting for 8:15am
_RANGE_SET = "RANGE_SET"   # range captured, waiting for market open
_WATCHING  = "WATCHING"    # scanning for first breakout
_PULLBACK  = "PULLBACK"    # breakout confirmed, waiting for midpoint rejection
_IN_TRADE  = "IN_TRADE"    # position open
_DONE      = "DONE"        # trade done or daily cutoff reached


class ORBMidpointBot(BaseBot):
    """
    Bot 14 – 8am ORB Midpoint Strategy.
    Self-contained: places and manages its own orders via AlpacaClient.
    Publishes Telegram alerts to CHANNEL_ALERTS.
    """

    BOT_ID = 14
    NAME   = "ORB Midpoint"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.alpaca = AlpacaClient()

        # Refreshed each morning
        self._trade_date:     Optional[dt_date] = None
        self._state:          str   = _IDLE

        # 8am range
        self._high:  float = 0.0
        self._low:   float = 0.0
        self._mid:   float = 0.0

        # Market regime
        self._regime:     str   = ""
        self._size_usd:   float = 0.0
        self._vix:        float = 0.0
        self._spy_vs_20ma: str  = ""

        # Breakout
        self._breakout_dir:   str   = ""    # "bullish" | "bearish"
        self._breakout_price: float = 0.0

        # Trade
        self._symbol:   str   = ""    # "SPY" or "SPXS"
        self._direction: str  = ""    # "long"  (always long — SPXS for bear)
        self._entry:    float = 0.0   # price paid
        self._stop:     float = 0.0   # SPY level for stop (both directions)
        self._target:   float = 0.0   # SPY level for target
        self._shares:   int   = 0
        self._risk_per_share: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log(
            "ORB Midpoint Bot starting | "
            "SPY 8am range → breakout → midpoint rejection entry"
        )

    async def run(self):
        """Main loop: every 60s run the state machine."""
        while self.running:
            await asyncio.sleep(60)
            try:
                await self._tick()
            except Exception as e:
                self.log(f"ORB tick error: {e}", "error")

    async def cleanup(self):
        self.log(f"ORB bot stopped | state={self._state}")

    # ── State machine ─────────────────────────────────────────────────────────

    async def _tick(self):
        now_et = datetime.now(MARKET_TZ)
        today  = now_et.date()
        t      = now_et.time()

        # Daily reset when a new trading day starts
        if self._trade_date != today and t >= dt_time(7, 0):
            self._daily_reset(today)

        if self._state == _IDLE:
            if t >= _RANGE_CAPTURE:
                await self._capture_range()

        elif self._state == _RANGE_SET:
            if t >= _MARKET_OPEN:
                self._state = _WATCHING
                self.log(
                    f"Market open — scanning for breakout | "
                    f"range ${self._low:.2f}–${self._high:.2f} mid ${self._mid:.2f}"
                )

        elif self._state == _WATCHING:
            if t >= _SEARCH_CUTOFF:
                self.log("2pm — no breakout detected today, done")
                self._state = _DONE
            else:
                await self._scan_breakout()

        elif self._state == _PULLBACK:
            if t >= _SEARCH_CUTOFF:
                self.log("2pm — no rejection entry found, done")
                self._state = _DONE
            else:
                await self._scan_rejection()

        elif self._state == _IN_TRADE:
            await self._manage_trade(t)

        # EOD safety close (belt-and-suspenders if somehow still in trade)
        if t >= _EOD_CLOSE and self._symbol and self._state == _IN_TRADE:
            await self._close_position("eod")

    # ── Step 1: Capture 8am range ─────────────────────────────────────────────

    async def _capture_range(self):
        """Fetch the 8:00–8:15am 15M SPY candle and set high/low/mid."""
        loop  = asyncio.get_event_loop()
        start = (datetime.now(MARKET_TZ) - timedelta(days=2)).strftime("%Y-%m-%d")
        tf_15m = TimeFrame(15, TimeFrameUnit.Minute)
        try:
            bars = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.alpaca.get_bars(["SPY"], tf_15m, start)
                ),
                timeout=15.0,
            )
        except Exception as e:
            self.log(f"Range capture error: {e} — will retry", "warning")
            return

        today = datetime.now(MARKET_TZ).date()
        bar   = None
        for b in reversed(bars.get("SPY", [])):
            try:
                dt = datetime.fromisoformat(b["timestamp"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_et = dt.astimezone(MARKET_TZ)
                if dt_et.date() == today and dt_et.hour == 8 and dt_et.minute == 0:
                    bar = b
                    break
            except Exception:
                continue

        if not bar:
            self.log("8am 15M bar not yet available — retrying next minute", "warning")
            return

        self._high  = bar["high"]
        self._low   = bar["low"]
        self._mid   = round((self._high + self._low) / 2, 4)
        self._state = _RANGE_SET

        self.log(
            f"8am range | high=${self._high:.2f} low=${self._low:.2f} "
            f"mid=${self._mid:.2f}"
        )

        await self._assess_regime()

        regime_icon = "🟢" if "BULL" in self._regime else "🟡" if "NORMAL" in self._regime else "🔴"
        skip_note   = "\n⚠️ <b>Trade skipped today</b>" if self._size_usd == 0 else ""
        await self._telegram(
            f"📊 <b>8am Range Set</b>\n\n"
            f"High: ${self._high:.2f}\n"
            f"Low:  ${self._low:.2f}\n"
            f"Mid:  ${self._mid:.2f}\n"
            f"Range: ${self._high - self._low:.2f}\n\n"
            f"{regime_icon} <b>Market: {self._regime}</b>\n"
            f"Trade Size: ${self._size_usd:,.0f}\n"
            f"VIX: {self._vix:.1f}\n"
            f"SPY vs 20MA: {self._spy_vs_20ma}"
            f"{skip_note}"
        )

    # ── Market regime ─────────────────────────────────────────────────────────

    async def _assess_regime(self):
        spy_price, sma20, prev_close = await self._spy_daily_stats()
        self._vix = await self._fetch_vix()

        above_20ma = spy_price > sma20 if sma20 else True
        spy_chg_pct = (spy_price - prev_close) / prev_close * 100 if prev_close else 0
        range_size  = self._high - self._low

        self._spy_vs_20ma = f"{'ABOVE' if above_20ma else 'BELOW'} (${spy_price:.2f} vs ${sma20:.2f})"

        if self._vix > 25:
            self._regime, self._size_usd = "HIGH VOLATILITY", 0.0
        elif spy_chg_pct < -2.0:
            self._regime, self._size_usd = f"SPY DOWN {spy_chg_pct:.1f}%", 0.0
        elif above_20ma and self._vix < 15 and range_size > 2.0:
            self._regime, self._size_usd = "STRONG BULL", 5000.0
        elif self._vix <= 20 and range_size >= 1.0:
            self._regime, self._size_usd = "NORMAL", 2500.0
        elif self._vix <= 25:
            self._regime, self._size_usd = "WEAK/CHOPPY", 1000.0
        else:
            self._regime, self._size_usd = "HIGH VOLATILITY", 0.0

        self.log(
            f"Regime={self._regime} size=${self._size_usd:,.0f} | "
            f"VIX={self._vix:.1f} | range=${range_size:.2f} | "
            f"spy_chg={spy_chg_pct:+.1f}% | {self._spy_vs_20ma}"
        )

    async def _spy_daily_stats(self) -> tuple[float, float, float]:
        """Return (latest_close, sma20, prev_close) from SPY daily bars."""
        loop  = asyncio.get_event_loop()
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        try:
            bars   = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.alpaca.get_bars(["SPY"], TimeFrame.Day, start)
                ),
                timeout=15.0,
            )
            closes = [b["close"] for b in bars.get("SPY", [])]
            if len(closes) < 2:
                return 0.0, 0.0, 0.0
            sma20  = sum(closes[-20:]) / min(20, len(closes))
            return closes[-1], round(sma20, 4), closes[-2]
        except Exception as e:
            self.log(f"SPY daily stats error: {e}", "warning")
            return 0.0, 0.0, 0.0

    async def _fetch_vix(self) -> float:
        """Fetch VIX from Yahoo Finance chart API. Returns 20.0 on failure (safe default)."""
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            "%5EVIX?interval=1d&range=2d"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=8),
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as resp:
                    if resp.status != 200:
                        return 20.0
                    data  = await resp.json()
                    price = (
                        data.get("chart", {})
                            .get("result", [{}])[0]
                            .get("meta", {})
                            .get("regularMarketPrice", 20.0)
                    )
                    return float(price)
        except Exception as e:
            self.log(f"VIX fetch error: {e} — using 20.0", "warning")
            return 20.0

    # ── Step 2: Breakout detection ─────────────────────────────────────────────

    async def _scan_breakout(self):
        """Check current SPY price for first close outside the 8am range."""
        price = await self._spy_price()
        if not price:
            return

        if price > self._high:
            self._breakout_dir   = "bullish"
            self._breakout_price = price
            self._state          = _PULLBACK
            self.log(f"BULLISH BREAK @ ${price:.2f} | waiting for mid ${self._mid:.2f}")
            await self._telegram(
                f"🔔 <b>BULLISH BREAK @ ${price:.2f}</b>\n"
                f"Waiting for pullback to mid ${self._mid:.2f}"
            )

        elif price < self._low:
            self._breakout_dir   = "bearish"
            self._breakout_price = price
            self._state          = _PULLBACK
            self.log(f"BEARISH BREAK @ ${price:.2f} | waiting for mid ${self._mid:.2f}")
            await self._telegram(
                f"🔔 <b>BEARISH BREAK @ ${price:.2f}</b>\n"
                f"Waiting for return to mid ${self._mid:.2f}"
            )

    # ── Step 3+4: Rejection candle ─────────────────────────────────────────────

    async def _scan_rejection(self):
        """
        Fetch the most recent completed 1M candle and check for a rejection at mid.

        Bullish rejection (→ long SPY):
          candle.low ≤ mid  AND  close > mid  AND  close > open  (green)

        Bearish rejection (→ long SPXS):
          candle.high ≥ mid  AND  close < mid  AND  close < open  (red)
        """
        if self._size_usd == 0.0:
            self.log(f"Skipping: {self._regime} → no trade", "warning")
            self._state = _DONE
            return

        loop  = asyncio.get_event_loop()
        start = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            bars = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.alpaca.get_bars(["SPY"], TimeFrame.Minute, start)
                ),
                timeout=10.0,
            )
        except Exception as e:
            self.log(f"Rejection scan error: {e}", "warning")
            return

        spy_bars = bars.get("SPY", [])
        if not spy_bars:
            return
        bar = spy_bars[-1]

        if self._breakout_dir == "bullish":
            if (bar["low"] <= self._mid
                    and bar["close"] > self._mid
                    and bar["close"] > bar["open"]):
                self.log(
                    f"BULLISH REJECTION at ${bar['close']:.2f} "
                    f"(low=${bar['low']:.2f} ≤ mid=${self._mid:.2f})"
                )
                await self._enter("long", "SPY", bar["close"])

        elif self._breakout_dir == "bearish":
            if (bar["high"] >= self._mid
                    and bar["close"] < self._mid
                    and bar["close"] < bar["open"]):
                self.log(
                    f"BEARISH REJECTION at ${bar['close']:.2f} "
                    f"(high=${bar['high']:.2f} ≥ mid=${self._mid:.2f})"
                )
                await self._enter("short", "SPXS", bar["close"])

    # ── Step 5: Enter trade ────────────────────────────────────────────────────

    async def _enter(self, direction: str, symbol: str, spy_price: float):
        """
        Calculate stop / target in SPY terms, fetch actual fill price,
        place market order, send Telegram.
        """
        if direction == "long":
            self._stop   = self._low                             # 8am low
            risk         = spy_price - self._stop
            self._target = round(spy_price + risk * 1.5, 4)    # 1.5× risk
        else:
            self._stop   = self._high                            # 8am high
            risk         = self._stop - spy_price
            self._target = round(spy_price - risk * 1.5, 4)    # 1.5× risk below

        if risk <= 0:
            self.log(f"Invalid risk ${risk:.4f} — skipping entry", "warning")
            return

        self._risk_per_share = round(risk, 4)

        # Get live price of the actual symbol to buy
        live_price = await self._live_price(symbol)
        if not live_price:
            live_price = spy_price

        self._shares = max(1, int(self._size_usd / live_price))
        if self._shares == 0:
            return

        # Place market order via Alpaca
        loop = asyncio.get_event_loop()
        try:
            order = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.alpaca.trading.submit_order(
                        MarketOrderRequest(
                            symbol=symbol,
                            qty=self._shares,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )
                    ),
                ),
                timeout=10.0,
            )
            self.log(f"Order placed: {order.id} {symbol} ×{self._shares}")
        except Exception as e:
            self.log(f"Order placement error: {e}", "error")
            return

        self._symbol    = symbol
        self._direction = direction
        self._entry     = live_price
        self._state     = _IN_TRADE

        reward = round(risk * 1.5, 4)
        dir_label = "BULLISH 📈" if direction == "long" else "BEARISH 📉 (SPXS)"

        self.log(
            f"IN TRADE: {symbol} ×{self._shares} | "
            f"entry=${live_price:.2f} stop(SPY)=${self._stop:.2f} "
            f"target(SPY)=${self._target:.2f} | "
            f"risk=${risk:.2f}/sh reward=${reward:.2f}/sh"
        )
        await self._telegram(
            f"🎯 <b>8am ORB TRADE</b>\n\n"
            f"Direction: {dir_label}\n"
            f"Symbol: {symbol}\n"
            f"Entry: ${live_price:.2f}\n"
            f"Stop (SPY): ${self._stop:.2f}\n"
            f"Target (SPY): ${self._target:.2f}\n"
            f"Risk: ${risk:.2f}/share\n"
            f"Reward: ${reward:.2f}/share\n"
            f"Shares: {self._shares}\n"
            f"Size: ${self._size_usd:,.0f}"
        )

    # ── Step 6: Manage trade ───────────────────────────────────────────────────

    async def _manage_trade(self, t: dt_time):
        """Every minute: watch SPY price for target/stop/EOD."""
        spy = await self._spy_price()
        if not spy:
            return

        if self._direction == "long":
            hit_target = spy >= self._target
            hit_stop   = spy <= self._stop
        else:
            hit_target = spy <= self._target
            hit_stop   = spy >= self._stop

        if t >= _EOD_CLOSE:
            await self._close_position("eod")
        elif hit_target:
            await self._close_position("target")
        elif hit_stop:
            await self._close_position("stop")

    async def _close_position(self, reason: str):
        """Close the open position and send exit Telegram."""
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.alpaca.trading.close_position(self._symbol),
                ),
                timeout=10.0,
            )
            self.log(f"Position closed: {self._symbol} reason={reason}")
        except Exception as e:
            self.log(f"Position close error: {e}", "error")

        self._state = _DONE

        # Estimate P&L (approximate — actual fill may differ slightly)
        spy = await self._spy_price() or 0.0
        if self._direction == "long":
            est_pnl = (spy - self._stop) * self._shares if reason == "stop" else \
                      (self._target - self._entry) * self._shares if reason == "target" else \
                      (spy - self._entry) * self._shares
        else:
            est_pnl = -(self._stop - spy) * self._shares if reason == "stop" else \
                       (self._entry - self._target) * self._shares if reason == "target" else \
                       (self._entry - spy) * self._shares

        if reason == "target":
            msg = f"✅ <b>TAKE PROFIT +${abs(est_pnl):.0f}</b>"
        elif reason == "stop":
            msg = f"❌ <b>STOP LOSS -${abs(est_pnl):.0f}</b>"
        else:
            msg = f"🕐 <b>EOD CLOSE</b> (${est_pnl:+.0f} est.)"

        await self._telegram(
            f"{msg}\n\n"
            f"Symbol: {self._symbol}\n"
            f"Direction: {self._direction.upper()}\n"
            f"Entry: ${self._entry:.2f}\n"
            f"Exit SPY ref: ${spy:.2f}\n"
            f"Shares: {self._shares}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _spy_price(self) -> Optional[float]:
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.alpaca.get_live_price("SPY", "buy")
                ),
                timeout=5.0,
            )
        except Exception:
            return None

    async def _live_price(self, symbol: str) -> Optional[float]:
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.alpaca.get_live_price(symbol, "buy")
                ),
                timeout=5.0,
            )
        except Exception:
            return None

    async def _telegram(self, message: str):
        """Publish a Telegram alert to CHANNEL_ALERTS."""
        await self.publish(RedisConfig.CHANNEL_ALERTS, {
            "type":      "telegram_alert",
            "message":   message,
            "source":    "orb_strategy",
            "priority":  "high",
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _daily_reset(self, today: dt_date):
        self._trade_date     = today
        self._state          = _IDLE
        self._high           = 0.0
        self._low            = 0.0
        self._mid            = 0.0
        self._regime         = ""
        self._size_usd       = 0.0
        self._vix            = 0.0
        self._spy_vs_20ma    = ""
        self._breakout_dir   = ""
        self._breakout_price = 0.0
        self._symbol         = ""
        self._direction      = ""
        self._entry          = 0.0
        self._stop           = 0.0
        self._target         = 0.0
        self._shares         = 0
        self._risk_per_share = 0.0
        self.log(f"Daily reset → IDLE [{today}]")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = ORBMidpointBot()
    asyncio.run(bot.start())
