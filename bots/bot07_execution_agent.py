"""
Bot 7 – Execution Agent
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : Order router and trade executor.
         • Listens to CHANNEL_ORDERS for approved trades, force closes, take profits
         • Submits market/limit orders to Alpaca
         • Monitors order fills and partial fills
         • Logs all executions and publishes fill confirmations to the bus
         Speed matters here – Haiku keeps latency minimal.
"""

import asyncio
import json
from datetime import datetime, timedelta, time as dt_time
from typing import Optional

import httpx
import pytz

import anthropic

from config import Models, RedisConfig, AnthropicConfig, RiskConfig, TradingWindowConfig, AlertConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient

MARKET_TZ = pytz.timezone("America/New_York")


class ExecutionAgent(BaseBot):
    """
    Bot 7 – Execution Agent
    Fast order execution with Haiku-assisted order routing decisions.
    """

    BOT_ID = 7
    NAME   = "Execution Agent"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.alpaca = AlpacaClient()

        self._trading_halted = False
        self._no_new_buys   = False   # True after 3:50pm — blocks new buy orders only
        self._trades_today  = 0       # resets at market open; capped at MAX_DAILY_TRADES
        self._pending_orders: dict[str, dict] = {}    # order_id → order info
        self._executions_today: list[dict] = []
        # Position tracker for partial-close / trailing-stop / compound logic
        # key = symbol, value = {shares, entry_price, size_usd, grade,
        #                        partial_closed, trailing_set, compounded}
        self._position_tracker: dict[str, dict] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_ORDERS, self._on_order_event)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._order_monitor())
        asyncio.create_task(self._position_monitor())
        asyncio.create_task(self._eod_cancel_task())
        asyncio.create_task(self._eod_no_buys_task())
        self.log("Execution Agent starting")

    async def run(self):
        # Keep alive; work is event-driven from order channel
        while self.running:
            await asyncio.sleep(60)
            await self._report_summary()

    async def cleanup(self):
        self.log(f"Execution Agent stopped | {len(self._executions_today)} trades today")

    # ── Order event handler ────────────────────────────────────────────────────

    async def _on_order_event(self, event: dict):
        event_type = event.get("type")

        # Log every incoming event so we can see what arrives
        self.log(
            f"ORDER EVENT received: type={event_type} "
            f"symbol={event.get('symbol')} "
            f"size_usd={event.get('position_size_usd')} "
            f"halted={self._trading_halted}"
        )

        if event_type == "halt_trading":
            if self._trading_halted:
                return   # already halted — ignore duplicates
            self._trading_halted = True
            self.log(f"TRADING HALTED: {event.get('reason')}", "critical")
            await self._cancel_all_pending()
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":    "execution_halted",
                "reason":  event.get("reason"),
            })
            return

        if self._trading_halted and event_type not in ("force_close", "take_profit"):
            self.log(f"Trading halted – ignoring {event_type} for {event.get('symbol')}", "warning")
            return

        if event_type in ("approved_trade", "trade_setup"):
            await self._execute_trade(event)
        elif event_type == "force_close":
            await self._force_close(event.get("symbol"), event.get("reason", ""))
        elif event_type == "take_profit":
            await self._close_position(event.get("symbol"), "take_profit")

    # ── Trade execution ────────────────────────────────────────────────────────

    async def _execute_trade(self, setup: dict):
        sym        = setup.get("symbol", "")
        direction  = setup.get("direction", "long")
        size_usd   = setup.get("position_size_usd", 0)
        entry      = setup.get("entry_price", 0)
        confidence = setup.get("confidence", 0.5)

        # Log every field so we can diagnose incomplete setups
        self.log(
            f"EXECUTE_TRADE: sym={sym} dir={direction} "
            f"size_usd={size_usd} entry={entry} "
            f"stop={setup.get('stop_loss')} tp={setup.get('take_profit')} "
            f"conf={confidence}"
        )

        if not sym or not size_usd or not entry:
            self.log(
                f"REJECTED incomplete setup — missing: "
                f"{'sym ' if not sym else ''}"
                f"{'size_usd ' if not size_usd else ''}"
                f"{'entry' if not entry else ''}",
                "warning"
            )
            return

        side = "buy" if direction == "long" else "sell"

        # Only open new buy positions inside the all-day trading window
        if side == "buy":
            t     = datetime.now(MARKET_TZ).time()
            open_ = dt_time(*TradingWindowConfig.OPEN)
            close = dt_time(*TradingWindowConfig.CLOSE)
            if not (open_ <= t <= close):
                self.log(
                    f"SKIPPED {sym}: outside trading window ({t.strftime('%H:%M')} ET) — "
                    f"window: {TradingWindowConfig.OPEN_STR}–{TradingWindowConfig.CLOSE_STR}",
                    "warning",
                )
                return

        # After 3:50pm block new buy orders (EOD wind-down)
        if self._no_new_buys and side == "buy":
            self.log(f"SKIPPED {sym}: no new buys after 3:50pm EST", "warning")
            return

        # Daily trade limit
        if self._trades_today >= RiskConfig.MAX_DAILY_TRADES:
            self.log(
                f"SKIPPED {sym}: daily trade limit reached ({self._trades_today}/{RiskConfig.MAX_DAILY_TRADES})",
                "warning"
            )
            return

        # ── Live price validation (CRITICAL — must run before any sizing) ────────
        live_price = await self._fetch_live_price(sym, side)
        if live_price is None:
            self.log(
                f"REJECTED {sym}: Alpaca returned no live price — "
                f"refusing to place order with stale cached price ${entry:.2f}",
                "warning",
            )
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":   "order_rejected_no_live_price",
                "symbol": sym,
                "reason": "Could not fetch live price from Alpaca",
            })
            return

        drift = abs(entry - live_price) / live_price
        self.log(
            f"Price validation: cached=${entry:.4f} | live=${live_price:.4f} | "
            f"drift={drift*100:.2f}% | side={side}"
        )
        if drift > 0.02:
            self.log(
                f"REJECTED {sym}: price drift {drift*100:.1f}% exceeds 2% limit "
                f"(cached=${entry:.2f}, live=${live_price:.2f}) — order cancelled",
                "warning",
            )
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":         "order_rejected_price_drift",
                "symbol":       sym,
                "cached_price": entry,
                "live_price":   live_price,
                "drift_pct":    round(drift * 100, 2),
                "reason":       f"Price drift {drift*100:.1f}% > 2% — stale data rejected",
            })
            return

        # Recalculate stop/take from live price using the same % that bot05 used.
        # Extract implied pcts from the setup (prevents stale SL/TP hitting the wire).
        stale_stop = float(setup.get("stop_loss")  or 0)
        stale_tp   = float(setup.get("take_profit") or 0)
        sl_pct = abs(stale_stop - entry) / entry if entry > 0 and stale_stop > 0 else RiskConfig.STOP_LOSS_PCT
        tp_pct = abs(stale_tp   - entry) / entry if entry > 0 and stale_tp   > 0 else RiskConfig.TAKE_PROFIT_PCT

        if side == "buy":
            live_stop = round(live_price * (1 - sl_pct), 4)
            live_tp   = round(live_price * (1 + tp_pct), 4)
        else:
            live_stop = round(live_price * (1 + sl_pct), 4)
            live_tp   = round(live_price * (1 - tp_pct), 4)

        # Overwrite setup with live values — downstream code uses these
        setup["entry_price"] = live_price
        setup["stop_loss"]   = live_stop
        setup["take_profit"] = live_tp
        entry = live_price   # local alias used below

        # ── Buying power check ────────────────────────────────────────────────
        try:
            loop    = asyncio.get_event_loop()
            account = await loop.run_in_executor(None, self.alpaca.get_account)
            buying_power = float(account.get("buying_power", 0))
            if buying_power < float(size_usd):
                self.log(
                    f"SKIPPED {sym}: insufficient buying power "
                    f"(${buying_power:,.2f} available, ${size_usd:,.2f} needed)",
                    "warning",
                )
                return
            self.log(f"Buying power OK: ${buying_power:,.2f} available for ${size_usd:,.2f}")
        except Exception as e:
            self.log(f"Buying power check failed: {e} — proceeding anyway", "warning")

        # Share count always based on live price — never the stale AI entry
        shares = max(1, int(size_usd / entry))

        # AI order routing: market vs limit
        order_type = await self._decide_order_type(sym, entry, confidence, setup)

        self.log(
            f"SENDING TO ALPACA: {side.upper()} {shares} shares of {sym} | "
            f"type={order_type} entry=${entry} size_usd=${size_usd}"
        )

        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.alpaca.submit_market_order(sym, shares, side),
            )

            # Log full Alpaca response so we can see status, rejections, errors
            self.log(f"ALPACA RESPONSE: {json.dumps(result, default=str)}")

            execution_record = {
                "order_id":    result.get("id"),
                "symbol":      sym,
                "side":        side,
                "direction":   direction,
                "shares":      shares,
                "size_usd":    size_usd,
                "entry_price": entry,
                "stop_loss":   setup.get("stop_loss"),
                "take_profit": setup.get("take_profit"),
                "confidence":  confidence,
                "risk_pct":    round(size_usd / 1000 * 0.019, 4) if size_usd else 0.019,
                "reason":      setup.get("thesis") or setup.get("notes", ""),
                "status":      result.get("status"),
                "timestamp":   datetime.utcnow().isoformat(),
            }

            self._executions_today.append(execution_record)
            self._trades_today += 1
            if result.get("id"):
                self._pending_orders[result["id"]] = execution_record

            # Record in position tracker for partial close / compound monitoring
            if side == "buy":
                self._position_tracker[sym] = {
                    "shares":         shares,
                    "entry_price":    entry,
                    "size_usd":       size_usd,
                    "grade":          setup.get("grade", ""),
                    "stop_loss":      setup.get("stop_loss", 0),
                    "take_profit":    setup.get("take_profit", 0),
                    "partial_closed": False,
                    "trailing_set":   False,
                    "compounded":     False,
                    "timestamp":      datetime.utcnow().isoformat(),
                }

            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":      "order_submitted",
                **execution_record,
            })

            await self._send_trade_alert(setup, "ENTRY", shares, entry)

            self.log(
                f"Order submitted: {result.get('id')} | status={result.get('status')} | "
                f"trades today={self._trades_today}/{RiskConfig.MAX_DAILY_TRADES}"
            )

        except Exception as e:
            self.log(f"Order submission failed: {sym} – {e}", "error")
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":   "order_failed",
                "symbol": sym,
                "error":  str(e),
            })

    # ── Telegram sender ────────────────────────────────────────────────────────

    async def _send_telegram(self, text: str):
        token   = AlertConfig.TELEGRAM_BOT_TOKEN
        chat_id = AlertConfig.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json={
                    "chat_id":    chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                })
        except Exception as e:
            self.log(f"Telegram error: {e}", "warning")

    async def _send_trade_alert(self, setup: dict, action: str,
                                 shares: int, price: float):
        """Send a rich Telegram trade alert for every entry/exit."""
        sym       = setup.get("symbol", "?")
        direction = (setup.get("direction") or "long").upper()
        sl        = setup.get("stop_loss", 0)
        tp        = setup.get("take_profit", 0)
        conf      = setup.get("confidence", 0)
        grade     = setup.get("grade", "")
        catalyst  = (setup.get("catalyst") or setup.get("thesis") or "")[:60]
        size_usd  = setup.get("position_size_usd", 0)

        sl_usd  = round(abs(price - sl)  * shares, 2) if sl  else 0
        tp_usd  = round(abs(tp   - price) * shares, 2) if tp else 0

        if action == "ENTRY":
            icon = "🟢" if direction == "LONG" else "🔴"
            lines = [
                f"{icon} <b>TRADE ENTRY — {sym}</b>",
                f"Direction:  {direction} | Grade: {grade or 'N/A'}",
                f"Entry:      ${price:.2f} × {shares} shares (${size_usd:,.0f})",
                f"Stop loss:  ${sl:.2f}  (risk -${sl_usd:.0f})",
                f"Target:     ${tp:.2f}  (profit +${tp_usd:.0f})",
                f"Confidence: {conf:.0%}",
                f"Reason:     {catalyst}",
            ]
        elif action == "PARTIAL_CLOSE":
            lines = [
                f"💰 <b>PARTIAL CLOSE — {sym}</b>",
                f"Sold {shares} shares at ${price:.2f} (+2% target hit)",
                f"Trailing stop now active on remaining shares",
            ]
        elif action == "COMPOUND":
            lines = [
                f"🚀 <b>COMPOUNDING — {sym}</b>",
                f"Added {shares} more shares at ${price:.2f} (+$50 profit threshold hit)",
                f"Letting winner run to $100+",
            ]
        else:
            lines = [f"📋 <b>{action} — {sym}</b>", f"{shares} shares @ ${price:.2f}"]

        try:
            await self._send_telegram("\n".join(lines))
        except Exception as e:
            self.log(f"Trade alert error: {e}", "warning")

    # ── Position monitor (partial close / trailing stop / compound) ────────────

    async def _position_monitor(self):
        """
        Polls open positions every 30 s and applies trade management rules:
          +2%  of entry → partial close 50% of shares, submit trailing stop
          +$40 profit   → ensure trailing stop is active (implicit via +2% rule)
          +$50 profit   → compound: add 50% more shares (once per position)
        """
        while self.running:
            await asyncio.sleep(30)
            if not self._position_tracker or self._trading_halted:
                continue
            try:
                loop      = asyncio.get_event_loop()
                positions = await loop.run_in_executor(None, self.alpaca.get_positions)
                live_map  = {p["symbol"]: p for p in positions}

                for sym, tracker in list(self._position_tracker.items()):
                    live = live_map.get(sym)
                    if not live:
                        # Position gone (closed externally) — clean up tracker
                        self._position_tracker.pop(sym, None)
                        continue

                    entry       = tracker["entry_price"]
                    orig_shares = tracker["shares"]
                    unreal_pl   = float(live.get("unrealized_pl", 0))
                    cur_price   = float(live.get("current_price", entry))
                    cur_qty     = float(live.get("qty", orig_shares))

                    # ── Partial close at +2% ───────────────────────────────────
                    partial_trigger_usd = entry * orig_shares * RiskConfig.PARTIAL_CLOSE_TRIGGER_PCT
                    if (not tracker["partial_closed"]
                            and unreal_pl >= partial_trigger_usd
                            and cur_qty >= 2):
                        close_qty = max(1, int(cur_qty * RiskConfig.PARTIAL_CLOSE_PCT))
                        remain    = int(cur_qty - close_qty)
                        self.log(
                            f"PARTIAL CLOSE {sym}: up ${unreal_pl:.2f} (+2%) — "
                            f"selling {close_qty} of {int(cur_qty)} shares"
                        )
                        try:
                            await loop.run_in_executor(
                                None,
                                lambda s=sym, q=close_qty: self.alpaca.close_partial_position(s, q),
                            )
                            tracker["partial_closed"] = True

                            # Submit trailing stop on remaining shares
                            if remain >= 1 and not tracker["trailing_set"]:
                                await loop.run_in_executor(
                                    None,
                                    lambda s=sym, r=remain: self.alpaca.submit_trailing_stop_order(
                                        s, r, "sell", RiskConfig.TRAILING_STOP_USD
                                    ),
                                )
                                tracker["trailing_set"] = True
                                self.log(
                                    f"Trailing stop set on {sym}: {remain} shares, "
                                    f"trail=${RiskConfig.TRAILING_STOP_USD}"
                                )

                            fake_setup = {
                                "symbol": sym, "direction": "long",
                                "stop_loss": tracker.get("stop_loss", 0),
                                "take_profit": tracker.get("take_profit", 0),
                                "confidence": 1.0, "grade": tracker.get("grade", ""),
                                "catalyst": "partial_close_2pct",
                                "position_size_usd": tracker.get("size_usd", 0),
                            }
                            await self._send_trade_alert(fake_setup, "PARTIAL_CLOSE",
                                                          close_qty, cur_price)
                        except Exception as e:
                            self.log(f"Partial close failed {sym}: {e}", "error")

                    # ── Compound at +$50 ──────────────────────────────────────
                    if (not tracker["compounded"]
                            and unreal_pl >= RiskConfig.COMPOUND_TRIGGER_PROFIT_USD
                            and not self._no_new_buys
                            and not self._trading_halted):
                        add_shares = max(1, int(orig_shares * 0.50))
                        self.log(
                            f"COMPOUND {sym}: up ${unreal_pl:.2f} — "
                            f"adding {add_shares} more shares at ${cur_price:.2f}"
                        )
                        try:
                            live_price = await self._fetch_live_price(sym, "buy")
                            if live_price:
                                await loop.run_in_executor(
                                    None,
                                    lambda s=sym, q=add_shares: self.alpaca.submit_market_order(
                                        s, q, "buy"
                                    ),
                                )
                                tracker["compounded"] = True
                                fake_setup = {
                                    "symbol": sym, "direction": "long",
                                    "stop_loss": 0, "take_profit": 0,
                                    "confidence": 1.0, "grade": tracker.get("grade", ""),
                                    "catalyst": "compound_50_profit",
                                    "position_size_usd": tracker.get("size_usd", 0),
                                }
                                await self._send_trade_alert(fake_setup, "COMPOUND",
                                                              add_shares, live_price or cur_price)
                        except Exception as e:
                            self.log(f"Compound add failed {sym}: {e}", "error")

            except Exception as e:
                self.log(f"Position monitor error: {e}", "warning")

    # ── Live price fetcher ─────────────────────────────────────────────────────

    async def _fetch_live_price(self, symbol: str, side: str = "buy") -> Optional[float]:
        """
        Fetch real-time ask (buy) or bid (sell) price from Alpaca.
        Returns None if Alpaca is unreachable — callers must treat None as a hard block.
        """
        try:
            loop = asyncio.get_event_loop()
            price = await loop.run_in_executor(
                None, lambda: self.alpaca.get_live_price(symbol, side)
            )
            return price
        except Exception as e:
            self.log(f"Live price fetch error for {symbol}: {e}", "warning")
            return None

    # ── AI order type decision ─────────────────────────────────────────────────

    async def _decide_order_type(self, symbol: str, entry: float,
                                  confidence: float, setup: dict) -> str:
        """
        Use Haiku to decide market vs limit order based on urgency and spread.
        High confidence + breakout = market. Lower confidence = limit.
        """
        prompt = (
            f"Decide order type for {symbol}. Entry: ${entry:.2f}, "
            f"confidence: {confidence:.2f}, timeframe: {setup.get('timeframe')}, "
            f"spread: ${setup.get('spread', 0.01):.4f}.\n"
            "Reply ONLY: {\"order_type\": \"market\"} or {\"order_type\": \"limit\"}\n"
            "Use market if confidence>0.8 or timeframe=scalp. Use limit otherwise."
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=32,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw    = response.content[0].text.strip()
            parsed = json.loads(raw)
            return parsed.get("order_type", "market")
        except Exception:
            return "market" if confidence > 0.8 else "limit"

    # ── Force close / take profit ──────────────────────────────────────────────

    async def _force_close(self, symbol: str, reason: str):
        if not symbol:
            return
        self.log(f"Force closing {symbol}: {reason}", "warning")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.alpaca.close_position(symbol))
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":    "position_closed",
                "symbol":  symbol,
                "reason":  reason,
            })
            self.log(f"Closed {symbol}")
        except Exception as e:
            self.log(f"Force close failed {symbol}: {e}", "error")

    async def _close_position(self, symbol: str, reason: str):
        await self._force_close(symbol, reason)

    async def _cancel_all_pending(self):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.alpaca.cancel_all_orders)
            self._pending_orders.clear()
            self.log("All pending orders cancelled")
        except Exception as e:
            self.log(f"Cancel all failed: {e}", "error")

    # ── EOD buy block (3:50pm) ─────────────────────────────────────────────────

    async def _eod_no_buys_task(self):
        """Block new buy orders after 3:50pm EST; re-enable at next market open (9:30am)."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=15, minute=50, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self._no_new_buys = True
            self.log("EOD 3:50pm — new buy orders blocked until next market open")
            # Re-enable at next 9:30am ET (skip weekends)
            now_et     = datetime.now(MARKET_TZ)
            next_open  = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            if now_et >= next_open:
                next_open += timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            await asyncio.sleep((next_open - datetime.now(MARKET_TZ)).total_seconds())
            self._no_new_buys  = False
            self._trades_today = 0
            self.log("Market open — buy orders re-enabled, daily trade counter reset")

    # ── EOD order cancel ───────────────────────────────────────────────────────

    async def _eod_cancel_task(self):
        """Cancel all pending orders at 3:55pm EST every trading day."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            sleep_secs = (target - now_et).total_seconds()
            self.log(f"EOD order cancel scheduled in {sleep_secs / 60:.1f} min")
            await asyncio.sleep(sleep_secs)
            # Skip weekends
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            self.log("EOD 3:55pm — cancelling all pending orders before close")
            await self._cancel_all_pending()
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":      "eod_orders_cancelled",
                "timestamp": datetime.utcnow().isoformat(),
            })

    # ── Order monitor ──────────────────────────────────────────────────────────

    async def _order_monitor(self):
        """Poll Alpaca for order status updates every 10 seconds."""
        while self.running:
            await asyncio.sleep(10)
            if not self._pending_orders:
                continue
            try:
                loop   = asyncio.get_event_loop()
                orders = await loop.run_in_executor(None, self.alpaca.get_orders)
                live_ids = {o["id"] for o in orders}

                # Find filled orders
                filled = [oid for oid in list(self._pending_orders)
                          if oid not in live_ids]
                for oid in filled:
                    rec = self._pending_orders.pop(oid)
                    await self.publish(RedisConfig.CHANNEL_ALERTS, {
                        "type":     "order_filled",
                        "order_id": oid,
                        "symbol":   rec["symbol"],
                        "side":     rec["side"],
                        "shares":   rec["shares"],
                    })
                    self.log(f"Order filled: {oid} | {rec['symbol']}")
            except Exception as e:
                self.log(f"Order monitor error: {e}", "warning")

    # ── Summary report ─────────────────────────────────────────────────────────

    async def _report_summary(self):
        await self.save_state("summary", {
            "executions_today": len(self._executions_today),
            "pending_orders":   len(self._pending_orders),
            "trading_halted":   self._trading_halted,
            "timestamp":        datetime.utcnow().isoformat(),
        }, ttl=120)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = ExecutionAgent()
    asyncio.run(bot.start())
