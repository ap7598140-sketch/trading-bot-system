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
from datetime import datetime, timedelta
from typing import Optional

import pytz

import anthropic

from config import Models, RedisConfig, AnthropicConfig, RiskConfig
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

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_ORDERS, self._on_order_event)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._order_monitor())
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

        # Buying power check before placing order
        try:
            loop    = asyncio.get_event_loop()
            account = await loop.run_in_executor(None, self.alpaca.get_account)
            buying_power = float(account.get("buying_power", 0))
            cost_basis   = float(size_usd)
            if buying_power < cost_basis:
                self.log(
                    f"SKIPPED {sym}: insufficient buying power "
                    f"(${buying_power:,.2f} available, ${cost_basis:,.2f} needed)",
                    "warning"
                )
                return
            self.log(f"Buying power OK: ${buying_power:,.2f} available for ${cost_basis:,.2f} order")
        except Exception as e:
            self.log(f"Buying power check failed: {e} — proceeding anyway", "warning")

        # Calculate share quantity
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

            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":      "order_submitted",
                **execution_record,
            })

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
