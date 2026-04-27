"""
Bot 13 – Telegram Controller
Model  : None (no LLM calls — pure control plane)
Role   : Accepts Telegram commands from the authorized operator only.
         Publishes control messages to CHANNEL_CONTROL for Bot 8 to execute.
         Handles read-only queries (/status, /profit, /positions) directly.

Security: All messages from chat IDs other than TELEGRAM_CHAT_ID are silently ignored.
"""

import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import pytz

from alpaca.data.timeframe import TimeFrame

from config import (
    AlertConfig, RedisConfig, AlpacaConfig, BASE_DIR,
    UniverseConfig, TradingWindowConfig, AnthropicConfig,
)
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


POLL_TIMEOUT  = 30   # long-polling timeout (seconds)
POLL_INTERVAL = 1    # sleep between poll attempts on error
LOCK_FILE     = os.path.join(BASE_DIR, "telegram_bot.lock")


class TelegramController(BaseBot):
    """
    Bot 13 – Telegram Controller
    Long-polls Telegram getUpdates, validates sender, dispatches commands.
    """

    BOT_ID = 13
    NAME   = "Telegram Controller"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, model="none")
        self.alpaca          = AlpacaClient()
        self._token          = AlertConfig.TELEGRAM_BOT_TOKEN
        self._authorized_id  = str(AlertConfig.TELEGRAM_CHAT_ID).strip()
        self._update_offset  = 0
        self._paused         = False   # local mirror of pause state

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        if not self._token:
            self.log("TELEGRAM_BOT_TOKEN not set — controller disabled", "warning")
            return
        if not self._authorized_id:
            self.log("TELEGRAM_CHAT_ID not set — controller disabled", "warning")
            return

        # ── Lock file: prevent 409 from two instances polling simultaneously ──
        if os.path.exists(LOCK_FILE):
            try:
                pid = int(open(LOCK_FILE).read().strip())
                # Check if that process is still running
                os.kill(pid, 0)
                self.log(
                    f"Telegram lock file exists (PID {pid} is running) — "
                    "another instance is already polling. This instance will idle.",
                    "warning",
                )
                self._token = ""   # disable polling for this instance
                return
            except (ProcessLookupError, ValueError, OSError):
                # Stale lock — previous process died without cleanup
                self.log("Stale lock file found — removing and taking over", "warning")
                os.remove(LOCK_FILE)

        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
        self.log(f"Telegram lock acquired (PID {os.getpid()}) → {LOCK_FILE}")

        # ── Clear any active webhook before starting long-poll ────────────────
        # A registered webhook causes 409 Conflict on getUpdates.
        await self._delete_webhook()
        await asyncio.sleep(2)

        self.log(
            f"Telegram Controller ready | authorized_chat={self._authorized_id}"
        )

    async def run(self):
        if not self._token or not self._authorized_id:
            while self.running:
                await asyncio.sleep(60)
            return

        async with httpx.AsyncClient(timeout=POLL_TIMEOUT + 5) as client:
            while self.running:
                try:
                    updates = await self._get_updates(client)
                    for update in updates:
                        await self._handle_update(update)
                except httpx.ReadTimeout:
                    pass   # expected — long poll expired, immediately re-poll
                except Exception as e:
                    self.log(f"Poll error: {e}", "warning")
                    await asyncio.sleep(POLL_INTERVAL)

    async def cleanup(self):
        # Release the lock so the next startup can proceed
        try:
            if os.path.exists(LOCK_FILE):
                pid = int(open(LOCK_FILE).read().strip())
                if pid == os.getpid():
                    os.remove(LOCK_FILE)
                    self.log("Telegram lock released")
        except Exception:
            pass
        self.log("Telegram Controller stopped")

    # ── Telegram API helpers ───────────────────────────────────────────────────

    async def _delete_webhook(self):
        """
        Call Telegram deleteWebhook before starting long-poll.
        A registered webhook causes 409 Conflict on every getUpdates call.
        drop_pending_updates=False preserves queued messages.
        """
        url = f"https://api.telegram.org/bot{self._token}/deleteWebhook"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={"drop_pending_updates": False})
                data = resp.json()
                if data.get("ok"):
                    self.log("Webhook deleted — safe to start long-polling")
                else:
                    self.log(f"deleteWebhook response: {data}", "warning")
        except Exception as e:
            self.log(f"deleteWebhook error: {e}", "warning")

    async def _get_updates(self, client: httpx.AsyncClient) -> list[dict]:
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        resp = await client.get(url, params={
            "offset":  self._update_offset,
            "timeout": POLL_TIMEOUT,
        })
        if resp.status_code == 409:
            self.log(
                "409 Conflict on getUpdates — another instance is polling or a webhook "
                "is still active. Calling deleteWebhook and waiting 5 s before retry.",
                "warning",
            )
            await self._delete_webhook()
            await asyncio.sleep(5)
            return []
        if resp.status_code != 200:
            self.log(f"getUpdates error {resp.status_code}", "warning")
            return []
        data = resp.json()
        if not data.get("ok"):
            return []
        updates = data.get("result", [])
        if updates:
            self._update_offset = updates[-1]["update_id"] + 1
        return updates

    async def _send(self, chat_id: str | int, text: str):
        """Send a message to a Telegram chat."""
        if not self._token:
            return
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json={
                    "chat_id":    chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                })
        except Exception as e:
            self.log(f"Telegram send error: {e}", "warning")

    # ── Update dispatch ────────────────────────────────────────────────────────

    async def _handle_update(self, update: dict):
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat_id  = str(message.get("chat", {}).get("id", ""))
        username = message.get("from", {}).get("username", "unknown")
        text     = (message.get("text") or "").strip()

        # Always log received commands (including unauthorized)
        self.log(f"Message from chat={chat_id} user={username}: {text!r}")

        if chat_id != self._authorized_id:
            self.log(f"UNAUTHORIZED: chat_id={chat_id} ignored", "warning")
            return

        if not text.startswith("/"):
            return

        cmd = text.split()[0].lower().lstrip("/")

        try:
            await self._dispatch(cmd, chat_id)
        except Exception as e:
            self.log(f"Command /{cmd} error: {e}", "error")
            await self._send(chat_id, f"❌ Error executing /{cmd}: {e}")

    async def _dispatch(self, cmd: str, chat_id: str):
        handlers = {
            "start":       self._cmd_start,
            "stop":        self._cmd_stop,
            "pause":       self._cmd_pause,
            "resume":      self._cmd_resume,
            "status":      self._cmd_status,
            "profit":      self._cmd_profit,
            "positions":   self._cmd_positions,
            "close":       self._cmd_close,
            "restart":     self._cmd_restart,
            "kill":        self._cmd_kill,
            "backtest":    self._cmd_backtest,
            "mocktest":    self._cmd_mocktest,
            "signaltest":  self._cmd_signaltest,
            "healthcheck": self._cmd_healthcheck,
        }
        handler = handlers.get(cmd)
        if handler:
            await handler(chat_id)
        else:
            await self._send(chat_id,
                "Unknown command. Available:\n"
                "/start /stop /pause /resume /status /profit /positions /close /restart /kill\n"
                "/backtest /mocktest /signaltest /healthcheck"
            )

    # ── Command handlers ───────────────────────────────────────────────────────

    async def _cmd_start(self, chat_id: str):
        """Resume trading — publishes resume_trading to CHANNEL_CONTROL."""
        self._paused = False
        await self.publish(RedisConfig.CHANNEL_CONTROL, {
            "type":      "resume_trading",
            "source":    "telegram",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.log("Operator command: /start → resume_trading published")
        await self._send(chat_id,
            "▶️ <b>Trading RESUMED</b>\n"
            "Bot 8 will re-open the market gate and allow new trades."
        )

    async def _cmd_stop(self, chat_id: str):
        """Close all positions then halt trading."""
        await self._send(chat_id,
            "🛑 <b>STOPPING — closing all positions now...</b>"
        )
        # First: close all positions
        loop = asyncio.get_event_loop()
        try:
            positions = await loop.run_in_executor(None, self.alpaca.get_positions)
            closed = []
            total_pnl = 0.0
            for pos in positions:
                sym = pos["symbol"]
                pnl = float(pos.get("unrealized_pl", 0))
                try:
                    await loop.run_in_executor(None, lambda s=sym: self.alpaca.close_position(s))
                    closed.append(sym)
                    total_pnl += pnl
                except Exception as e:
                    self.log(f"/stop close {sym} failed: {e}", "warning")
        except Exception as e:
            self.log(f"/stop get_positions error: {e}", "warning")
            closed = []
            total_pnl = 0.0

        # Then: halt new trades
        await self.publish(RedisConfig.CHANNEL_CONTROL, {
            "type":      "halt_trading",
            "source":    "telegram",
            "reason":    "Operator /stop command",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.log("Operator command: /stop → positions closed + halt_trading published")

        pnl_sign = "+" if total_pnl >= 0 else ""
        summary  = f"Closed {len(closed)} positions: {', '.join(closed) or 'none'}\n" \
                   f"P/L: {pnl_sign}${total_pnl:.2f}" if closed \
                   else "No open positions to close."
        await self._send(chat_id,
            f"🛑 <b>Trading STOPPED</b>\n{summary}\n\n"
            f"Use /start to resume trading."
        )

    async def _cmd_pause(self, chat_id: str):
        """Halt new trades without closing existing positions."""
        self._paused = True
        await self.publish(RedisConfig.CHANNEL_CONTROL, {
            "type":      "halt_new_trades",
            "source":    "telegram",
            "reason":    "Operator /pause command",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.log("Operator command: /pause → halt_new_trades published")
        await self._send(chat_id,
            "⏸ <b>Trading PAUSED</b>\n"
            "No new positions will be opened.\n"
            "Existing positions remain open.\n\n"
            "Use /resume to continue trading."
        )

    async def _cmd_resume(self, chat_id: str):
        """Resume trading after /pause."""
        self._paused = False
        await self.publish(RedisConfig.CHANNEL_CONTROL, {
            "type":      "resume_trading",
            "source":    "telegram",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.log("Operator command: /resume → resume_trading published")
        await self._send(chat_id,
            "▶️ <b>Trading RESUMED</b>\n"
            "New positions will be accepted again."
        )

    async def _cmd_status(self, chat_id: str):
        """Fetch system status from Bot 8 dashboard state in Redis."""
        try:
            dash = await self.bus.get_state("bot8:dashboard") or {}
        except Exception as e:
            await self._send(chat_id, f"❌ Could not read dashboard: {e}")
            return

        if not dash:
            await self._send(chat_id,
                "⚠️ No dashboard data yet — Bot 8 may still be initialising."
            )
            return

        session  = dash.get("market_session", "unknown")
        halted   = dash.get("system_halted", False)
        pv       = dash.get("portfolio_value", 0)
        pnl      = dash.get("daily_pnl", 0)
        pnl_pct  = dash.get("daily_pnl_pct", 0)
        regime   = dash.get("regime", {})
        cb       = dash.get("circuit_breaker", {})
        bhealth  = dash.get("bot_health", {})

        alive = sum(1 for s in bhealth.values() if s == "alive")
        dead  = [n for n, s in bhealth.items() if s == "dead"]

        pnl_sign = "+" if pnl >= 0 else ""
        status_icon = "🔴" if halted else ("⏸" if self._paused else "🟢")

        lines = [
            f"{status_icon} <b>System Status</b>",
            f"Session:   {session}",
            f"Trading:   {'HALTED' if halted else ('PAUSED' if self._paused else 'ACTIVE')}",
            f"Portfolio: ${pv:,.2f}",
            f"Day P/L:   {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.2f}%)",
            f"Regime:    {regime.get('name','?')} (scale={regime.get('scale',1.0):.2f})",
            f"Circuit:   L{cb.get('level',0)} — {cb.get('action','continue')}",
            f"Bots alive: {alive}/{len(bhealth)}",
        ]
        if dead:
            lines.append(f"⚠️ Dead bots: {', '.join(dead)}")

        ts = dash.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                lines.append(f"Updated: {dt.strftime('%H:%M:%S UTC')}")
            except Exception:
                pass

        await self._send(chat_id, "\n".join(lines))

    async def _cmd_profit(self, chat_id: str):
        """Fetch account equity and daily P/L from Alpaca + Redis."""
        loop = asyncio.get_event_loop()
        try:
            account = await loop.run_in_executor(None, self.alpaca.get_account)
        except Exception as e:
            await self._send(chat_id, f"❌ Alpaca account error: {e}")
            return

        pv        = float(account.get("portfolio_value", 0))
        equity    = float(account.get("equity", 0))
        last_eq   = float(account.get("last_equity", equity))
        day_pnl   = equity - last_eq
        day_pct   = (day_pnl / last_eq * 100) if last_eq else 0
        cash      = float(account.get("cash", 0))
        buying_pw = float(account.get("buying_power", 0))

        sign = "+" if day_pnl >= 0 else ""
        icon = "📈" if day_pnl >= 0 else "📉"

        await self._send(chat_id,
            f"{icon} <b>Profit Report</b>\n"
            f"Portfolio value: ${pv:,.2f}\n"
            f"Day P/L:        {sign}${day_pnl:.2f} ({sign}{day_pct:.2f}%)\n"
            f"Cash:           ${cash:,.2f}\n"
            f"Buying power:   ${buying_pw:,.2f}"
        )

    async def _cmd_positions(self, chat_id: str):
        """List all open positions from Alpaca."""
        loop = asyncio.get_event_loop()
        try:
            positions = await loop.run_in_executor(None, self.alpaca.get_positions)
        except Exception as e:
            await self._send(chat_id, f"❌ Alpaca positions error: {e}")
            return

        if not positions:
            await self._send(chat_id, "📋 <b>No open positions</b>")
            return

        lines = [f"📋 <b>Open Positions ({len(positions)})</b>"]
        total_pnl = 0.0
        for pos in positions:
            sym  = pos["symbol"]
            qty  = pos.get("qty", "?")
            side = pos.get("side", "long")
            pnl  = float(pos.get("unrealized_pl", 0))
            pct  = float(pos.get("unrealized_plpc", 0)) * 100
            mv   = float(pos.get("market_value", 0))
            sign = "+" if pnl >= 0 else ""
            total_pnl += pnl
            lines.append(
                f"  {sym} {side} {qty} | ${mv:,.2f} | {sign}${pnl:.2f} ({sign}{pct:.1f}%)"
            )

        tot_sign = "+" if total_pnl >= 0 else ""
        lines.append(f"\n<b>Total unrealised P/L: {tot_sign}${total_pnl:.2f}</b>")
        await self._send(chat_id, "\n".join(lines))

    async def _cmd_close(self, chat_id: str):
        """Close all open positions immediately."""
        await self._send(chat_id, "🔄 <b>Closing all positions...</b>")
        loop = asyncio.get_event_loop()
        try:
            positions = await loop.run_in_executor(None, self.alpaca.get_positions)
        except Exception as e:
            await self._send(chat_id, f"❌ Could not fetch positions: {e}")
            return

        if not positions:
            await self._send(chat_id, "✅ No open positions to close.")
            return

        closed, failed = [], []
        total_pnl = 0.0
        for pos in positions:
            sym = pos["symbol"]
            pnl = float(pos.get("unrealized_pl", 0))
            try:
                await loop.run_in_executor(None, lambda s=sym: self.alpaca.close_position(s))
                closed.append({"symbol": sym, "pnl": pnl})
                total_pnl += pnl
            except Exception as e:
                failed.append(sym)
                self.log(f"/close failed for {sym}: {e}", "warning")

        # Publish close_all event so Bot 8 is aware
        await self.publish(RedisConfig.CHANNEL_CONTROL, {
            "type":      "close_all",
            "source":    "telegram",
            "reason":    "Operator /close command",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.log(f"Operator command: /close → closed {len(closed)}, failed {len(failed)}")

        tot_sign = "+" if total_pnl >= 0 else ""
        lines = [f"✅ <b>Closed {len(closed)} positions</b>"]
        for c in closed:
            sign = "+" if c["pnl"] >= 0 else ""
            lines.append(f"  {c['symbol']}: {sign}${c['pnl']:.2f}")
        if failed:
            lines.append(f"❌ Failed to close: {', '.join(failed)}")
        lines.append(f"\n<b>Total P/L: {tot_sign}${total_pnl:.2f}</b>")
        await self._send(chat_id, "\n".join(lines))

    async def _cmd_restart(self, chat_id: str):
        """Restart the entire trading system process."""
        await self._send(chat_id,
            "🔁 <b>Restarting system in 3 seconds...</b>\n"
            "The bot will reconnect automatically."
        )
        self.log("Operator command: /restart — restarting process", "warning")
        await asyncio.sleep(3)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    async def _cmd_kill(self, chat_id: str):
        """Terminate the entire trading system process."""
        await self._send(chat_id,
            "💀 <b>System KILL initiated</b>\n"
            "The trading system process will terminate now."
        )
        self.log("Operator command: /kill — sending SIGTERM", "critical")
        await asyncio.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    # ── Simulation / diagnostic commands ──────────────────────────────────────

    async def _cmd_backtest(self, chat_id: str):
        """Run a backtest on 90 days of historical daily bars for the top 10 watchlist symbols."""
        await self._send(chat_id,
            "🔄 <b>Running backtest — last 90 days...</b>\n"
            "Fetching daily bars for top 10 symbols."
        )
        try:
            loop    = asyncio.get_event_loop()
            symbols = UniverseConfig.WATCHLIST[:10]
            start   = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")

            bars = await loop.run_in_executor(
                None, lambda: self.alpaca.get_bars(symbols, TimeFrame.Day, start)
            )
            if not bars:
                await self._send(chat_id, "❌ No historical data returned from Alpaca.")
                return

            strategy = {
                "entry_rules":      ["rsi_14 > 50", "rsi_14 < 70", "macd_histogram > 0"],
                "exit_rules":       ["rsi_14 > 75", "macd_histogram < 0"],
                "stop_loss_pct":    0.015,
                "take_profit_pct":  0.03,
                "position_size_pct": 0.10,
            }

            sym_pnl: dict[str, float] = {}
            all_trades: list[dict]    = []
            for sym, bar_list in bars.items():
                if len(bar_list) < 30:
                    continue
                df     = self._bt_make_df(bar_list)
                df     = self._bt_add_indicators(df)
                trades = self._bt_simulate(df, strategy)
                for t in trades:
                    t["symbol"] = sym
                all_trades.extend(trades)
                sym_pnl[sym] = sum(t.get("pnl", 0) for t in trades if "pnl" in t)

            completed = [t for t in all_trades if "exit_date" in t and "pnl" in t]
            if not completed:
                await self._send(chat_id,
                    "⚠️ Backtest complete: no completed trades "
                    "(market may need more bars for indicator warmup)."
                )
                return

            total_pnl = sum(t["pnl"] for t in completed)
            pnl_pcts  = [t["pnl_pct"] for t in completed]
            win_rate  = sum(1 for p in pnl_pcts if p > 0) / len(pnl_pcts) * 100
            best      = max(completed, key=lambda t: t["pnl"])
            worst     = min(completed, key=lambda t: t["pnl"])
            best_sym  = max(sym_pnl, key=sym_pnl.get) if sym_pnl else "?"
            sign      = "+" if total_pnl >= 0 else ""

            lines = [
                "📊 <b>Backtest Results — Last 90 Days</b>",
                f"Strategy: momentum + RSI + MACD  |  Symbols: {len(sym_pnl)}",
                "",
                f"Trades:      {len(completed)}",
                f"Win rate:    {win_rate:.1f}%",
                f"Total P&amp;L: {sign}${total_pnl:.2f}",
                "",
                f"Best:  {best['symbol']}  +${best['pnl']:.2f}  "
                f"({best.get('entry_date','?')} → {best.get('exit_date','?')})",
                f"Worst: {worst['symbol']}  ${worst['pnl']:.2f}  "
                f"({worst.get('entry_date','?')} → {worst.get('exit_date','?')})",
                f"Best stock:  {best_sym}  (${sym_pnl.get(best_sym, 0):.2f})",
            ]
            await self._send(chat_id, "\n".join(lines))

        except Exception as e:
            self.log(f"/backtest error: {e}", "error")
            await self._send(chat_id, f"❌ Backtest failed: {e}")

    async def _cmd_mocktest(self, chat_id: str):
        """
        Simulate last 5 trading days across all 55 watchlist symbols.

        Five filters applied in sequence:
          1. SPY direction (BULL/BEAR/NEUTRAL) — controls which direction to trade
          2. Momentum threshold — 2.0% open→close (grade A ≥ 3% for neutral days)
          3. Volume confirmation — today's volume must be ≥ average daily volume
          4. Sector alignment — stock direction must match its sector ETF
          5. Intraday stop-loss — if low dipped ≤ −1.5% from open, bot was stopped out
        """
        await self._send(chat_id,
            "🎭 <b>Running mock market simulation (v2)...</b>\n"
            "55 symbols · 5 days · direction + volume + sector filters"
        )
        try:
            loop = asyncio.get_event_loop()

            # All 55 symbols + XLY (needed for sector alignment but not in WATCHLIST_ETFS)
            symbols = list(dict.fromkeys(
                UniverseConfig.WATCHLIST + UniverseConfig.WATCHLIST_ETFS + ["XLY"]
            ))

            start = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
            await self._send(chat_id, f"📥 Fetching daily bars for {len(symbols)} symbols…")
            bars = await loop.run_in_executor(
                None, lambda: self.alpaca.get_bars(symbols, TimeFrame.Day, start)
            )
            if not bars:
                await self._send(chat_id, "❌ No historical data returned from Alpaca.")
                return

            # ── Find 5 most recent complete trading dates ──────────────────────
            all_dates: set[str] = set()
            for bl in bars.values():
                for b in bl:
                    d = (b.get("timestamp") or "")[:10]
                    if d:
                        all_dates.add(d)
            sorted_dates = sorted(all_dates)[-5:]
            if not sorted_dates:
                await self._send(chat_id, "⚠️ No recent trading data found.")
                return

            # ── Build lookup tables ────────────────────────────────────────────
            sym_date_bar: dict[str, dict[str, dict]] = {}
            sym_avg_vol:  dict[str, float]           = {}
            for sym, bl in bars.items():
                date_map: dict[str, dict] = {}
                vols: list[float] = []
                for b in bl:
                    d = (b.get("timestamp") or "")[:10]
                    date_map[d] = b
                    v = float(b.get("volume", 0) or 0)
                    if v > 0:
                        vols.append(v)
                sym_date_bar[sym] = date_map
                sym_avg_vol[sym]  = sum(vols) / len(vols) if vols else 0.0

            # ── Sector reverse map: stock → sector ETF ─────────────────────────
            stock_sector: dict[str, str] = {
                stock: etf
                for etf, members in UniverseConfig.SECTOR_ETFS.items()
                for stock in members
            }

            # ── Simulation constants ───────────────────────────────────────────
            POSITION_USD     = 2000.0
            MOMENTUM_PCT      = 2.0   # min move on NEUTRAL / BEAR days
            BULL_MOMENTUM_PCT = 1.8   # lower threshold on confirmed BULL days
            GRADE_A_PCT      = 3.0   # grade A = strong enough for neutral days
            BULL_BEAR_THRESH = 0.3   # SPY ±0.3% separates BULL / BEAR / NEUTRAL
            TAKE_PROFIT_PCT  = 3.0   # bot exits at +3%
            STOP_LOSS_PCT    = 1.5   # stop exit at −1.5% from open
            INVERSE_ETFS     = {"SQQQ", "SPXS", "SOXS"}
            REGIME_ICON      = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}

            day_results: dict[str, dict] = {
                d: {
                    "trades": 0, "wins": 0, "losses": 0,
                    "pnl": 0.0, "skipped": 0, "regime": "",
                    "best_sym": "", "best_pct": 0.0,
                }
                for d in sorted_dates
            }
            sym_pnl:    dict[str, float]             = {}
            top_movers: list[tuple[str, str, float]] = []

            for date in sorted_dates:
                # ── Filter 1: market direction via SPY ────────────────────────
                spy   = sym_date_bar.get("SPY", {}).get(date, {})
                spy_o = float(spy.get("open",  0) or 0)
                spy_c = float(spy.get("close", 0) or 0)
                spy_pct = (spy_c - spy_o) / spy_o * 100 if spy_o > 0 else 0.0

                if   spy_pct >  BULL_BEAR_THRESH: regime = "BULL"
                elif spy_pct < -BULL_BEAR_THRESH: regime = "BEAR"
                else:                              regime = "NEUTRAL"
                day_results[date]["regime"] = regime

                # Per-sector direction for this date
                sector_pct: dict[str, float] = {}
                for etf in UniverseConfig.SECTOR_ETFS:
                    eb = sym_date_bar.get(etf, {}).get(date, {})
                    eo = float(eb.get("open",  0) or 0)
                    ec = float(eb.get("close", 0) or 0)
                    sector_pct[etf] = (ec - eo) / eo * 100 if eo > 0 else 0.0

                for sym in symbols:
                    bar = sym_date_bar.get(sym, {}).get(date)
                    if not bar:
                        continue
                    open_p  = float(bar.get("open",  0) or 0)
                    close_p = float(bar.get("close", 0) or 0)
                    high_p  = float(bar.get("high",  0) or 0)
                    low_p   = float(bar.get("low",   0) or 0)
                    volume  = float(bar.get("volume", 0) or 0)
                    if open_p <= 0:
                        continue

                    daily_pct  = (close_p - open_p) / open_p * 100
                    intra_low  = (low_p   - open_p) / open_p * 100
                    intra_high = (high_p  - open_p) / open_p * 100
                    is_inverse = sym in INVERSE_ETFS
                    dr         = day_results[date]

                    # Filter 1 cont.: regime-based direction gate
                    if regime == "BULL":
                        if is_inverse or daily_pct < 0:
                            dr["skipped"] += 1
                            continue
                    elif regime == "BEAR":
                        if not is_inverse or daily_pct < 0:
                            dr["skipped"] += 1
                            continue
                    else:  # NEUTRAL: grade A positive moves only
                        if daily_pct < GRADE_A_PCT:
                            dr["skipped"] += 1
                            continue

                    # Filter 2: momentum threshold (relaxed to 1.8% on bull days)
                    threshold = BULL_MOMENTUM_PCT if regime == "BULL" else MOMENTUM_PCT
                    if daily_pct < threshold:
                        dr["skipped"] += 1
                        continue

                    # Filter 3: volume must be at or above average
                    avg_vol = sym_avg_vol.get(sym, 0)
                    if avg_vol > 0 and volume < avg_vol:
                        dr["skipped"] += 1
                        continue

                    # Filter 4: sector alignment (skip if sector moves opposite)
                    sec_etf = stock_sector.get(sym)
                    if sec_etf:
                        sec_dir = sector_pct.get(sec_etf, 0)
                        if (daily_pct > 0 and sec_dir < 0) or (daily_pct < 0 and sec_dir > 0):
                            dr["skipped"] += 1
                            continue

                    # Filter 5 + trade outcome: intraday stop-loss check
                    # If low dipped past stop before recovering → bot was filled at −1.5%
                    if intra_low <= -STOP_LOSS_PCT:
                        loss = round(POSITION_USD * STOP_LOSS_PCT / 100, 2)
                        sym_pnl[sym] = sym_pnl.get(sym, 0.0) - loss
                        dr["trades"] += 1
                        dr["losses"] += 1
                        dr["pnl"]    -= loss
                    else:
                        # Clean win: exit at take-profit if high reached it, else close
                        profit_pct = TAKE_PROFIT_PCT if intra_high >= TAKE_PROFIT_PCT else daily_pct
                        profit     = round(POSITION_USD * profit_pct / 100, 2)
                        sym_pnl[sym] = sym_pnl.get(sym, 0.0) + profit
                        dr["trades"] += 1
                        dr["wins"]   += 1
                        dr["pnl"]    += profit
                        if daily_pct > dr["best_pct"]:
                            dr["best_sym"] = sym
                            dr["best_pct"] = daily_pct
                        top_movers.append((sym, date, daily_pct))

            # ── Aggregate ──────────────────────────────────────────────────────
            total_trades = sum(d["trades"]  for d in day_results.values())
            total_wins   = sum(d["wins"]    for d in day_results.values())
            total_losses = sum(d["losses"]  for d in day_results.values())
            total_pnl    = sum(d["pnl"]     for d in day_results.values())
            total_skip   = sum(d["skipped"] for d in day_results.values())
            win_rate     = total_wins / total_trades * 100 if total_trades else 0.0

            top_movers.sort(key=lambda x: x[2], reverse=True)
            top5      = top_movers[:5]
            best_syms = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
            sign      = "+" if total_pnl >= 0 else ""

            # ── Report ─────────────────────────────────────────────────────────
            lines = [
                "🎭 <b>Mock Market Simulation v2 — Last 5 Trading Days</b>",
                f"Symbols: {len(bars)} | $2,000/trade | SPY-direction · 2% threshold · volume · sector",
                "",
                "📅 <b>Day-by-day results:</b>",
            ]
            for date in sorted_dates:
                dr   = day_results[date]
                ds   = "+" if dr["pnl"] >= 0 else ""
                icon = "🟢" if dr["pnl"] >= 0 else "🔴"
                ri   = REGIME_ICON.get(dr["regime"], "⬜")
                best = f"  (best: {dr['best_sym']} +{dr['best_pct']:.1f}%)" if dr["best_sym"] else ""
                lines.append(
                    f"{icon} {date} [{ri} {dr['regime']}]:  "
                    f"{dr['trades']} trades, {dr['wins']}W/{dr['losses']}L, "
                    f"{ds}${dr['pnl']:.0f}{best}"
                )

            lines += [
                "",
                "📊 <b>Overall summary:</b>",
                f"Total trades:    {total_trades}  ({total_wins}W / {total_losses}L)",
                f"Win rate:        {win_rate:.1f}%",
                f"Signals skipped: {total_skip}  (failed direction/volume/sector filters)",
                f"Estimated P&amp;L: {sign}${total_pnl:.2f}",
            ]

            if top5:
                lines += ["", "🚀 <b>Top momentum moves caught:</b>"]
                for sym, date, pct in top5:
                    profit = POSITION_USD * min(pct, TAKE_PROFIT_PCT) / 100
                    lines.append(f"  {sym} ({date}): +{pct:.1f}%  →  +${profit:.0f}")

            if best_syms:
                lines += ["", "🏆 <b>Best symbols this week:</b>"]
                for sym, pnl in best_syms:
                    s = "+" if pnl >= 0 else ""
                    lines.append(f"  {sym}: {s}${pnl:.2f}")

            lines.append("\n✅ Simulation complete — no real orders placed")
            await self._send(chat_id, "\n".join(lines))

        except Exception as e:
            self.log(f"/mocktest error: {e}", "error")
            await self._send(chat_id, f"❌ Mock test failed: {e}")

    async def _cmd_signaltest(self, chat_id: str):
        """Verify that all five signal pipelines are actively flowing."""
        await self._send(chat_id, "🔍 <b>Testing all signal pipelines...</b>")
        results: list[str] = []
        loop = asyncio.get_event_loop()

        # 1. Momentum signals (bot03)
        try:
            mom = await self.bus.get_state("bot3:leaderboard") or {}
            if mom:
                leaders = mom.get("top_momentum") or mom.get("leaders") or []
                count   = len(leaders) if isinstance(leaders, list) else len(mom)
                results.append(f"✅ Momentum: {count} signals in leaderboard")
            else:
                results.append("⚠️ Momentum: no leaderboard data (bot03 not scanned yet?)")
        except Exception as e:
            results.append(f"❌ Momentum: {e}")

        # 2. News sentiment (bot01)
        try:
            news = await self.bus.get_state("bot1:latest_summary") or {}
            if news:
                ts = news.get("timestamp", "")
                results.append(f"✅ News: summary available (ts={ts[:16] if ts else '?'})")
            else:
                results.append("⚠️ News: no summary yet (bot01 loading or market closed?)")
        except Exception as e:
            results.append(f"❌ News: {e}")

        # 3. Strategy setups (bot05)
        try:
            strat  = await self.bus.get_state("bot5:latest") or {}
            if strat:
                setups = strat.get("setups") or strat.get("trade_setups") or []
                count  = len(setups) if isinstance(setups, list) else "n/a"
                results.append(f"✅ Strategy: {count} setups generated")
            else:
                results.append("⚠️ Strategy: no setups yet (bot05 idle or market closed?)")
        except Exception as e:
            results.append(f"❌ Strategy: {e}")

        # 4. Execution agent ready (bot07)
        try:
            ex = await self.bus.get_state("bot7:summary") or {}
            if ex:
                trades_today = ex.get("trades_today", "?")
                results.append(f"✅ Execution: ready ({trades_today} trades today)")
            else:
                results.append("⚠️ Execution: no status (bot07 may be idle)")
        except Exception as e:
            results.append(f"❌ Execution: {e}")

        # 5. Live price check — confirm prices are real, not stale cached values
        try:
            price = await loop.run_in_executor(
                None, lambda: self.alpaca.get_live_price("SPY", "buy")
            )
            if price and 50 < price < 10000:
                results.append(f"✅ Live prices: SPY=${price:.2f} (real-time from Alpaca)")
            elif price:
                results.append(f"⚠️ Live prices: SPY=${price:.2f} (unusual range — verify)")
            else:
                results.append("❌ Live prices: SPY unavailable (market closed or API issue?)")
        except Exception as e:
            results.append(f"❌ Live prices: {e}")

        all_ok = all(r.startswith("✅") for r in results)
        icon   = "✅" if all_ok else "⚠️"
        lines  = [f"{icon} <b>Signal Test Results</b>", ""] + results
        await self._send(chat_id, "\n".join(lines))

    async def _cmd_healthcheck(self, chat_id: str):
        """Full 10-point system diagnostic."""
        await self._send(chat_id, "🏥 <b>Running full health check (10 checks)...</b>")
        checks: list[tuple[str, str]] = []
        loop = asyncio.get_event_loop()

        # 1. Alpaca connection
        try:
            acct = await loop.run_in_executor(None, self.alpaca.get_account)
            pv   = acct.get("portfolio_value", 0)
            checks.append(("Alpaca API", f"✅  ${pv:,.0f} portfolio value"))
        except Exception as e:
            checks.append(("Alpaca API", f"❌  {str(e)[:60]}"))

        # 2. Anthropic API key configured
        key = AnthropicConfig.API_KEY
        if key and len(key) > 10:
            checks.append(("Anthropic API", f"✅  key configured ({key[:4]}...{key[-4:]})"))
        else:
            checks.append(("Anthropic API", "❌  ANTHROPIC_API_KEY not set"))

        # 3. Redis connection
        try:
            await self.bus.get_state("_healthcheck_probe")
            checks.append(("Redis", "✅  connected"))
        except Exception as e:
            checks.append(("Redis", f"❌  {e}"))

        # Load dashboard once for checks 4, 9, 10
        dash: dict = {}
        try:
            dash = await self.bus.get_state("bot8:dashboard") or {}
        except Exception:
            pass

        # 4. All bots alive
        bhealth = dash.get("bot_health", {})
        if bhealth:
            alive = sum(1 for s in bhealth.values() if s == "alive")
            dead  = [n for n, s in bhealth.items() if s == "dead"]
            total = len(bhealth)
            if dead:
                checks.append(("Bots alive", f"⚠️  {alive}/{total} — dead: {', '.join(dead)}"))
            else:
                checks.append(("Bots alive", f"✅  {alive}/{total} all alive"))
        else:
            checks.append(("Bots alive", "⚠️  dashboard unavailable (bot08 not running?)"))

        # 5. Watchlist size
        wl_size = len(UniverseConfig.WATCHLIST) + len(UniverseConfig.WATCHLIST_ETFS)
        wl_icon = "✅" if wl_size >= 50 else "⚠️"
        checks.append(("Watchlist size", (
            f"{wl_icon}  {wl_size} symbols "
            f"({len(UniverseConfig.WATCHLIST)} stocks + {len(UniverseConfig.WATCHLIST_ETFS)} ETFs)"
        )))

        # 6. Last momentum scan time
        try:
            mom = await self.bus.get_state("bot3:leaderboard") or {}
            ts  = mom.get("timestamp", "")
            if ts:
                age  = (datetime.now(timezone.utc) -
                        datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 60
                icon = "✅" if age < 10 else "⚠️"
                checks.append(("Momentum scan", f"{icon}  {age:.0f} min ago"))
            else:
                checks.append(("Momentum scan", "⚠️  no scan data (bot03 not running?)"))
        except Exception:
            checks.append(("Momentum scan", "⚠️  state unavailable"))

        # 7. Last news scan time
        try:
            news = await self.bus.get_state("bot1:latest_summary") or {}
            ts   = news.get("timestamp", "")
            if ts:
                age  = (datetime.now(timezone.utc) -
                        datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 60
                icon = "✅" if age < 60 else "⚠️"
                checks.append(("News scan", f"{icon}  {age:.0f} min ago"))
            else:
                checks.append(("News scan", "⚠️  no news data (bot01 not running?)"))
        except Exception:
            checks.append(("News scan", "⚠️  state unavailable"))

        # 8. Trading window status
        et        = pytz.timezone(TradingWindowConfig.TIMEZONE)
        now_et    = datetime.now(et)
        t         = now_et.time()
        open_t    = dt_time(*TradingWindowConfig.OPEN)
        close_t   = dt_time(*TradingWindowConfig.CLOSE)
        in_window = open_t <= t <= close_t
        day_str   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][now_et.weekday()]
        if now_et.weekday() >= 5:
            checks.append(("Trading window", f"🔴  Weekend ({day_str}) — market closed"))
        elif in_window:
            checks.append(("Trading window",
                f"✅  OPEN ({now_et.strftime('%H:%M')} ET, {day_str})"))
        else:
            checks.append(("Trading window", (
                f"🔴  CLOSED ({now_et.strftime('%H:%M')} ET — "
                f"window {TradingWindowConfig.OPEN_STR}–{TradingWindowConfig.CLOSE_STR})"
            )))

        # 9. Current market regime
        regime = dash.get("regime") or {}
        if not regime:
            try:
                regime = await self.bus.get_state("bot8:regime") or {}
            except Exception:
                pass
        if regime:
            name  = regime.get("name", "?")
            scale = regime.get("scale", 1.0)
            checks.append(("Regime", f"✅  {name} (position scale={scale:.2f})"))
        else:
            checks.append(("Regime", "⚠️  no regime data"))

        # 10. Circuit breaker status
        cb = dash.get("circuit_breaker") or {}
        if not cb:
            try:
                cb = await self.bus.get_state("bot8:cb_state") or {}
            except Exception:
                pass
        if cb:
            level  = cb.get("level", 0)
            action = cb.get("action", "continue")
            icon   = "✅" if level == 0 else ("⚠️" if level < 3 else "🔴")
            checks.append(("Circuit breaker", f"{icon}  Level {level} — {action}"))
        else:
            checks.append(("Circuit breaker", "⚠️  no CB state"))

        # Build report
        lines = ["🏥 <b>Health Check Report</b>", ""]
        for label, status in checks:
            lines.append(f"<b>{label}:</b>  {status}")
        pass_count = sum(1 for _, s in checks if s.startswith("✅"))
        lines += ["", f"<b>Overall: {pass_count}/{len(checks)} checks passed</b>"]
        await self._send(chat_id, "\n".join(lines))

    # ── Backtest simulation helpers ────────────────────────────────────────────

    @staticmethod
    def _bt_make_df(bar_list: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(bar_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _bt_add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi_14"]         = 100 - (100 / (1 + rs))
        ema12                = df["close"].ewm(span=12, adjust=False).mean()
        ema26                = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]           = ema12 - ema26
        df["macd_signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        df["sma_20"]         = df["close"].rolling(20).mean()
        return df.dropna(subset=["rsi_14", "macd_histogram"])

    @staticmethod
    def _bt_eval_rule(row, rule: str) -> bool:
        parts = rule.strip().split()
        if len(parts) != 3:
            return False
        col, op, val = parts
        lhs = row.get(col)
        if lhs is None:
            return False
        rhs = float(val)
        lhs = float(lhs)
        if op == ">":  return lhs > rhs
        if op == "<":  return lhs < rhs
        if op == ">=": return lhs >= rhs
        if op == "<=": return lhs <= rhs
        return False

    def _bt_simulate(self, df: pd.DataFrame, strategy: dict,
                     initial_capital: float = 10000.0) -> list[dict]:
        stop_pct    = strategy.get("stop_loss_pct", 0.015)
        target_pct  = strategy.get("take_profit_pct", 0.03)
        size_pct    = strategy.get("position_size_pct", 0.10)
        entry_rules = strategy.get("entry_rules", [])
        exit_rules  = strategy.get("exit_rules", [])
        SLIP        = 0.0005   # 5 bps slippage

        capital  = initial_capital
        position = 0
        entry_px = 0.0
        trades: list[dict] = []
        in_trade = False

        rows = list(df.iterrows())
        for i, (ts, row) in enumerate(rows):
            price = float(row["close"])
            if not in_trade:
                if entry_rules and all(self._bt_eval_rule(row, r) for r in entry_rules):
                    shares  = max(1, int(capital * size_pct / price))
                    exec_px = price * (1 + SLIP)
                    cost    = shares * exec_px
                    if cost <= capital:
                        capital   -= cost
                        position   = shares
                        entry_px   = exec_px
                        in_trade   = True
                        trades.append({
                            "entry_date": str(ts.date()),
                            "entry":      round(exec_px, 2),
                            "shares":     shares,
                        })
            else:
                pnl_pct  = (price - entry_px) / entry_px
                exit_sig = (
                    pnl_pct <= -stop_pct or
                    pnl_pct >= target_pct or
                    any(self._bt_eval_rule(row, r) for r in exit_rules) or
                    i == len(rows) - 1
                )
                if exit_sig:
                    exec_px  = price * (1 - SLIP)
                    proceeds = position * exec_px
                    capital += proceeds
                    cost     = trades[-1]["shares"] * trades[-1]["entry"]
                    pnl      = proceeds - cost
                    trades[-1].update({
                        "exit_date": str(ts.date()),
                        "exit":      round(exec_px, 2),
                        "pnl":       round(pnl, 2),
                        "pnl_pct":   round(pnl / cost * 100, 2),
                        "reason":    ("stop"   if pnl_pct <= -stop_pct else
                                      "target" if pnl_pct >= target_pct else "signal"),
                    })
                    position = 0
                    in_trade = False

        return trades


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = TelegramController()
    asyncio.run(bot.start())
