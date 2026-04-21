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
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import AlertConfig, RedisConfig, AlpacaConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


POLL_TIMEOUT  = 30   # long-polling timeout (seconds)
POLL_INTERVAL = 1    # sleep between poll attempts on error


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
        self.log(
            f"Telegram Controller ready | authorized_chat={self._authorized_id}"
        )

    async def run(self):
        if not self._token or not self._authorized_id:
            # Sit idle rather than crash the system
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
        self.log("Telegram Controller stopped")

    # ── Telegram API helpers ───────────────────────────────────────────────────

    async def _get_updates(self, client: httpx.AsyncClient) -> list[dict]:
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        resp = await client.get(url, params={
            "offset":  self._update_offset,
            "timeout": POLL_TIMEOUT,
        })
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
            "start":     self._cmd_start,
            "stop":      self._cmd_stop,
            "pause":     self._cmd_pause,
            "resume":    self._cmd_resume,
            "status":    self._cmd_status,
            "profit":    self._cmd_profit,
            "positions": self._cmd_positions,
            "close":     self._cmd_close,
            "restart":   self._cmd_restart,
            "kill":      self._cmd_kill,
        }
        handler = handlers.get(cmd)
        if handler:
            await handler(chat_id)
        else:
            await self._send(chat_id,
                "Unknown command. Available: /start /stop /pause /resume "
                "/status /profit /positions /close /restart /kill"
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = TelegramController()
    asyncio.run(bot.start())
