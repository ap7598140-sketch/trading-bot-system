"""
Bot 9 – Alert Bot
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : Unified notification gateway.
         • Listens to CHANNEL_ALERTS for all system events
         • Deduplicates and throttles alerts
         • Routes to: console log, Slack webhook, Telegram, email (configurable)
         • Generates human-readable summaries with Haiku
         • Sends Telegram BUY/SELL trade notifications and daily pre-market briefing
         • Maintains an alert history for the daily report
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional
from collections import defaultdict

import aiohttp
import httpx
import anthropic

from config import Models, RedisConfig, AlertConfig, AnthropicConfig
from shared.base_bot import BaseBot


# Alert severity levels
SEVERITY = {
    "system_status":        "info",
    "operator_alert":       "critical",
    "daily_loss_halt":      "critical",
    "execution_halted":     "critical",
    "stop_loss_triggered":  "warning",
    "take_profit":          "info",
    "order_submitted":      "info",
    "order_filled":         "info",
    "order_failed":         "error",
    "trade_rejected":       "warning",
    "position_closed":      "info",
    "news_sentiment":       "info",
    "emergency_halt":       "critical",
}

# Throttle: don't send same alert type more than once per N seconds
THROTTLE_SECONDS = {
    "stop_loss_triggered": 60,
    "order_submitted":     10,
    "order_filled":        10,
    "news_sentiment":      300,
    "system_status":       120,
}


class AlertBot(BaseBot):
    """
    Bot 9 – Alert Bot
    Keeps the operator informed without drowning them in noise.
    """

    BOT_ID = 9
    NAME   = "Alert Bot"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.HAIKU)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)

        self._alert_history: list[dict]       = []
        self._last_sent: dict[str, datetime]  = {}   # alert_type → last_sent time
        self._last_seen: dict[str, datetime]  = {}   # alert_type → last received time (dedup)
        self._counts: dict[str, int]          = defaultdict(int)
        self._DEDUP_SECONDS                   = 60   # ignore same alert type within this window

        # Track open positions for SELL message enrichment: symbol → {price, time, shares}
        self._open_trades: dict[str, dict]    = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_ALERTS, self._on_alert)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._daily_summary_scheduler())
        asyncio.create_task(self._pre_market_briefing_scheduler())
        self.log("Alert Bot starting")

    async def run(self):
        while self.running:
            await asyncio.sleep(300)
            await self._save_history_state()

    async def cleanup(self):
        self.log(f"Alert Bot stopped | {len(self._alert_history)} alerts logged")

    # ── Alert handler ──────────────────────────────────────────────────────────

    async def _on_alert(self, data: dict):
        alert_type = data.get("type", "unknown")

        # Deduplicate: drop the same alert type if seen within the last 60 seconds
        # Exception: trade events (buy/sell) are keyed by symbol+type so each trade fires
        dedup_key = alert_type
        if alert_type in ("order_submitted", "position_closed",
                          "stop_loss_triggered", "take_profit"):
            sym = data.get("symbol", "")
            dedup_key = f"{alert_type}:{sym}"

        now = datetime.utcnow()
        last = self._last_seen.get(dedup_key)
        if last and (now - last).total_seconds() < self._DEDUP_SECONDS:
            return
        self._last_seen[dedup_key] = now

        severity   = SEVERITY.get(alert_type, "info")

        # Log always
        self._alert_history.append({
            **data,
            "severity":  severity,
            "logged_at": datetime.utcnow().isoformat(),
        })
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

        self._counts[alert_type] += 1

        # Console log with severity formatting
        log_fn = {
            "critical": self.logger.critical,
            "error":    self.logger.error,
            "warning":  self.logger.warning,
            "info":     self.logger.info,
        }.get(severity, self.logger.info)

        log_fn(f"[ALERT] {alert_type} | {self._format_alert(data)}")

        # ── Telegram trade notifications (bypass throttle — each trade matters) ──
        if alert_type == "order_submitted":
            await self._telegram_buy(data)
        elif alert_type in ("position_closed", "stop_loss_triggered", "take_profit"):
            await self._telegram_sell(data)

        # Throttle check before external notifications
        if not self._should_send(alert_type):
            return

        # Generate human-readable message
        message = await self._ai_format_alert(data, severity)

        # Route to external channels
        if severity in ("critical", "error"):
            await asyncio.gather(
                self._send_slack(message, severity),
                return_exceptions=True,
            )
        elif severity == "warning":
            await self._send_slack(message, severity)

    # ── Throttle logic ─────────────────────────────────────────────────────────

    def _should_send(self, alert_type: str) -> bool:
        throttle = THROTTLE_SECONDS.get(alert_type, 0)
        if throttle == 0:
            return True
        last = self._last_sent.get(alert_type)
        if last and (datetime.utcnow() - last).total_seconds() < throttle:
            return False
        self._last_sent[alert_type] = datetime.utcnow()
        return True

    # ── Alert formatting ───────────────────────────────────────────────────────

    def _format_alert(self, data: dict) -> str:
        """Quick one-line summary without AI."""
        t = data.get("type", "")
        if t == "stop_loss_triggered":
            return f"{data.get('symbol')} SL hit | pnl={data.get('pnl_pct')}%"
        elif t == "order_submitted":
            return f"{data.get('symbol')} {data.get('side')} {data.get('shares')} @ market"
        elif t == "order_filled":
            return f"{data.get('symbol')} filled | id={data.get('order_id', '')[:8]}"
        elif t == "trade_rejected":
            return f"{data.get('symbol')} rejected | {data.get('reasons', ['?'])[0]}"
        elif t == "daily_loss_halt":
            return f"DAILY LOSS HALT | pnl={data.get('pct')}%"
        elif t == "system_status":
            return f"status={data.get('status')} session={data.get('market_session')}"
        else:
            return json.dumps({k: v for k, v in data.items()
                               if k not in ("type", "timestamp", "source_bot")})[:120]

    async def _ai_format_alert(self, data: dict, severity: str) -> str:
        """Use Haiku to write a clear, concise notification message."""
        prompt = (
            f"Write a concise trading alert notification (max 2 sentences, no jargon).\n"
            f"Severity: {severity}\n"
            f"Event data: {json.dumps(data)}\n"
            "Reply with ONLY the notification text."
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=128,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            return response.content[0].text.strip()
        except Exception:
            return self._format_alert(data)

    # ── Slack ──────────────────────────────────────────────────────────────────

    async def _send_slack(self, message: str, severity: str):
        if not AlertConfig.SLACK_WEBHOOK or not AlertConfig.SLACK_WEBHOOK.startswith("http"):
            return
        emoji = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(severity, "")
        payload = {
            "text": f"{emoji} *Trading Bot Alert* [{severity.upper()}]\n{message}",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AlertConfig.SLACK_WEBHOOK,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        self.log(f"Slack send failed: {resp.status}", "warning")
        except Exception as e:
            self.log(f"Slack error: {e}", "warning")

    # ── Telegram ───────────────────────────────────────────────────────────────

    async def _send_telegram(self, text: str):
        """Send a Telegram message via Bot API using httpx."""
        token = AlertConfig.TELEGRAM_BOT_TOKEN
        chat_id = AlertConfig.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    self.log(f"Telegram send failed: {resp.status_code} {resp.text[:100]}", "warning")
        except Exception as e:
            self.log(f"Telegram error: {e}", "warning")

    async def _telegram_buy(self, data: dict):
        """Format and send a BUY notification to Telegram."""
        sym        = data.get("symbol", "?")
        price      = data.get("price") or data.get("limit_price")
        confidence = data.get("confidence")
        reason     = data.get("reason") or data.get("signal", "")
        risk_pct   = data.get("risk_pct") or data.get("portfolio_risk_pct")
        shares     = data.get("shares") or data.get("qty")

        # Record entry so we can compute P&L on close
        if sym:
            self._open_trades[sym] = {
                "entry_price": price,
                "entry_time":  datetime.utcnow(),
                "shares":      shares,
            }

        price_str      = f"${price:,.2f}" if price else "market"
        confidence_str = f"{round(confidence * 100)}%" if confidence else "N/A"
        risk_str       = f"{round(risk_pct * 100, 1)}%" if risk_pct else "N/A"
        reason_str     = str(reason)[:80] if reason else "N/A"

        text = (
            f"🟢 <b>BUY {sym}</b>\n"
            f"Price: {price_str}\n"
            f"Confidence: {confidence_str}\n"
            f"Reason: {reason_str}\n"
            f"Risk: {risk_str}"
        )
        await self._send_telegram(text)

    async def _telegram_sell(self, data: dict):
        """Format and send a SELL notification to Telegram."""
        sym       = data.get("symbol", "?")
        exit_price = data.get("price") or data.get("exit_price") or data.get("fill_price")
        pnl_pct   = data.get("pnl_pct")
        pnl_abs   = data.get("pnl") or data.get("realized_pl")

        trade = self._open_trades.pop(sym, {})
        entry_price = trade.get("entry_price") or data.get("entry_price")
        entry_time  = trade.get("entry_time")
        shares      = trade.get("shares") or data.get("shares") or data.get("qty")

        # Compute hold time
        if entry_time:
            delta = datetime.utcnow() - entry_time
            total_mins = int(delta.total_seconds() / 60)
            hold_str = f"{total_mins} mins" if total_mins < 60 else f"{total_mins // 60}h {total_mins % 60}m"
        else:
            hold_str = "N/A"

        # Compute P&L if not provided
        if pnl_abs is None and entry_price and exit_price and shares:
            try:
                pnl_abs = round((float(exit_price) - float(entry_price)) * float(shares), 2)
            except Exception:
                pass
        if pnl_pct is None and entry_price and exit_price:
            try:
                pnl_pct = round((float(exit_price) - float(entry_price)) / float(entry_price) * 100, 2)
            except Exception:
                pass

        entry_str = f"${float(entry_price):,.2f}" if entry_price else "N/A"
        exit_str  = f"${float(exit_price):,.2f}" if exit_price else "market"
        pnl_sign  = "+" if (pnl_abs or 0) >= 0 else ""
        pnl_abs_str = f"{pnl_sign}${abs(float(pnl_abs)):,.2f}" if pnl_abs is not None else "N/A"
        pnl_pct_str = f"{pnl_sign}{pnl_pct}%" if pnl_pct is not None else "N/A"

        # Choose emoji based on alert type and P&L
        alert_type = data.get("type", "")
        if alert_type == "stop_loss_triggered":
            icon = "🛑"
        elif (pnl_abs or 0) >= 0:
            icon = "🔴"
        else:
            icon = "🔴"

        text = (
            f"{icon} <b>SELL {sym}</b>\n"
            f"Entry: {entry_str}  →  Exit: {exit_str}\n"
            f"Profit: {pnl_abs_str} ({pnl_pct_str})\n"
            f"Hold time: {hold_str}"
        )
        await self._send_telegram(text)

    # ── Pre-market briefing ────────────────────────────────────────────────────

    async def _pre_market_briefing_scheduler(self):
        """Fire a pre-market briefing at 9:25 AM ET every trading day."""
        import pytz
        et = pytz.timezone("America/New_York")
        while self.running:
            now_et = datetime.now(et)
            target = now_et.replace(hour=9, minute=25, second=0, microsecond=0)
            if now_et >= target:
                target = target + timedelta(days=1)
            wait_seconds = (target - now_et).total_seconds()
            # Sleep in chunks of max 1 hour so we handle day boundaries correctly
            await asyncio.sleep(min(wait_seconds, 3600))
            now_et = datetime.now(et)
            if now_et.hour == 9 and 25 <= now_et.minute < 30:
                await self._send_pre_market_briefing()

    async def _send_pre_market_briefing(self):
        """Compose and send a 9:25 AM pre-market briefing to Telegram."""
        # Fetch state snapshots from Redis
        try:
            dashboard  = await self.bus.get_state("bot8:dashboard") or {}
            momentum   = await self.bus.get_state("bot3:leaderboard") or {}
            news       = await self.bus.get_state("bot1:latest_summary") or {}
            risk_stats = await self.bus.get_state("bot6:stats") or {}
        except Exception:
            dashboard = momentum = news = risk_stats = {}

        portfolio_val  = dashboard.get("portfolio_value", 0)
        daily_pnl      = dashboard.get("daily_pnl", 0)
        market_mood    = news.get("market_mood", "neutral")
        bull_count     = news.get("bullish", 0)
        bear_count     = news.get("bearish", 0)

        # Top 3 momentum setups
        top_setups: list[str] = []
        if isinstance(momentum, dict):
            leaders = momentum.get("leaders", [])
            if not leaders and isinstance(momentum, list):
                leaders = momentum
            for item in leaders[:3]:
                sym   = item.get("symbol", "?")
                score = item.get("score") or item.get("momentum_score", 0)
                top_setups.append(f"  • {sym} (score: {score})")

        setups_text = "\n".join(top_setups) if top_setups else "  • No top setups yet"
        pnl_sign    = "+" if daily_pnl >= 0 else ""
        mood_emoji  = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(market_mood, "➡️")

        text = (
            f"📊 <b>PRE-MARKET BRIEFING</b> — {datetime.utcnow().strftime('%b %d, %Y')}\n\n"
            f"💼 Portfolio: <b>${portfolio_val:,.2f}</b>  "
            f"({pnl_sign}${daily_pnl:,.2f} today)\n\n"
            f"{mood_emoji} Market sentiment: <b>{market_mood.upper()}</b> "
            f"(📰 {bull_count} bull / {bear_count} bear)\n\n"
            f"🎯 <b>Top setups:</b>\n{setups_text}\n\n"
            f"⏰ Market opens in ~5 minutes"
        )
        await self._send_telegram(text)
        self.log("Pre-market briefing sent to Telegram")

    # ── Daily summary ──────────────────────────────────────────────────────────

    async def _daily_summary_scheduler(self):
        """Send a daily summary at 4:15 PM ET."""
        import pytz
        et = pytz.timezone("America/New_York")
        while self.running:
            now_et = datetime.now(et)
            target = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
            if now_et >= target:
                target = target + timedelta(days=1)
            wait_seconds = (target - now_et).total_seconds()
            await asyncio.sleep(min(wait_seconds, 3600))
            if (datetime.now(et).hour == 16 and
                    datetime.now(et).minute >= 15):
                await self._send_daily_summary()

    async def _send_daily_summary(self):
        """Generate and send end-of-day summary using Haiku."""
        today_alerts = [
            a for a in self._alert_history
            if a.get("logged_at", "")[:10] == datetime.utcnow().strftime("%Y-%m-%d")
        ]
        counts = defaultdict(int)
        for a in today_alerts:
            counts[a.get("type", "unknown")] += 1

        prompt = (
            "Write a brief end-of-day trading system summary (3-5 bullet points).\n"
            f"Alert counts today: {dict(counts)}\n"
            f"Total alerts: {len(today_alerts)}\n"
            "Focus on: trades executed, risk events, system health.\n"
            "Reply with ONLY the bullet points."
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            summary = response.content[0].text.strip()
        except Exception:
            summary = f"Total alerts today: {len(today_alerts)}\nCounts: {dict(counts)}"

        self.log(f"Daily Summary:\n{summary}")
        await self._send_slack(f"📊 *End of Day Summary*\n{summary}", "info")
        await self._send_telegram(f"📊 <b>End of Day Summary</b>\n{summary}")

    # ── State persistence ──────────────────────────────────────────────────────

    async def _save_history_state(self):
        await self.save_state("alert_counts", dict(self._counts), ttl=86400)
        await self.save_state("recent_alerts", self._alert_history[-20:], ttl=3600)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = AlertBot()
    asyncio.run(bot.start())
