"""
Bot 9 – Alert Bot
Model  : Haiku 4.5  (fast, cheap, high-frequency)
Role   : Unified notification gateway.
         • Listens to CHANNEL_ALERTS for all system events
         • Deduplicates and throttles alerts
         • Routes to: console log, Slack webhook, email (configurable)
         • Generates human-readable summaries with Haiku
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
        self._counts: dict[str, int]          = defaultdict(int)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_ALERTS, self._on_alert)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._daily_summary_scheduler())
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
        if not AlertConfig.SLACK_WEBHOOK:
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
