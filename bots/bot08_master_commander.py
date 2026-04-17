"""
Bot 8 – Master Commander
Model  : Sonnet 4.6  (live trading decisions)
Role   : System orchestrator and supreme decision authority.
         • Monitors health of all 11 other bots via heartbeats
         • Aggregates system-wide state every 30 seconds
         • Makes override decisions: halt/resume trading, emergency close-all
         • Manages market session logic (pre-market, open, close, after-hours)
         • Publishes system status to CHANNEL_ALERTS
         • Provides the human operator a unified system dashboard via Redis
"""

import asyncio
import json
import re
from datetime import datetime, timezone, time as dt_time, timedelta
from typing import Optional
import pytz

import anthropic
import httpx

from config import Models, RedisConfig, AnthropicConfig, RiskConfig, AlertConfig, RegimeConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient
from shared.circuit_breaker import CircuitBreaker, CBState
from shared.regime_detector import RegimeDetector


COMMAND_INTERVAL  = 300   # seconds between commander reviews (5 min)
HEARTBEAT_TIMEOUT = 90    # seconds before a bot is considered dead
MARKET_TZ         = pytz.timezone("America/New_York")

MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
PRE_MARKET   = dt_time(4, 0)
AFTER_HOURS  = dt_time(20, 0)

# Bot registry: id → name
BOT_REGISTRY = {
    1:  "News Sentiment Bot",
    2:  "Options Flow Bot",
    3:  "Momentum Scanner",
    4:  "Data Agent",
    5:  "Strategy Agent",
    6:  "Risk Agent",
    7:  "Execution Agent",
    9:  "Alert Bot",
    10: "Backtesting Bot",
    11: "Strategy Builder",
    12: "Research Bot",
}


class MasterCommander(BaseBot):
    """
    Bot 8 – Master Commander
    The system leader. No trade happens without passing through the risk pipeline
    that this bot oversees.
    """

    BOT_ID = 8
    NAME   = "Master Commander"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.SONNET)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.alpaca = AlpacaClient()

        # Bot health tracking
        self._heartbeats: dict[int, str] = {}       # bot_id → last_seen ISO
        self._bot_status:  dict[int, str] = {}       # bot_id → alive/dead/unknown

        # System state
        self._system_halted     = False
        self._halt_published    = False   # prevents publishing halt signal more than once
        self._market_session    = "unknown"
        self._daily_pnl         = 0.0
        self._portfolio_value   = 0.0
        self._session_start_val = 0.0
        self._total_commands    = 0
        self._real_trades_today = 0       # submitted orders; halt requires >= 10
        self._halt_ignored_logged = False # log "ignoring halt" warning only once
        self._consecutive_losses  = 0    # incremented on stop_loss_triggered, reset on market open
        self._wins_today          = 0    # positions closed at take-profit
        self._losses_today        = 0    # positions closed at stop-loss

        # Aggregated signals
        self._risk_stats:      dict = {}
        self._strategy_state:  dict = {}
        self._momentum_board:  dict = {}
        self._news_summary:    dict = {}

        # Circuit breaker and regime detection
        self._cb              = CircuitBreaker()
        self._regime          = RegimeDetector()
        self._cb_state:       CBState | None = None
        self._current_regime: str  = "neutral"
        self._regime_scale:   float = 1.0
        self._regime_trained: bool  = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_HEARTBEAT, self._on_heartbeat)
        await self.bus.subscribe(RedisConfig.CHANNEL_RISK,      self._on_risk_update)
        await self.bus.subscribe(RedisConfig.CHANNEL_ALERTS,    self._on_alert)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._eod_close_task())
        asyncio.create_task(self._regime_train_task())
        await self._refresh_portfolio()
        self._session_start_val = self._portfolio_value
        self._cb.set_week_start(self._portfolio_value)
        self.log(
            f"Master Commander online | portfolio=${self._portfolio_value:,.2f} | "
            f"paper={True}"
        )

    async def run(self):
        while self.running:
            await asyncio.sleep(COMMAND_INTERVAL)
            try:
                await self._command_cycle()
            except Exception as e:
                self.log(f"Command cycle error: {e}", "error")

    async def cleanup(self):
        self.log("Master Commander shutting down")
        await self._publish_system_status("shutdown")

    # ── Event handlers ─────────────────────────────────────────────────────────

    async def _on_heartbeat(self, data: dict):
        bot_id = data.get("bot_id")
        if bot_id:
            self._heartbeats[bot_id] = data.get("timestamp", datetime.utcnow().isoformat())
            self._bot_status[bot_id] = "alive"

    async def _on_risk_update(self, data: dict):
        self._risk_stats = data

    async def _on_alert(self, data: dict):
        alert_type = data.get("type", "")

        # Track real submitted orders so we know real trading is happening
        if alert_type == "order_submitted":
            self._real_trades_today += 1

        # Critical alerts — only act during market_open with real trading activity
        if alert_type in ("daily_loss_halt", "execution_halted"):
            if (not self._system_halted
                    and self._market_session == "open"
                    and self._real_trades_today >= 10):
                self._system_halted = True
                self.log(f"System halt triggered: {alert_type}", "critical")
                await self._emergency_halt(data.get("reason", alert_type))
            elif self._market_session != "open":
                if not self._halt_ignored_logged:
                    self._halt_ignored_logged = True
                    self.log(f"Ignoring {alert_type} outside market hours (further duplicates suppressed)", "warning")
            elif self._real_trades_today < 10:
                self.log(
                    f"Ignoring {alert_type} — only {self._real_trades_today} real trades "
                    f"submitted (need 10 before halt activates)", "warning"
                )

        elif alert_type == "stop_loss_triggered":
            sym = data.get("symbol")
            self._consecutive_losses += 1
            self._losses_today += 1
            self.log(
                f"Stop loss triggered: {sym} | pnl={data.get('pnl_pct')}% | "
                f"consecutive_losses={self._consecutive_losses}", "warning"
            )

        elif alert_type == "take_profit":
            self._wins_today += 1
            self._consecutive_losses = 0  # reset streak on a winner

    # ── Regime detection ──────────────────────────────────────────────────────

    async def _regime_train_task(self):
        """Train HMM regime detector once at startup, then every 24 h."""
        while self.running:
            await self._train_regime()
            await asyncio.sleep(RegimeConfig.RETRAIN_HOURS * 3600)

    async def _train_regime(self):
        """Fetch 2 years of SPY daily bars and train the HMM."""
        try:
            from alpaca.data.timeframe import TimeFrame
            from datetime import timezone
            start = (datetime.now(timezone.utc) - timedelta(days=RegimeConfig.TRAIN_DAYS + 5)
                     ).strftime("%Y-%m-%d")
            loop  = asyncio.get_event_loop()
            bars  = await loop.run_in_executor(
                None, lambda: self.alpaca.get_bars(["SPY"], TimeFrame.Day, start)
            )
            spy_bars = bars.get("SPY", [])
            if len(spy_bars) < 60:
                self.log("Regime training: insufficient SPY bars", "warning")
                return
            import numpy as np
            closes  = np.array([b["close"]  for b in spy_bars], dtype=float)
            volumes = np.array([b["volume"] for b in spy_bars], dtype=float)
            ok = await loop.run_in_executor(
                None, lambda: self._regime.train(closes, volumes)
            )
            if ok:
                self._regime_trained = True
                self.log("Regime detector trained successfully")
            else:
                self.log("Regime training failed (hmmlearn unavailable?)", "warning")
        except Exception as e:
            self.log(f"Regime training error: {e}", "warning")

    async def _update_regime(self):
        """Predict current regime from recent SPY bars and cache to Redis."""
        if not self._regime_trained:
            return
        try:
            from alpaca.data.timeframe import TimeFrame
            from datetime import timezone
            start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
            loop  = asyncio.get_event_loop()
            bars  = await loop.run_in_executor(
                None, lambda: self.alpaca.get_bars(["SPY"], TimeFrame.Day, start)
            )
            spy_bars = bars.get("SPY", [])
            if len(spy_bars) < 6:
                return
            import numpy as np
            closes  = np.array([b["close"]  for b in spy_bars], dtype=float)
            volumes = np.array([b["volume"] for b in spy_bars], dtype=float)
            regime = await loop.run_in_executor(
                None, lambda: self._regime.predict(closes, volumes)
            )
            self._current_regime = regime
            self._regime_scale   = self._regime.allocation_scale()
            uncertain            = self._regime.is_uncertain()
            # Publish for bot05 and dashboard
            await self.save_state("regime", {
                "regime":    regime,
                "scale":     self._regime_scale,
                "uncertain": uncertain,
                "timestamp": datetime.utcnow().isoformat(),
            }, ttl=600)
            self.log(
                f"Regime: {regime} | scale={self._regime_scale:.2f} | "
                f"uncertain={uncertain}"
            )
        except Exception as e:
            self.log(f"Regime update error: {e}", "warning")

    # ── Portfolio ──────────────────────────────────────────────────────────────

    async def _refresh_portfolio(self):
        loop = asyncio.get_event_loop()
        try:
            account   = await loop.run_in_executor(None, self.alpaca.get_account)
            positions = await loop.run_in_executor(None, self.alpaca.get_positions)
            self._portfolio_value = account["portfolio_value"]
            self._daily_pnl       = sum(p["unrealized_pl"] for p in positions)
        except Exception as e:
            self.log(f"Portfolio refresh error: {e}", "warning")

    # ── Market session logic ───────────────────────────────────────────────────

    def _get_market_session(self) -> str:
        now_et = datetime.now(MARKET_TZ).time()
        if now_et < PRE_MARKET:
            return "closed"
        elif PRE_MARKET <= now_et < MARKET_OPEN:
            return "pre_market"
        elif MARKET_OPEN <= now_et < MARKET_CLOSE:
            return "open"
        elif MARKET_CLOSE <= now_et < AFTER_HOURS:
            return "after_hours"
        else:
            return "closed"

    def _check_bot_health(self) -> dict[int, str]:
        """Check which bots are alive based on heartbeat recency."""
        now = datetime.now(timezone.utc)
        status = {}
        for bot_id, name in BOT_REGISTRY.items():
            last_seen = self._heartbeats.get(bot_id)
            if not last_seen:
                status[bot_id] = "never_seen"
            else:
                try:
                    last_dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    age = (now - last_dt).total_seconds()
                    status[bot_id] = "alive" if age < HEARTBEAT_TIMEOUT else "dead"
                except Exception:
                    status[bot_id] = "unknown"
        return status

    # ── AI command decision ────────────────────────────────────────────────────

    async def _ai_command_review(self, system_state: dict) -> dict:
        """
        Sonnet reviews full system state and issues high-level commands.
        This is the strategic override layer.
        """
        prompt = (
            "You are the Master Commander of an AI trading system. "
            "Review the system state and issue commands.\n\n"
            f"SYSTEM STATE:\n{json.dumps(system_state, indent=2)}\n\n"
            "IMPORTANT RULES:\n"
            "- NEVER recommend halt or pause because data feeds (strategy/momentum/news) are empty. "
            "Empty data feeds are normal — bots update on their own schedule.\n"
            "- ONLY recommend halt if daily_pnl_pct is worse than -3.0% (financial loss only).\n"
            "- Default session_action to 'continue' unless there is a clear financial emergency.\n\n"
            "Assess:\n"
            "1. Are risk levels acceptable based on portfolio P/L only?\n"
            "2. Are there any bots that appear to be malfunctioning?\n"
            "3. Any strategic adjustments needed?\n\n"
            "Respond ONLY with JSON:\n"
            "{\"trading_allowed\": true, "
            "\"confidence\": 0.85, "
            "\"commands\": [\"string\"], "
            "\"risk_level\": \"low|medium|high|critical\", "
            "\"session_action\": \"continue|close_all\", "
            "\"notes\": \"brief commentary\", "
            "\"alert_operator\": false, "
            "\"operator_message\": \"\"}"
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
            # Robust cleaning: strip fences, comments, trailing commas
            raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            raw = re.sub(r"//[^\n]*", "", raw)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            # Bracket-tracking extraction of first complete {...}
            start = raw.find("{")
            if start != -1:
                depth, in_str, escape, end = 0, False, False, -1
                for i, ch in enumerate(raw[start:], start):
                    if escape: escape = False; continue
                    if ch == "\\" and in_str: escape = True; continue
                    if ch == '"': in_str = not in_str
                    elif not in_str:
                        if ch == "{": depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0: end = i; break
                if end != -1:
                    raw = raw[start:end + 1]
            return json.loads(raw)
        except json.JSONDecodeError:
            pass   # silent fallback below
        except Exception as e:
            self.log(f"AI command review error: {e}", "warning")
            return {
                "trading_allowed": not self._system_halted,
                "session_action":  "continue",
                "risk_level":      "medium",
                "notes":           "AI review failed – maintaining current state",
            }

    # ── Emergency halt ─────────────────────────────────────────────────────────

    async def _emergency_halt(self, reason: str):
        # Never publish the halt signal more than once per session
        if self._halt_published:
            return
        self._halt_published = True
        self._system_halted  = True
        self.log(f"EMERGENCY HALT: {reason}", "critical")

        await self.publish(RedisConfig.CHANNEL_ORDERS, {
            "type":     "halt_trading",
            "reason":   reason,
            "priority": "immediate",
        })
        await self._publish_system_status("emergency_halt", reason)

    # ── System status broadcast ────────────────────────────────────────────────

    async def _publish_system_status(self, status: str, notes: str = ""):
        await self.publish(RedisConfig.CHANNEL_ALERTS, {
            "type":           "system_status",
            "status":         status,
            "notes":          notes,
            "portfolio_value": self._portfolio_value,
            "daily_pnl":      self._daily_pnl,
            "market_session": self._market_session,
            "system_halted":  self._system_halted,
        })

    # ── Telegram ───────────────────────────────────────────────────────────────

    async def _send_telegram(self, text: str):
        """Send a Telegram message directly from Master Commander."""
        token   = AlertConfig.TELEGRAM_BOT_TOKEN
        chat_id = AlertConfig.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={
                    "chat_id":    chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                })
                if resp.status_code != 200:
                    self.log(f"Telegram send failed: {resp.status_code}", "warning")
        except Exception as e:
            self.log(f"Telegram error: {e}", "warning")

    # ── End-of-day position close ──────────────────────────────────────────────

    async def _eod_close_task(self):
        """Close all positions at 3:55pm EST every trading day."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            sleep_secs = (target - now_et).total_seconds()
            self.log(f"EOD position close scheduled in {sleep_secs / 60:.1f} min")
            await asyncio.sleep(sleep_secs)
            # Skip weekends
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            await self._run_eod_close()

    async def _run_eod_close(self):
        """
        EOD close: fetch ALL positions, close every one, verify none remain,
        retry any stragglers, then send Telegram summary.
        """
        self.log("EOD 3:55pm — starting end-of-day position close")
        loop = asyncio.get_event_loop()

        # ── Pass 1: fetch and close every position ─────────────────────────
        try:
            positions = await loop.run_in_executor(None, self.alpaca.get_positions)
        except Exception as e:
            self.log(f"EOD: failed to fetch positions: {e}", "error")
            positions = []

        closed: list[dict] = []
        total_pnl = 0.0

        for pos in positions:
            sym = pos["symbol"]
            pnl = float(pos.get("unrealized_pl", 0.0))
            try:
                await loop.run_in_executor(None, lambda s=sym: self.alpaca.close_position(s))
                closed.append({"symbol": sym, "pnl": pnl})
                total_pnl += pnl
                sign = "+" if pnl >= 0 else ""
                self.log(f"EOD closed {sym}: {sign}${pnl:.2f}")
            except Exception as e:
                self.log(f"EOD close failed for {sym}: {e}", "warning")

        # ── Wait 5 s then verify ────────────────────────────────────────────
        await asyncio.sleep(5)

        try:
            remaining = await loop.run_in_executor(None, self.alpaca.get_positions)
        except Exception as e:
            self.log(f"EOD: verification fetch failed: {e}", "warning")
            remaining = []

        # ── Pass 2: retry any positions still open ──────────────────────────
        if remaining:
            self.log(
                f"EOD: {len(remaining)} position(s) still open after pass 1 — retrying: "
                f"{[p['symbol'] for p in remaining]}",
                "warning",
            )
            for pos in remaining:
                sym = pos["symbol"]
                pnl = float(pos.get("unrealized_pl", 0.0))
                try:
                    await loop.run_in_executor(None, lambda s=sym: self.alpaca.close_position(s))
                    # Update P/L if already recorded (duplicate position), otherwise append
                    existing = next((c for c in closed if c["symbol"] == sym), None)
                    if existing:
                        existing["pnl"] += pnl
                        total_pnl += pnl
                    else:
                        closed.append({"symbol": sym, "pnl": pnl})
                        total_pnl += pnl
                    sign = "+" if pnl >= 0 else ""
                    self.log(f"EOD retry closed {sym}: {sign}${pnl:.2f}")
                except Exception as e:
                    self.log(f"EOD retry close failed for {sym}: {e}", "warning")

            # Final verification
            await asyncio.sleep(3)
            try:
                still_open = await loop.run_in_executor(None, self.alpaca.get_positions)
                if still_open:
                    syms = [p["symbol"] for p in still_open]
                    self.log(f"EOD: WARNING — {len(still_open)} position(s) could not be closed: {syms}", "warning")
                else:
                    self.log("EOD: all positions confirmed closed after retry")
            except Exception as e:
                self.log(f"EOD: final verification failed: {e}", "warning")
        else:
            self.log("EOD: all positions confirmed closed")

        # ── Tally EOD closed positions into win/loss counters ───────────────
        for c in closed:
            if c["pnl"] > 0:
                self._wins_today   += 1
            elif c["pnl"] < 0:
                self._losses_today += 1

        # ── Reset daily trade counter ───────────────────────────────────────
        total_trades = self._real_trades_today
        self._real_trades_today = 0
        self.log("EOD: daily trade counter reset to 0")

        # ── Refresh portfolio for accurate closing value ────────────────────
        await self._refresh_portfolio()

        # ── Build and send Telegram summary ────────────────────────────────
        pnl_sign  = "+" if total_pnl >= 0 else ""
        pv_str    = f"${self._portfolio_value:,.2f}" if self._portfolio_value else "N/A"

        lines = ["🔔 <b>END OF DAY SUMMARY</b>", ""]
        lines.append(f"Total trades today: {total_trades}")
        lines.append(f"Winning trades: {self._wins_today}")
        lines.append(f"Losing trades:  {self._losses_today}")
        lines.append("")
        if closed:
            lines.append("<b>Positions closed:</b>")
            for c in closed:
                sign = "+" if c["pnl"] >= 0 else ""
                lines.append(f"  - {c['symbol']}: {sign}${c['pnl']:.2f}")
            lines.append("")
        lines.append(f"<b>Total P/L: {pnl_sign}${total_pnl:.2f}</b>")
        lines.append(f"<b>Portfolio value: {pv_str}</b>")

        await self._send_telegram("\n".join(lines))

        # Publish EOD event so other bots can react
        await self.publish(RedisConfig.CHANNEL_ALERTS, {
            "type":      "eod_positions_closed",
            "positions": closed,
            "total_pnl": total_pnl,
            "timestamp": datetime.utcnow().isoformat(),
        })

    # ── Command cycle ──────────────────────────────────────────────────────────

    async def _command_cycle(self):
        self._total_commands += 1
        await self._refresh_portfolio()

        # ── Circuit breaker VETO check (highest priority) ──────────────────────
        if CircuitBreaker.is_locked():
            self.log("LOCKFILE.lock exists — trading locked out. Delete file to resume.", "critical")
            if not self._system_halted:
                await self._emergency_halt("LOCKFILE.lock present — manual intervention required")
        else:
            cb = self._cb.check(self._portfolio_value, self._session_start_val)
            self._cb_state = cb
            if cb.level > 0:
                self.log(f"Circuit breaker L{cb.level}: {cb.action} — {cb.reason}", "warning")
            if cb.action == "emergency_exit" and not self._system_halted:
                await self._emergency_halt(cb.reason)
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(None, self.alpaca.close_all_positions)
                except Exception as e:
                    self.log(f"CB emergency close error: {e}", "error")
                await self._send_telegram(f"🚨 CIRCUIT BREAKER EMERGENCY\n{cb.reason}")
            elif cb.action in ("lockout",) and not self._system_halted:
                await self._emergency_halt(cb.reason)
            elif cb.action == "close_all" and not self._system_halted:
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(None, self.alpaca.close_all_positions)
                    self.log("Circuit breaker: all positions closed")
                except Exception as e:
                    self.log(f"CB close_all error: {e}", "error")
            # Publish CB scale so bot05 can adjust position sizing
            await self.save_state("cb_state", {
                "level":          cb.level,
                "action":         cb.action,
                "reason":         cb.reason,
                "position_scale": cb.position_scale,
                "trading_allowed": cb.trading_allowed,
                "timestamp":      datetime.utcnow().isoformat(),
            }, ttl=120)

        # ── Update regime ──────────────────────────────────────────────────────
        await self._update_regime()

        # Market session
        session = self._get_market_session()
        if session != self._market_session:
            self.log(f"Market session: {self._market_session} → {session}")
            self._market_session = session

            # Auto-halt outside market hours
            if session == "closed":
                if not self._system_halted:
                    await self.publish(RedisConfig.CHANNEL_ORDERS, {
                        "type":   "halt_trading",
                        "reason": "Market closed",
                    })
                    self.log("Auto-halt: market closed")

            elif session == "open":
                if self._system_halted:
                    self._system_halted  = False
                    self._halt_published = False
                    self.log("Market opened – trading resumed")
                self._real_trades_today    = 0     # reset daily counter each open
                self._halt_ignored_logged  = False # allow one warning next pre-market
                self._consecutive_losses   = 0     # reset loss streak each day
                self._wins_today           = 0
                self._losses_today         = 0

        # Bot health
        bot_health = self._check_bot_health()
        dead_bots  = [bid for bid, s in bot_health.items() if s == "dead"]
        if dead_bots:
            dead_names = [BOT_REGISTRY.get(b, f"Bot {b}") for b in dead_bots]
            self.log(f"Dead bots detected: {dead_names}", "warning")

        # Load aggregated state from Redis
        try:
            self._risk_stats     = await self.bus.get_state("bot6:stats") or {}
            self._strategy_state = await self.bus.get_state("bot5:latest") or {}
            self._momentum_board = await self.bus.get_state("bot3:leaderboard") or {}
            self._news_summary   = await self.bus.get_state("bot1:latest_summary") or {}
        except Exception as e:
            self.log(f"State load error: {e}", "warning")

        # Build system state snapshot
        cb = self._cb_state
        system_state = {
            "timestamp":       datetime.utcnow().isoformat(),
            "market_session":  self._market_session,
            "system_halted":   self._system_halted,
            "portfolio_value": self._portfolio_value,
            "daily_pnl":       self._daily_pnl,
            "daily_pnl_pct":   round(self._daily_pnl / self._session_start_val * 100, 2)
                               if self._session_start_val > 0 else 0,
            "bot_health":      {BOT_REGISTRY.get(k, k): v for k, v in bot_health.items()},
            "dead_bots":       dead_bots,
            "risk":            self._risk_stats,
            "strategy":        self._strategy_state,
            "momentum":        self._momentum_board,
            "news":            self._news_summary,
            "circuit_breaker": {
                "level":          cb.level if cb else 0,
                "action":         cb.action if cb else "continue",
                "position_scale": cb.position_scale if cb else 1.0,
                "reason":         cb.reason if cb else "",
            },
            "regime": {
                "name":           self._current_regime,
                "scale":          self._regime_scale,
                "uncertain":      self._regime.is_uncertain(),
                "trained":        self._regime_trained,
            },
        }

        # ── Hard-coded halt rules (never based on data feed availability) ──────
        if self._market_session == "open" and not self._system_halted:
            daily_pnl_pct = system_state.get("daily_pnl_pct", 0)
            if daily_pnl_pct < -2.0:
                await self._emergency_halt(
                    f"Portfolio down {daily_pnl_pct:.1f}% — exceeds 2% daily loss limit"
                )
            elif self._consecutive_losses > 5:
                await self._emergency_halt(
                    f"Halted after {self._consecutive_losses} consecutive stop-loss triggers"
                )

        # ── AI review — only when data feeds have actual content ─────────────
        if self._market_session == "open" and not self._system_halted:
            data_available = any([
                self._strategy_state,
                self._momentum_board,
                self._news_summary,
                self._risk_stats,
            ])
            if not data_available:
                self.log("Data feeds empty — skipping AI review, will retry next cycle", "warning")
            else:
                decision = await self._ai_command_review(system_state)

                # AI may only trigger close_all (not halt/pause — those are hard-coded above)
                session_action = decision.get("session_action", "continue")
                if session_action == "close_all":
                    self.log("AI commanded: close all positions", "warning")
                    loop = asyncio.get_event_loop()
                    try:
                        await loop.run_in_executor(None, self.alpaca.close_all_positions)
                    except Exception as e:
                        self.log(f"Close all failed: {e}", "error")

                if decision.get("alert_operator") and decision.get("operator_message"):
                    await self.publish(RedisConfig.CHANNEL_ALERTS, {
                        "type":       "operator_alert",
                        "message":    decision["operator_message"],
                        "risk_level": decision.get("risk_level", "medium"),
                    })

                risk_level = decision.get("risk_level", "medium")
                system_state["commander_risk_level"] = risk_level
                system_state["commander_notes"]      = decision.get("notes", "")

        # Save dashboard state (for external monitoring)
        await self.save_state("dashboard", system_state, ttl=60)

        cb_lvl = self._cb_state.level if self._cb_state else 0
        self.log(
            f"Cycle #{self._total_commands} | session={self._market_session} | "
            f"portfolio=${self._portfolio_value:,.2f} | pnl=${self._daily_pnl:+,.2f} | "
            f"halted={self._system_halted} | dead_bots={len(dead_bots)} | "
            f"cb=L{cb_lvl} | regime={self._current_regime}(x{self._regime_scale:.2f})"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    commander = MasterCommander()
    asyncio.run(commander.start())
