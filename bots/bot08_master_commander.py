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
from shared.llm_router import LLMRouter


COMMAND_INTERVAL  = 300   # seconds between commander reviews (5 min)
HEARTBEAT_TIMEOUT = 90    # seconds before a bot is considered dead
MARKET_TZ         = pytz.timezone("America/New_York")   # hardcoded EST — never trust system TZ

# Market session boundaries (all EST, hardcoded — never query broker for these)
PRE_MARKET   = dt_time(4,  0)
MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
AFTER_HOURS  = dt_time(20, 0)

# EOD closing sequence (EST, hardcoded)
EOD_FRI_NO_TRADES = dt_time(14,  0)  # 2:00pm Fri — stop new trades early
EOD_WARN          = dt_time(15, 30)  # 3:30pm daily — stop new buys + warning telegram
EOD_CLOSE_START   = dt_time(15, 45)  # 3:45pm — begin closing positions (Fri: primary close)
EOD_HARD_CLOSE    = dt_time(15, 50)  # 3:50pm — hard close all remaining + cancel orders
EOD_VERIFY        = dt_time(15, 55)  # 3:55pm — verify zero positions + EOD telegram
EOD_BACKUP_END    = dt_time(16,  0)  # 4:00pm — backup watcher stops

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
    13: "Telegram Controller",
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
        self._router = LLMRouter(
            self.client,
            save_fn=self.save_state,
            get_fn=self.bus.get_state,
        )

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
        await self.bus.subscribe(RedisConfig.CHANNEL_CONTROL,   self._on_control_command)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._eod_close_task())
        asyncio.create_task(self._backup_close_task())
        asyncio.create_task(self._regime_train_task())
        asyncio.create_task(self._dashboard_refresh_task())
        await self._refresh_portfolio()
        self._session_start_val = self._portfolio_value
        self._cb.set_week_start(self._portfolio_value)
        # Write dashboard immediately so /status works before first command cycle
        self._market_session = self._get_market_session()
        await self.save_state("dashboard", self._build_dashboard(), ttl=90)
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

    async def _on_control_command(self, data: dict):
        """Handle commands published by Bot 13 (Telegram Controller) via CHANNEL_CONTROL."""
        cmd    = data.get("type", "")
        source = data.get("source", "unknown")
        reason = data.get("reason", cmd)
        self.log(f"Control command from {source}: {cmd} — {reason}")

        if cmd == "halt_trading":
            self._system_halted  = True
            self._halt_published = True
            await self._emergency_halt(reason)

        elif cmd == "halt_new_trades":
            await self._publish_halt_new_trades(reason)

        elif cmd == "resume_trading":
            self._system_halted  = False
            self._halt_published = False
            await self.save_state("market_gate", {
                "allow_new_trades": True,
                "reason":           "operator_resume",
                "timestamp":        datetime.utcnow().isoformat(),
            }, ttl=86400)
            await self.publish(RedisConfig.CHANNEL_ORDERS, {
                "type":   "resume_trading",
                "reason": reason,
            })
            self.log("Trading resumed by operator command")

        elif cmd == "close_all":
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self.alpaca.close_all_positions)
                self.log("Operator close_all: all positions closed")
            except Exception as e:
                self.log(f"Operator close_all error: {e}", "error")

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

    def _now_et(self) -> datetime:
        """Current datetime in EST — never trusts system timezone."""
        return datetime.now(MARKET_TZ)

    def _get_market_session(self) -> str:
        now_et   = self._now_et()
        t        = now_et.time()
        is_fri   = now_et.weekday() == 4

        if t < PRE_MARKET:
            return "closed"
        if PRE_MARKET <= t < MARKET_OPEN:
            return "pre_market"
        if MARKET_OPEN <= t < MARKET_CLOSE:
            if is_fri and t >= EOD_FRI_NO_TRADES:
                return "friday_wind_down"   # 2pm–4pm Friday: no new trades
            if t >= EOD_WARN:
                return "closing_soon"       # 3:30pm+: no new buys any day
            return "open"
        if MARKET_CLOSE <= t < AFTER_HOURS:
            return "after_hours"
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
        Sonnet reviews compressed system state (~75 tokens) and issues commands.
        Cached for 5 min — same conditions = same response, zero extra Sonnet cost.
        Only called when market is open and data feeds have content.
        """
        summary = LLMRouter.compress_system_state(system_state)
        ck      = LLMRouter.cache_key(summary)

        prompt = (
            "Commander. Only halt if pnl<-3%. Ignore empty data feeds.\n"
            f"{summary}\n"
            "JSON:{\"trading_allowed\":true,\"risk_level\":\"low\","
            "\"session_action\":\"continue\",\"alert_operator\":false,"
            "\"operator_message\":\"\",\"notes\":\"\"}"
        )

        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="sonnet",
            max_tokens=150,
            cache_key=ck,
        )
        if not raw:
            return {"trading_allowed": not self._system_halted,
                    "session_action": "continue", "risk_level": "medium", "notes": ""}
        try:
            raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()
            raw = re.sub(r"//[^\n]*", "", raw)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
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
        except Exception as e:
            self.log(f"AI review parse error: {e}", "warning")
            return {"trading_allowed": not self._system_halted,
                    "session_action": "continue", "risk_level": "medium", "notes": ""}

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

    # ── EOD helper methods ─────────────────────────────────────────────────────

    async def _sleep_until_et(self, hour: int, minute: int, second: int = 0):
        """Sleep until h:m:s EST today. No-op if already past that time."""
        now_et = self._now_et()
        target = now_et.replace(hour=hour, minute=minute, second=second, microsecond=0)
        secs   = (target - now_et).total_seconds()
        if secs > 0:
            await asyncio.sleep(secs)

    async def _get_positions_safe(self) -> list[dict]:
        """Fetch current positions; returns [] on error."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self.alpaca.get_positions)
        except Exception as e:
            self.log(f"get_positions error: {e}", "warning")
            return []

    async def _close_all_market_positions(self, label: str) -> tuple[list[dict], float]:
        """
        Fetch all open positions and close each one individually at market.
        Returns (closed_list, total_pnl).
        """
        positions = await self._get_positions_safe()
        loop      = asyncio.get_event_loop()
        closed:    list[dict] = []
        total_pnl: float      = 0.0
        for pos in positions:
            sym = pos["symbol"]
            pnl = float(pos.get("unrealized_pl", 0.0))
            try:
                await loop.run_in_executor(None, lambda s=sym: self.alpaca.close_position(s))
                closed.append({"symbol": sym, "pnl": pnl})
                total_pnl += pnl
                sign = "+" if pnl >= 0 else ""
                self.log(f"EOD [{label}] closed {sym}: {sign}${pnl:.2f}")
            except Exception as e:
                self.log(f"EOD [{label}] close failed {sym}: {e}", "warning")
        return closed, total_pnl

    async def _cancel_all_orders_safe(self):
        """Cancel all open orders; silent on error."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.alpaca.cancel_all_orders)
            self.log("EOD: all open orders cancelled")
        except Exception as e:
            self.log(f"EOD: cancel orders error: {e}", "warning")

    async def _publish_halt_new_trades(self, reason: str = "EOD — no new positions"):
        """Publish halt_new_trades to the order channel and update Redis gate."""
        await self.publish(RedisConfig.CHANNEL_ORDERS, {
            "type":   "halt_new_trades",
            "reason": reason,
        })
        await self.save_state("market_gate", {
            "allow_new_trades": False,
            "reason":           reason,
            "timestamp":        datetime.utcnow().isoformat(),
        }, ttl=7200)

    # ── End-of-day position close (milestone sequence) ────────────────────────

    async def _eod_close_task(self):
        """
        EOD close sequence — all times hardcoded EST (America/New_York).
        Never relies on Alpaca market hours API.

        Every weekday:
          3:30pm — warning telegram + halt new trades
          3:45pm — first close pass (Friday: primary close for weekend safety)
          3:50pm — hard close remaining + cancel all open orders
          3:55pm — verify zero positions + EOD telegram
                   If any remain: circuit-breaker force close + critical alert

        Friday only:
          2:00pm — halt new trades early (weekend risk buffer)
          (3:45-3:55 sequence same as above)
        """
        while self.running:
            # Next weekday 3:30pm (or 2:00pm on Friday) as first wake-up
            now_et    = self._now_et()
            is_friday = now_et.weekday() == 4
            first_h, first_m = (14, 0) if is_friday else (15, 30)

            target = now_et.replace(hour=first_h, minute=first_m, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            while target.weekday() >= 5:
                target += timedelta(days=1)

            wait_min = (target - self._now_et()).total_seconds() / 60
            self.log(f"EOD close task: next run in {wait_min:.0f} min "
                     f"({'Fri 2pm' if target.weekday() == 4 else '3:30pm'})")
            await asyncio.sleep((target - self._now_et()).total_seconds())

            now_et    = self._now_et()
            is_friday = now_et.weekday() == 4

            # ── 2:00pm Friday: stop new trades early ──────────────────────────
            if is_friday and now_et.time() < EOD_WARN:
                self.log("EOD: 2:00pm Friday — halting new trades for weekend safety")
                await self._publish_halt_new_trades("Friday 2pm — weekend risk buffer, no new trades")
                await self._send_telegram(
                    f"📅 <b>Friday 2:00pm — No new trades for the rest of the day</b>\n"
                    f"Extra buffer for weekend risk.\n"
                    f"Positions will close at 3:45pm EST."
                )
                # Now sleep until 3:30pm for the warning
                await self._sleep_until_et(15, 30)
                now_et = self._now_et()

            # ── 3:30pm: Closing warning ────────────────────────────────────────
            self.log("EOD: 3:30pm — closing warning + halting new trades")
            await self._publish_halt_new_trades("3:30pm EOD — no new positions accepted")
            close_str = "3:45pm (Friday — weekend safety)" if is_friday else "3:50pm"
            await self._send_telegram(
                f"⏰ <b>Closing warning — {now_et.strftime('%A %b %d')}</b>\n"
                f"No new trades accepted from 3:30pm.\n"
                f"Positions begin closing at {close_str} EST.\n"
                f"Hard close at 3:50pm EST no matter what."
            )

            # ── 3:45pm: First close pass ───────────────────────────────────────
            await self._sleep_until_et(15, 45)
            self.log(f"EOD: 3:45pm — starting position close {'(Friday primary)' if is_friday else ''}")
            await self._send_telegram(
                f"🔄 <b>{'Friday' if is_friday else 'EOD'} — closing all positions now "
                f"({self._now_et().strftime('%I:%M %p')} EST)</b>"
            )
            closed, total_pnl = await self._close_all_market_positions("3:45pm")

            # ── 3:50pm: Hard close remaining + cancel all orders ──────────────
            await self._sleep_until_et(15, 50)
            self.log("EOD: 3:50pm HARD CLOSE — force closing any remaining + cancel orders")
            await self._cancel_all_orders_safe()
            remaining = await self._get_positions_safe()
            if remaining:
                syms = [p["symbol"] for p in remaining]
                self.log(f"EOD 3:50pm: {len(remaining)} still open — force closing: {syms}", "warning")
                extra_closed, extra_pnl = await self._close_all_market_positions("3:50pm hard close")
                existing_syms = {c["symbol"] for c in closed}
                for c in extra_closed:
                    if c["symbol"] not in existing_syms:
                        closed.append(c)
                        total_pnl += c["pnl"]

            # ── 3:55pm: Verify zero positions + EOD telegram ──────────────────
            await self._sleep_until_et(15, 55)
            self.log("EOD: 3:55pm — verifying zero positions")
            still_open    = await self._get_positions_safe()
            verified_clear = not bool(still_open)

            if still_open:
                syms = [p["symbol"] for p in still_open]
                self.log(
                    f"EOD 3:55pm CRITICAL: {syms} still open — emergency force close!", "critical"
                )
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(None, self.alpaca.close_all_positions)
                except Exception as e:
                    self.log(f"EOD 3:55pm emergency close failed: {e}", "error")
                await asyncio.sleep(5)
                final_check = await self._get_positions_safe()
                if final_check:
                    await self._send_telegram(
                        f"🚨 <b>CRITICAL — Positions STILL OPEN at 3:55pm!</b>\n"
                        f"Symbols: {[p['symbol'] for p in final_check]}\n"
                        f"Time: {self._now_et().strftime('%I:%M:%S %p')} EST\n"
                        f"⚠️ Manual intervention required immediately!"
                    )

            # Tally win/loss
            for c in closed:
                if c.get("pnl", 0) > 0:
                    self._wins_today   += 1
                elif c.get("pnl", 0) < 0:
                    self._losses_today += 1

            total_trades = self._real_trades_today
            self._real_trades_today = 0
            await self._refresh_portfolio()

            # Build and send EOD Telegram summary
            pnl_sign = "+" if total_pnl >= 0 else ""
            pv_str   = f"${self._portfolio_value:,.2f}" if self._portfolio_value else "N/A"
            now_et   = self._now_et()

            header = "📅 <b>FRIDAY — ALL POSITIONS CLOSED FOR WEEKEND</b>" if is_friday \
                     else "🔔 <b>END OF DAY SUMMARY</b>"
            lines = [header, f"<b>{now_et.strftime('%A, %b %d %Y')}</b>", ""]
            lines += [
                f"Total trades today: {total_trades}",
                f"Winning trades:     {self._wins_today}",
                f"Losing trades:      {self._losses_today}",
                "",
            ]
            if closed:
                lines.append("<b>Positions closed:</b>")
                for c in closed:
                    sign = "+" if c.get("pnl", 0) >= 0 else ""
                    lines.append(f"  - {c['symbol']}: {sign}${c.get('pnl', 0):.2f}")
                lines.append("")
            lines += [
                f"<b>Total P/L: {pnl_sign}${total_pnl:.2f}</b>",
                f"<b>Portfolio value: {pv_str}</b>",
                "",
            ]
            if verified_clear:
                if is_friday:
                    lines.append("All positions closed for the weekend ✅")
                else:
                    lines.append("All positions closed ✅")
            else:
                lines.append("⚠️ Some positions may still be open — check Alpaca dashboard!")

            lines.append(f"Closed at: {now_et.strftime('%I:%M:%S %p')} EST")

            # Late-alert detection: if this fires after 4pm, flag it as a bug
            if now_et.time() >= MARKET_CLOSE:
                self.log(
                    f"EOD telegram sent at {now_et.strftime('%I:%M:%S %p')} — AFTER 4:00pm! "
                    f"Timing bug — investigate immediately.", "critical"
                )
                lines += ["", "🚨 <b>WARNING: EOD alert sent AFTER 4:00pm!</b>",
                          "This is a critical timing bug — please investigate."]

            await self._send_telegram("\n".join(lines))

            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":       "eod_positions_closed",
                "positions":  closed,
                "total_pnl":  total_pnl,
                "is_friday":  is_friday,
                "verified":   verified_clear,
                "timestamp":  datetime.utcnow().isoformat(),
            })
            self.log(
                f"EOD complete | closed={len(closed)} | pnl={pnl_sign}${total_pnl:.2f} | "
                f"verified_clear={verified_clear} | is_friday={is_friday}"
            )

    async def _backup_close_task(self):
        """
        Backup safety net: polls every 60 seconds between 3:45pm–4:00pm EST.
        Force-closes any positions the main EOD logic may have missed.
        Runs independently of _eod_close_task so a single failure can't block both.
        """
        while self.running:
            now_et = self._now_et()
            target = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            while target.weekday() >= 5:
                target += timedelta(days=1)
            await asyncio.sleep((target - self._now_et()).total_seconds())

            if self._now_et().weekday() >= 5:
                continue

            self.log("Backup close watcher: active (3:45–4:00pm)")
            while self._now_et().time() < EOD_BACKUP_END:
                positions = await self._get_positions_safe()
                if positions:
                    syms = [p["symbol"] for p in positions]
                    self.log(
                        f"Backup watcher: {len(positions)} position(s) still open at "
                        f"{self._now_et().strftime('%I:%M:%S %p')} EST — {syms}", "warning"
                    )
                    await self._close_all_market_positions("backup_watcher")
                await asyncio.sleep(60)

    # ── Dashboard helpers ──────────────────────────────────────────────────────

    def _build_dashboard(self) -> dict:
        """
        Snapshot of all in-memory state — used for the initial write and the
        60-second refresh task. Does NOT call Alpaca or Redis.
        """
        bot_health = self._check_bot_health()
        dead_bots  = [bid for bid, s in bot_health.items() if s == "dead"]
        cb = self._cb_state
        return {
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
                "name":      self._current_regime,
                "scale":     self._regime_scale,
                "uncertain": self._regime.is_uncertain(),
                "trained":   self._regime_trained,
            },
        }

    async def _dashboard_refresh_task(self):
        """
        Writes dashboard to Redis every 60 s regardless of market session.
        Keeps /status responsive at all hours and prevents the key from
        expiring between the 300-second command cycles.
        Only hits Alpaca for portfolio refresh during active market sessions.
        """
        await asyncio.sleep(10)   # let setup() finish first
        while self.running:
            try:
                # Refresh portfolio value from Alpaca during active sessions
                if self._market_session not in ("closed", "after_hours"):
                    await self._refresh_portfolio()
                # Always update session (may have transitioned while market closed)
                self._market_session = self._get_market_session()
                await self.save_state("dashboard", self._build_dashboard(), ttl=90)
            except Exception as e:
                self.log(f"Dashboard refresh error: {e}", "warning")
            await asyncio.sleep(60)

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

        # Market session — hardcoded EST checks, no broker API
        session = self._get_market_session()
        if session != self._market_session:
            self.log(f"Market session: {self._market_session} → {session}")
            self._market_session = session

            if session == "closed":
                if not self._system_halted:
                    await self.publish(RedisConfig.CHANNEL_ORDERS, {
                        "type":   "halt_trading",
                        "reason": "Market closed",
                    })
                    self.log("Auto-halt: market closed")
                # Re-open the market gate for tomorrow
                await self.save_state("market_gate", {
                    "allow_new_trades": True,
                    "reason":           "market_closed_reset",
                    "timestamp":        datetime.utcnow().isoformat(),
                }, ttl=86400)

            elif session == "open":
                if self._system_halted:
                    self._system_halted  = False
                    self._halt_published = False
                    self.log("Market opened – trading resumed")
                self._real_trades_today   = 0
                self._halt_ignored_logged = False
                self._consecutive_losses  = 0
                self._wins_today          = 0
                self._losses_today        = 0
                # Ensure gate is open at market open
                await self.save_state("market_gate", {
                    "allow_new_trades": True,
                    "reason":           "market_open",
                    "timestamp":        datetime.utcnow().isoformat(),
                }, ttl=86400)

            elif session in ("closing_soon", "friday_wind_down"):
                # Publish halt_new_trades on transition (EOD task also sends at exact times;
                # this catches the edge case where command_cycle fires exactly at the boundary)
                reason = (
                    "Friday 2pm wind-down — no new trades until Monday"
                    if session == "friday_wind_down"
                    else "3:30pm closing window — no new trades"
                )
                await self._publish_halt_new_trades(reason)
                self.log(f"Market gate closed: {reason}")

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

        # ── AI review — only during open session with real data ─────────────
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

        # Save full dashboard state (refresh task also writes every 60 s)
        await self.save_state("dashboard", system_state, ttl=90)

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
