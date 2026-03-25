"""
Bot 6 – Risk Agent
Model  : Sonnet 4.6  (live trading decisions)
Role   : Portfolio risk gatekeeper.
         • Validates every trade setup from Bot 5 against hard risk limits
         • Monitors open positions for stop-loss / take-profit breaches
         • Tracks portfolio-level exposure, drawdown, and sector concentration
         • Publishes APPROVED or REJECTED setups to CHANNEL_ORDERS
         • Sends risk alerts to CHANNEL_ALERTS
"""

import asyncio
import json
from datetime import datetime, timezone, time as dt_time
from typing import Optional

import anthropic
import pytz

from config import Models, RedisConfig, RiskConfig, AnthropicConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


POSITION_MONITOR_INTERVAL = 15   # seconds

MARKET_TZ    = pytz.timezone("America/New_York")
MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


class RiskAgent(BaseBot):
    """
    Bot 6 – Risk Agent
    Hard rules first, AI nuance second.
    """

    BOT_ID = 6
    NAME   = "Risk Agent"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.SONNET)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.alpaca = AlpacaClient()

        self._portfolio_value: float    = 0.0
        self._daily_pnl: float          = 0.0
        self._open_positions: list[dict] = []
        self._daily_start_value: float  = 0.0
        self._rejected_today: int       = 0
        self._approved_today: int       = 0
        self._pre_market_logged: set    = set()  # symbols already logged during pre-market

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_STRATEGY, self._on_trade_setup)
        asyncio.create_task(self.bus.listen())
        await self._refresh_portfolio()
        self._daily_start_value = self._portfolio_value
        self.log(f"Risk Agent starting | portfolio=${self._portfolio_value:,.2f}")

    async def run(self):
        _last_session = self._get_market_session()
        while self.running:
            session = self._get_market_session()
            # Reset pre-market dedup set when market opens
            if session == "open" and _last_session != "open":
                self._pre_market_logged.clear()
                self.log("Market open — pre-market simulation log reset")
            _last_session = session
            try:
                await self._monitor_positions()
            except Exception as e:
                self.log(f"Position monitor error: {e}", "error")
            await asyncio.sleep(POSITION_MONITOR_INTERVAL)

    async def cleanup(self):
        self.log("Risk Agent stopped")

    # ── Market session ─────────────────────────────────────────────────────────

    def _get_market_session(self) -> str:
        now_et = datetime.now(MARKET_TZ).time()
        if MARKET_OPEN <= now_et < MARKET_CLOSE:
            return "open"
        elif now_et < MARKET_OPEN:
            return "pre_market"
        return "closed"

    # ── Portfolio refresh ──────────────────────────────────────────────────────

    async def _refresh_portfolio(self):
        loop = asyncio.get_event_loop()
        try:
            account   = await loop.run_in_executor(None, self.alpaca.get_account)
            positions = await loop.run_in_executor(None, self.alpaca.get_positions)
            self._portfolio_value  = account["portfolio_value"]
            self._open_positions   = positions
            self._daily_pnl = sum(p["unrealized_pl"] for p in positions)
        except Exception as e:
            self.log(f"Portfolio refresh error: {e}", "warning")

    # ── Hard risk checks ───────────────────────────────────────────────────────

    def _hard_checks(self, setup: dict) -> tuple[bool, list[str]]:
        """
        Return (passed, [reasons_if_failed]).
        These checks are deterministic and never overridden by AI.
        """
        reasons = []
        sym    = setup.get("symbol", "")
        dir_   = setup.get("direction", "long")
        size   = setup.get("position_size_usd", 0)
        entry  = setup.get("entry_price", 0)
        sl     = setup.get("stop_loss", 0)
        tp     = setup.get("take_profit", 0)

        # 1. Max position size
        if size > RiskConfig.MAX_POSITION_SIZE:
            reasons.append(
                f"Position size ${size:,.0f} exceeds max ${RiskConfig.MAX_POSITION_SIZE:,.0f}"
            )

        # 2. Max portfolio risk per trade
        if self._portfolio_value > 0 and entry > 0 and sl > 0:
            risk_per_share = abs(entry - sl)
            shares         = size / entry
            total_risk     = risk_per_share * shares
            pct_risk       = total_risk / self._portfolio_value
            if pct_risk > RiskConfig.MAX_PORTFOLIO_RISK:
                reasons.append(
                    f"Trade risk {pct_risk*100:.2f}% exceeds max {RiskConfig.MAX_PORTFOLIO_RISK*100:.1f}%"
                )

        # 3. Daily loss limit
        if self._daily_start_value > 0:
            daily_loss_pct = self._daily_pnl / self._daily_start_value
            if daily_loss_pct < -RiskConfig.MAX_DAILY_LOSS:
                reasons.append(
                    f"Daily loss {daily_loss_pct*100:.2f}% exceeds limit {RiskConfig.MAX_DAILY_LOSS*100:.1f}%"
                )

        # 4. Max open positions
        if len(self._open_positions) >= RiskConfig.MAX_OPEN_POSITIONS:
            # Allow closing/reducing but not new positions
            existing_syms = {p["symbol"] for p in self._open_positions}
            if sym not in existing_syms:
                reasons.append(
                    f"Max open positions ({RiskConfig.MAX_OPEN_POSITIONS}) reached"
                )

        # 5. Stop loss must be set and reasonable
        if not sl or sl <= 0:
            reasons.append("Stop loss not set")
        elif dir_ == "long" and sl >= entry:
            reasons.append("Stop loss must be below entry for long trades")
        elif dir_ == "short" and sl <= entry:
            reasons.append("Stop loss must be above entry for short trades")

        # 6. Take profit must be set
        if not tp or tp <= 0:
            reasons.append("Take profit not set")

        # 7. Risk/reward must be >= 1.5
        if entry and sl and tp:
            risk    = abs(entry - sl)
            reward  = abs(tp - entry)
            rr      = reward / risk if risk > 0 else 0
            if rr < 1.5:
                reasons.append(f"Risk/reward {rr:.2f} below minimum 1.5")

        # 8. Confidence threshold
        if setup.get("confidence", 0) < 0.65:
            reasons.append(f"Confidence {setup.get('confidence'):.2f} below threshold 0.65")

        return len(reasons) == 0, reasons

    # ── AI risk review ─────────────────────────────────────────────────────────

    async def _ai_risk_review(self, setup: dict, hard_passed: bool,
                               hard_reasons: list[str]) -> dict:
        """
        Sonnet reviews the setup for qualitative risks not captured by hard rules:
        • Market regime (trending vs choppy)
        • Correlation with existing positions
        • News risk / earnings proximity
        • Setup quality / timing
        """
        positions_summary = [
            {"sym": p["symbol"], "side": "long", "pnl": p["unrealized_pl"]}
            for p in self._open_positions
        ]

        prompt = (
            "You are a risk manager reviewing a trade setup.\n\n"
            f"SETUP:\n{json.dumps(setup, indent=2)}\n\n"
            f"CURRENT PORTFOLIO:\n{json.dumps(positions_summary, indent=2)}\n\n"
            f"Portfolio value: ${self._portfolio_value:,.2f}\n"
            f"Daily P&L: ${self._daily_pnl:,.2f}\n"
            f"Hard rule check: {'PASSED' if hard_passed else 'FAILED'}\n"
            f"Hard rule failures: {hard_reasons}\n\n"
            "Evaluate:\n"
            "1. Correlation risk with existing positions\n"
            "2. Setup timing quality (is entry well-defined?)\n"
            "3. Any qualitative risks (earnings, macro, overextension)\n"
            "4. Position sizing appropriateness\n\n"
            "Respond ONLY with JSON:\n"
            "{\"approve\": true, \"confidence_adjustment\": 0.0, "
            "\"adjusted_size_usd\": 5000, \"qualitative_risks\": [\"string\"], "
            "\"notes\": \"brief explanation\"}"
            "\n'approve' must be false if hard rules failed."
            "\nadjusted_size_usd can be <= original to reduce size."
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
            # Hard rules cannot be overridden
            if not hard_passed:
                result["approve"] = False
            return result
        except Exception as e:
            self.log(f"AI risk review error: {e}", "warning")
            return {"approve": hard_passed, "notes": "AI review failed, using hard rules only"}

    # ── Trade setup handler ────────────────────────────────────────────────────

    async def _on_trade_setup(self, setup: dict):
        if setup.get("type") != "trade_setup":
            return

        sym = setup.get("symbol", "?")

        # Outside market hours: simulate validation only, never publish orders or rejections
        if self._get_market_session() != "open":
            if sym not in self._pre_market_logged:
                hard_passed, hard_reasons = self._hard_checks(setup)
                status = "PASS" if hard_passed else f"FAIL: {', '.join(hard_reasons[:2])}"
                self.log(f"SIMULATED (pre-market) {sym}: {status}")
                self._pre_market_logged.add(sym)
            return

        await self._refresh_portfolio()

        # 1. Hard rules
        hard_passed, hard_reasons = self._hard_checks(setup)

        # 2. AI qualitative review
        ai_result = await self._ai_risk_review(setup, hard_passed, hard_reasons)
        approved  = ai_result.get("approve", False)

        # 3. Apply adjustments
        adjusted_setup = {**setup}
        if approved:
            adj_size = ai_result.get("adjusted_size_usd")
            if adj_size and adj_size < setup.get("position_size_usd", adj_size):
                adjusted_setup["position_size_usd"] = adj_size
                self.log(f"Size reduced: {sym} ${setup['position_size_usd']} → ${adj_size}")
            conf_adj = ai_result.get("confidence_adjustment", 0)
            adjusted_setup["confidence"] = round(
                adjusted_setup.get("confidence", 0.5) + conf_adj, 3
            )
            adjusted_setup["risk_notes"] = ai_result.get("qualitative_risks", [])
            self._approved_today += 1
        else:
            self._rejected_today += 1

        # 4. Publish result
        if approved:
            await self.publish(RedisConfig.CHANNEL_ORDERS, {
                "type":     "approved_trade",
                "status":   "approved",
                "risk_notes": ai_result.get("notes", ""),
                **adjusted_setup,
            })
            self.log(
                f"APPROVED: {sym} {setup.get('direction')} | "
                f"size=${adjusted_setup.get('position_size_usd'):,.0f} | "
                f"conf={adjusted_setup.get('confidence'):.2f}"
            )
        else:
            all_reasons = hard_reasons + [ai_result.get("notes", "")]
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":     "trade_rejected",
                "symbol":   sym,
                "reasons":  all_reasons,
                "setup":    setup,
            })
            self.log(
                f"REJECTED: {sym} | reasons={all_reasons[:2]}"
            )

        # Save state
        await self.save_state("stats", {
            "approved_today": self._approved_today,
            "rejected_today": self._rejected_today,
            "open_positions": len(self._open_positions),
            "daily_pnl":      self._daily_pnl,
            "portfolio_value": self._portfolio_value,
            "timestamp":      datetime.utcnow().isoformat(),
        }, ttl=300)

    # ── Position monitoring ────────────────────────────────────────────────────

    async def _monitor_positions(self):
        await self._refresh_portfolio()

        for pos in self._open_positions:
            sym        = pos["symbol"]
            entry      = pos["avg_entry"]
            current    = pos["current_price"]
            pnl_pct    = (current - entry) / entry if entry > 0 else 0

            # Stop loss breach
            if pnl_pct <= -RiskConfig.STOP_LOSS_PCT:
                await self.publish(RedisConfig.CHANNEL_ORDERS, {
                    "type":      "force_close",
                    "symbol":    sym,
                    "reason":    f"Stop loss triggered: {pnl_pct*100:.2f}%",
                    "priority":  "immediate",
                })
                await self.publish(RedisConfig.CHANNEL_ALERTS, {
                    "type":     "stop_loss_triggered",
                    "symbol":   sym,
                    "pnl_pct":  round(pnl_pct * 100, 2),
                    "entry":    entry,
                    "current":  current,
                })
                self.log(f"STOP LOSS: {sym} | pnl={pnl_pct*100:.2f}%", "warning")

            # Take profit breach
            elif pnl_pct >= RiskConfig.TAKE_PROFIT_PCT:
                await self.publish(RedisConfig.CHANNEL_ORDERS, {
                    "type":    "take_profit",
                    "symbol":  sym,
                    "reason":  f"Take profit reached: {pnl_pct*100:.2f}%",
                    "priority": "normal",
                })
                self.log(f"TAKE PROFIT: {sym} | pnl={pnl_pct*100:.2f}%")

        # Daily loss circuit breaker — only during market hours
        if self._daily_start_value > 0 and self._get_market_session() == "open":
            daily_loss_pct = self._daily_pnl / self._daily_start_value
            if daily_loss_pct < -RiskConfig.MAX_DAILY_LOSS:
                await self.publish(RedisConfig.CHANNEL_ORDERS, {
                    "type":    "halt_trading",
                    "reason":  f"Daily loss limit {daily_loss_pct*100:.2f}% exceeded",
                    "priority": "immediate",
                })
                await self.publish(RedisConfig.CHANNEL_ALERTS, {
                    "type":     "daily_loss_halt",
                    "daily_pnl": self._daily_pnl,
                    "pct":      round(daily_loss_pct * 100, 2),
                })
                self.log(
                    f"TRADING HALTED: daily loss {daily_loss_pct*100:.2f}%", "critical"
                )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = RiskAgent()
    asyncio.run(bot.start())
