"""
Bot 5 – Strategy Agent
Model  : Sonnet 4.6  (live trading decisions)
Role   : The brain that synthesises ALL input signals into concrete trade setups.
         Listens to:  CHANNEL_MARKET_DATA, CHANNEL_NEWS, CHANNEL_MOMENTUM,
                      CHANNEL_OPTIONS_FLOW
         Produces:    CHANNEL_STRATEGY  →  structured trade proposals
                      consumed by Bot 6 (Risk) → Bot 7 (Execution) → Bot 8 (Commander)
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

import anthropic

from config import Models, RedisConfig, RiskConfig, AnthropicConfig
from shared.base_bot import BaseBot


DECISION_DEBOUNCE = 600  # min seconds between decision cycles (10 min)


class StrategyAgent(BaseBot):
    """
    Bot 5 – Strategy Agent
    Synthesises momentum, news, and options signals into trade setups.
    """

    BOT_ID = 5
    NAME   = "Strategy Agent"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.SONNET)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)

        # Signal buffers (reset each decision cycle)
        self._market_data: dict[str, dict]  = {}
        self._news_signals: list[dict]       = []
        self._momentum_signals: list[dict]   = []
        self._options_signals: list[dict]    = []
        self._last_decision: datetime | None = None  # debounce for event-driven cycles
        self._portfolio_value: float         = 0.0   # synced from risk agent state

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA,  self._on_market_data)
        await self.bus.subscribe(RedisConfig.CHANNEL_NEWS,         self._on_news)
        await self.bus.subscribe(RedisConfig.CHANNEL_MOMENTUM,     self._on_momentum)
        await self.bus.subscribe(RedisConfig.CHANNEL_OPTIONS_FLOW, self._on_options)
        asyncio.create_task(self.bus.listen())
        self.log("Strategy Agent starting")

    async def run(self):
        # Keep alive; decisions fire event-driven from _on_market_data
        while self.running:
            await asyncio.sleep(3600)

    async def cleanup(self):
        self.log("Strategy Agent stopped")

    # ── Signal ingestion ───────────────────────────────────────────────────────

    async def _on_market_data(self, data: dict):
        sym = data.get("symbol")
        if sym:
            self._market_data[sym] = data
        # Trigger a decision cycle at most once per DECISION_DEBOUNCE seconds
        now = datetime.now(timezone.utc)
        if (self._last_decision is None or
                (now - self._last_decision).total_seconds() >= DECISION_DEBOUNCE):
            self._last_decision = now
            try:
                await self._decision_cycle()
            except Exception as e:
                self.log(f"Decision cycle error: {e}", "error")

    async def _on_news(self, data: dict):
        if data.get("sentiment") != "neutral":
            self._news_signals.append(data)
            if len(self._news_signals) > 50:
                self._news_signals = self._news_signals[-50:]

    async def _on_momentum(self, data: dict):
        self._momentum_signals.append(data)
        if len(self._momentum_signals) > 100:
            self._momentum_signals = self._momentum_signals[-100:]

    async def _on_options(self, data: dict):
        self._options_signals.append(data)
        if len(self._options_signals) > 30:
            self._options_signals = self._options_signals[-30:]

    # ── Position sizing ────────────────────────────────────────────────────────

    def _safe_position_size(self, entry, stop_loss) -> float:
        """
        Hardcoded for a $1,000 account:
          max_position = $200  (20% of $1,000)
          max_risk     = $19   (1.9% of $1,000 — safely under the 2% gate)
          shares       = int($19 / stop_distance)   ← whole shares only
          position_usd = min(shares * entry, $200)  ← hard $200 cap
        """
        max_position_usd = 200.0   # hard cap: 20% of $1,000
        max_risk_usd     = 19.0    # hard cap: 1.9% of $1,000

        # Coerce None → 0 so comparisons never raise TypeError
        entry     = float(entry or 0)
        stop_loss = float(stop_loss or 0)

        if entry <= 0 or stop_loss <= 0:
            return max_risk_usd    # fallback: $19

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return max_risk_usd

        shares       = int(max_risk_usd / stop_distance)   # whole shares, truncated
        position_usd = min(shares * entry, max_position_usd)
        return round(position_usd, 2)

    # ── Signal aggregation ─────────────────────────────────────────────────────

    def _aggregate_signals(self) -> dict:
        """Build a compact multi-signal view for Sonnet to reason over."""

        # Top momentum candidates
        top_momentum = sorted(
            [s for s in self._momentum_signals if s.get("direction") != "neutral"],
            key=lambda x: x.get("score", 0.5),
            reverse=True
        )[:10]

        # News sentiment by symbol
        news_by_sym: dict[str, list] = {}
        for n in self._news_signals[-20:]:
            for sym in n.get("symbols", []):
                news_by_sym.setdefault(sym, []).append({
                    "sentiment": n.get("sentiment"),
                    "catalyst":  n.get("catalyst"),
                    "score":     n.get("score"),
                })

        # Options signals
        options = [{
            "sym":        o.get("symbol"),
            "bias":       o.get("bias"),
            "conf":       o.get("confidence"),
            "horizon":    o.get("horizon"),
            "insight":    o.get("insight"),
        } for o in self._options_signals[-5:]]

        # Price context for top candidates
        candidate_syms = list({s["symbol"] for s in top_momentum} |
                               {o["sym"] for o in options if o["sym"]})
        price_context = {}
        for sym in candidate_syms:
            md = self._market_data.get(sym, {})
            if md:
                ind = md.get("indicators", {})
                price_context[sym] = {
                    "price":   md.get("price"),
                    "rsi_14":  ind.get("rsi_14"),
                    "macd_h":  (ind.get("macd") or {}).get("histogram"),
                    "pct_b":   (ind.get("bollinger") or {}).get("pct_b"),
                    "vwap":    ind.get("vwap"),
                    "vol_ratio": ind.get("volume_ratio"),
                    "sma50":   ind.get("sma_50"),
                    "sma200":  ind.get("sma_200"),
                    "atr":     ind.get("atr_14"),
                    "ai_flag": md.get("ai_flag"),
                }

        return {
            "momentum":      top_momentum,
            "news":          news_by_sym,
            "options":       options,
            "price_context": price_context,
        }

    # ── AI strategy generation ─────────────────────────────────────────────────

    async def _generate_trade_setups(self, signals: dict) -> list[dict]:
        """Use Sonnet to synthesise signals into actionable trade setups."""

        # Hardcoded account constraints — never derive from Alpaca portfolio value
        # (paper account starts at $100,000 which would inflate all calculations)
        PORTFOLIO    = 1000.0
        MAX_POSITION = 200.0    # 20% of $1,000
        MAX_RISK     = 19.0     # 1.9% of $1,000 (safely under the 2.5% gate)

        prompt = (
            "You are an expert stock trader synthesising real-time signals into trade setups.\n\n"
            "POSITION SIZING (HARD LIMITS — small $1,000 account):\n"
            f"  • Max position_size_usd: ${MAX_POSITION:.0f}  ← absolute ceiling, never exceed\n"
            f"  • Max risk per trade:    ${MAX_RISK:.0f}  (1.9% of portfolio)\n"
            f"  • Sizing: shares = int({MAX_RISK:.0f} / stop_distance), "
            f"then position_size_usd = min(shares * entry, {MAX_POSITION:.0f})\n"
            f"  • Example: entry=$50, stop=$49 → shares=int(19/1)=19 → $950 → capped ${MAX_POSITION:.0f}\n"
            f"  • Stop loss: {RiskConfig.STOP_LOSS_PCT*100:.1f}% from entry\n"
            f"  • Take profit: {RiskConfig.TAKE_PROFIT_PCT*100:.1f}% from entry\n"
            f"  • Max open positions: {RiskConfig.MAX_OPEN_POSITIONS}\n\n"
            "SIGNALS:\n"
            f"{json.dumps(signals, indent=2)}\n\n"
            "Generate up to 3 high-conviction trade setups. Each setup must include:\n"
            "  - symbol, direction (long/short), entry_price, stop_loss, take_profit\n"
            f"  - position_size_usd (must be <= {MAX_POSITION:.0f})\n"
            "  - confidence (0-1), timeframe (scalp/intraday/swing)\n"
            "  - thesis (2-3 sentences explaining the confluence of signals)\n"
            "  - required_confirmations (list of conditions that must be met before entry)\n"
            "  - risk_reward_ratio\n\n"
            "Only include setups with strong signal confluence (momentum + news or options).\n"
            "Respond ONLY with JSON:\n"
            "{\"setups\": [{\"symbol\": \"AAPL\", \"direction\": \"long\", "
            "\"entry_price\": 195.50, \"stop_loss\": 191.59, \"take_profit\": 203.32, "
            f"\"position_size_usd\": {MAX_POSITION:.0f}, \"confidence\": 0.78, "
            "\"timeframe\": \"intraday\", \"thesis\": \"...\", "
            "\"required_confirmations\": [\"price above VWAP\", \"RSI > 55\"], "
            "\"risk_reward_ratio\": 2.0}], "
            "\"market_bias\": \"bullish|bearish|neutral\", "
            "\"notes\": \"brief market commentary\"}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            setups = parsed.get("setups", [])

            # ALWAYS overwrite position_size_usd — never use Claude's value
            for s in setups:
                ai_size = s.get("position_size_usd", "?")
                s["position_size_usd"] = self._safe_position_size(
                    s.get("entry_price") or 0,
                    s.get("stop_loss") or 0,
                )
                self.log(
                    f"Position size override: {s.get('symbol')} "
                    f"AI=${ai_size} → enforced=${s['position_size_usd']}"
                )

            return setups, parsed.get("market_bias", "neutral"), parsed.get("notes", "")
        except Exception as e:
            self.log(f"Strategy generation error: {e}", "error")
            return [], "neutral", ""

    # ── Decision cycle ─────────────────────────────────────────────────────────

    async def _decision_cycle(self):
        if not self._market_data:
            self.log("No market data yet")
            return

        signals = self._aggregate_signals()

        # Skip only if ALL signal types are empty (1-signal threshold)
        if not signals["momentum"] and not signals["options"] and not signals["news"]:
            self.log("Insufficient signals, skipping cycle")
            return

        result = await self._generate_trade_setups(signals)
        setups, market_bias, notes = result

        if not setups:
            self.log(f"No setups generated | market_bias={market_bias}")
        else:
            for setup in setups:
                # Final hard cap — catches any edge case before hitting the wire
                setup["position_size_usd"] = min(
                    float(setup.get("position_size_usd") or 19.0), 200.0
                )
                await self.publish(RedisConfig.CHANNEL_STRATEGY, {
                    "type":         "trade_setup",
                    "market_bias":  market_bias,
                    **setup,
                })
                self.log(
                    f"Trade setup: {setup.get('symbol')} {setup.get('direction')} | "
                    f"conf={setup.get('confidence')} | RR={setup.get('risk_reward_ratio')} | "
                    f"{setup.get('thesis', '')[:80]}"
                )

        # Save strategy state
        await self.save_state("latest", {
            "type":          "strategy_state",
            "market_bias":   market_bias,
            "notes":         notes,
            "setup_count":   len(setups),
            "signal_count":  {
                "momentum": len(self._momentum_signals),
                "news":     len(self._news_signals),
                "options":  len(self._options_signals),
            },
            "timestamp":     datetime.utcnow().isoformat(),
        }, ttl=120)

        # Reset signal buffers
        self._news_signals     = []
        self._momentum_signals = []
        self._options_signals  = []

        self.log(f"Cycle complete | {len(setups)} setups | bias={market_bias}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = StrategyAgent()
    asyncio.run(bot.start())
