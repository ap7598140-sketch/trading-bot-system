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
from datetime import datetime, timezone, timedelta
from typing import Optional

import anthropic

import pytz
from config import Models, RedisConfig, RiskConfig, AnthropicConfig, UniverseConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient
from shared.llm_router import LLMRouter

MARKET_TZ = pytz.timezone("America/New_York")


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
        self.alpaca = AlpacaClient()
        self._router = LLMRouter(
            self.client,
            save_fn=self.save_state,
            get_fn=self.bus.get_state,
        )

        # Signal buffers (reset each decision cycle)
        self._market_data: dict[str, dict]  = {}
        self._news_signals: list[dict]       = []
        self._momentum_signals: list[dict]   = []
        self._options_signals: list[dict]    = []
        self._last_decision: datetime | None = None  # debounce for event-driven cycles
        self._portfolio_value: float         = 0.0   # refreshed from Alpaca each cycle
        self._strongest_sector: str          = ""    # set at 9am, used all day
        self._sector_symbols: list[str]      = []    # symbols in strongest sector
        self._regime_scale: float            = 1.0   # from bot08 Redis state
        self._cb_scale: float                = 1.0   # circuit breaker position scale

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA,  self._on_market_data)
        await self.bus.subscribe(RedisConfig.CHANNEL_NEWS,         self._on_news)
        await self.bus.subscribe(RedisConfig.CHANNEL_MOMENTUM,     self._on_momentum)
        await self.bus.subscribe(RedisConfig.CHANNEL_OPTIONS_FLOW, self._on_options)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._morning_sector_task())
        await self._refresh_portfolio()
        self.log(f"Strategy Agent starting | portfolio=${self._portfolio_value:,.2f}")

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
            self.log(
                f"News signal received: {data.get('symbols')} "
                f"{data.get('sentiment')} score={data.get('score')} "
                f"[{data.get('catalyst') or data.get('title','')[:50]}] "
                f"| buffer={len(self._news_signals)}"
            )

    async def _on_momentum(self, data: dict):
        self._momentum_signals.append(data)
        if len(self._momentum_signals) > 100:
            self._momentum_signals = self._momentum_signals[-100:]
        self.log(
            f"Momentum signal received: {data.get('symbol')} "
            f"{data.get('direction')} score={data.get('score')} "
            f"grade={data.get('grade')} | buffer={len(self._momentum_signals)}"
        )
        # Trigger a decision cycle immediately when fresh momentum arrives —
        # don't wait for the next market-data event (which may already have fired)
        now = datetime.now(timezone.utc)
        if (self._last_decision is None or
                (now - self._last_decision).total_seconds() >= DECISION_DEBOUNCE):
            self._last_decision = now
            try:
                await self._decision_cycle()
            except Exception as e:
                self.log(f"Decision cycle error (momentum trigger): {e}", "error")

    async def _on_options(self, data: dict):
        self._options_signals.append(data)
        if len(self._options_signals) > 30:
            self._options_signals = self._options_signals[-30:]
        self.log(
            f"Options signal received: {data.get('symbol')} "
            f"{data.get('bias')} conf={data.get('confidence')} "
            f"| buffer={len(self._options_signals)}"
        )

    # ── Morning sector strength check ─────────────────────────────────────────

    async def _morning_sector_task(self):
        """At 9:00am EST, check sector ETFs and set the day's strongest sector."""
        while self.running:
            now_et = datetime.now(MARKET_TZ)
            target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now_et).total_seconds())
            if datetime.now(MARKET_TZ).weekday() >= 5:
                continue
            await self._check_sector_strength()

    async def _check_sector_strength(self):
        """Fetch sector ETF bars, find the strongest, store symbols to prioritize."""
        etfs = list(UniverseConfig.SECTOR_ETFS.keys())
        start = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            from alpaca.data.timeframe import TimeFrame
            loop = asyncio.get_event_loop()
            bars = await loop.run_in_executor(
                None, lambda: self.alpaca.get_bars(etfs, TimeFrame.Day, start)
            )
            best_etf, best_change = "", -999.0
            for etf, bar_list in bars.items():
                if len(bar_list) < 2:
                    continue
                pct = (bar_list[-1]["close"] - bar_list[-2]["close"]) / bar_list[-2]["close"] * 100
                if pct > best_change:
                    best_change, best_etf = pct, etf
            if best_etf:
                self._strongest_sector = best_etf
                self._sector_symbols   = UniverseConfig.SECTOR_ETFS.get(best_etf, [])
                self.log(
                    f"Sector strength: {best_etf} leading (+{best_change:.2f}%) | "
                    f"priority symbols: {self._sector_symbols}"
                )
        except Exception as e:
            self.log(f"Sector check error: {e}", "warning")

    # ── Portfolio refresh ──────────────────────────────────────────────────────

    async def _refresh_portfolio(self):
        """Read real portfolio value from Alpaca."""
        try:
            loop    = asyncio.get_event_loop()
            account = await loop.run_in_executor(None, self.alpaca.get_account)
            self._portfolio_value = float(account["portfolio_value"])
            self.log(f"Portfolio value refreshed: ${self._portfolio_value:,.2f}")
        except Exception as e:
            self.log(f"Portfolio refresh error: {e} — keeping ${self._portfolio_value:,.2f}", "warning")

    # ── Regime + Circuit Breaker scale ────────────────────────────────────────

    async def _refresh_scales(self):
        """Read regime allocation scale, circuit breaker scale, and market gate from Redis."""
        try:
            regime_state = await self.bus.get_state("bot8:regime") or {}
            self._regime_scale = float(regime_state.get("scale", 1.0))
            if regime_state.get("regime") == "crash":
                self._regime_scale = 0.0
        except Exception:
            pass
        try:
            cb_state = await self.bus.get_state("bot8:cb_state") or {}
            cb_scale = float(cb_state.get("position_scale", 1.0))
            if not cb_state.get("trading_allowed", True):
                cb_scale = 0.0
            self._cb_scale = cb_scale
        except Exception:
            pass
        # Market gate: EOD window or Friday wind-down blocks new trades
        try:
            gate = await self.bus.get_state("bot8:market_gate") or {}
            if not gate.get("allow_new_trades", True):
                self._cb_scale = 0.0   # zero out so decision_cycle exits early
        except Exception:
            pass

    # ── Position sizing ────────────────────────────────────────────────────────

    def _safe_position_size(self, entry, stop_loss) -> float:
        """
        Size based on real Alpaca portfolio value, scaled by regime + circuit breaker.
          max_position = 15% portfolio × regime_scale × cb_scale, cap $1,000
          max_risk     = 1% portfolio
          shares       = int(max_risk / stop_distance)
          position_usd = min(shares * entry, max_position)
        """
        pv = self._portfolio_value if self._portfolio_value > 0 else 1000.0

        # Combined scale: regime allocation × circuit breaker
        combined_scale = self._regime_scale * self._cb_scale
        combined_scale = max(0.0, min(combined_scale, 1.0))  # clamp [0, 1]

        max_position_usd = (
            min(pv * 0.15, RiskConfig.MAX_SINGLE_POSITION_USD) * combined_scale
        )
        max_risk_usd = pv * 0.01

        entry     = float(entry or 0)
        stop_loss = float(stop_loss or 0)

        if entry <= 0 or stop_loss <= 0:
            return round(max_risk_usd * combined_scale, 2)

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return round(max_risk_usd * combined_scale, 2)

        shares       = int(max_risk_usd / stop_distance)
        position_usd = min(shares * entry, max_position_usd)
        return round(position_usd, 2)

    # ── Signal aggregation ─────────────────────────────────────────────────────

    def _aggregate_signals(self) -> dict:
        """Build a compact multi-signal view for Sonnet to reason over."""

        # Top momentum candidates
        top_momentum = sorted(
            [s for s in self._momentum_signals if s.get("direction") != "neutral"],
            key=lambda x: x.get("score") or 0.5,
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
        """Use Sonnet (compressed prompt, cached) to synthesise signals into trade setups."""

        PV       = self._portfolio_value if self._portfolio_value > 0 else 1000.0
        MAX_POS  = min(round(PV * 0.15, 0), RiskConfig.MAX_SINGLE_POSITION_USD)
        MAX_RISK = round(PV * 0.01, 0)

        # Compress signals to ~60-80 tokens (no raw history, no indented JSON)
        compressed = LLMRouter.compress_signals(signals)
        sector_tag = f"sector={self._strongest_sector}" if self._strongest_sector else ""
        regime_tag = f"regime_scale={self._regime_scale:.0%} cb={self._cb_scale:.0%}"

        # Cache key: hash of compressed signals + sizing params (not time-sensitive data)
        ck = LLMRouter.cache_key(compressed, int(MAX_POS), sector_tag)

        prompt = (
            f"Expert trader. Output 1-3 setups. "
            f"Acct=${PV:.0f} MaxPos=${MAX_POS:.0f} MaxRisk=${MAX_RISK:.0f} "
            f"SL=1% TP=2% conf>={RiskConfig.CONFIDENCE_THRESHOLD}. "
            + (f"{sector_tag} " if sector_tag else "")
            + f"{regime_tag}.\n"
            f"Signals:{LLMRouter.j(compressed)}\n"
            "JSON:{\"setups\":[{\"symbol\":\"X\",\"direction\":\"long\","
            "\"entry_price\":0,\"confidence\":0.7,\"timeframe\":\"intraday\","
            "\"thesis\":\"\",\"required_confirmations\":[]}],"
            "\"market_bias\":\"neutral\",\"notes\":\"\"}"
        )

        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="sonnet",
            max_tokens=600,
            cache_key=ck,
        )
        if not raw:
            return [], "neutral", ""

        try:
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            setups = parsed.get("setups", [])

            # ALWAYS overwrite stop/take-profit and position size — never trust Claude's values
            valid_setups = []
            for s in setups:
                entry     = float(s.get("entry_price") or 0)
                direction = (s.get("direction") or "long").lower()

                # Skip setups with no entry price — nothing to calculate from
                if entry <= 0:
                    self.log(f"Dropped setup for {s.get('symbol')}: entry_price=0", "warning")
                    continue

                # 2:1 reward-to-risk — always calculated, never from AI
                if direction == "long":
                    s["stop_loss"]   = round(entry * 0.99, 4)   # 1% risk
                    s["take_profit"] = round(entry * 1.02, 4)   # 2% reward
                else:  # short
                    s["stop_loss"]   = round(entry * 1.01, 4)   # 1% risk
                    s["take_profit"] = round(entry * 0.98, 4)   # 2% reward

                # Recalculate position size using the corrected stop_loss
                ai_size = s.get("position_size_usd", "?")
                s["position_size_usd"] = self._safe_position_size(
                    entry,
                    s["stop_loss"],
                )

                # Final guard — never send a setup missing critical fields
                if not s.get("stop_loss") or not s.get("take_profit"):
                    self.log(f"Dropped setup for {s.get('symbol')}: sl/tp still zero", "warning")
                    continue

                self.log(
                    f"Setup enforced: {s.get('symbol')} {direction} "
                    f"entry={entry} sl={s['stop_loss']} tp={s['take_profit']} "
                    f"size AI=${ai_size} → ${s['position_size_usd']}"
                )
                valid_setups.append(s)

            setups = valid_setups
            return setups, parsed.get("market_bias", "neutral"), parsed.get("notes", "")
        except Exception as e:
            self.log(f"Strategy generation error: {e}", "error")
            return [], "neutral", ""

    # ── Decision cycle ─────────────────────────────────────────────────────────

    async def _decision_cycle(self):
        if not self._market_data:
            self.log("No market data yet")
            return

        await self._refresh_portfolio()
        await self._refresh_scales()

        # CB scale or market gate set to 0 → no new trades this cycle
        if self._cb_scale == 0.0:
            self.log("New trades blocked (circuit breaker or market gate). Skipping cycle.", "warning")
            return

        signals = self._aggregate_signals()

        self.log(
            f"Decision cycle | momentum={len(self._momentum_signals)} "
            f"news={len(self._news_signals)} options={len(self._options_signals)}"
        )

        # Require at least one signal type — momentum alone is sufficient to trade
        if not signals["momentum"] and not signals["options"] and not signals["news"]:
            self.log(
                "No signals available yet — waiting for momentum/news. "
                "Will retry when bot03 publishes next scan or market data arrives."
            )
            return

        result = await self._generate_trade_setups(signals)
        setups, market_bias, notes = result

        if not setups:
            self.log(f"No setups generated | market_bias={market_bias}")
        else:
            pv = self._portfolio_value if self._portfolio_value > 0 else 1000.0
            max_pos_cap = min(pv * 0.15, RiskConfig.MAX_SINGLE_POSITION_USD)
            for setup in setups:
                # Final hard cap — catches any edge case before hitting the wire
                setup["position_size_usd"] = min(
                    float(setup.get("position_size_usd") or pv * 0.01), max_pos_cap
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

        # Keep the last 5 of each buffer as carry-forward so signals that arrived
        # during this cycle aren't silently dropped before the next run
        self._news_signals     = self._news_signals[-5:]
        self._momentum_signals = self._momentum_signals[-5:]
        self._options_signals  = self._options_signals[-5:]

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
