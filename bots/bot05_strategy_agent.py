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
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import anthropic

import pytz
from config import Models, RedisConfig, RiskConfig, AnthropicConfig, UniverseConfig, TradingWindowConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient
from shared.llm_router import LLMRouter
from shared.mtf_analyzer import MultiTimeframeAnalyzer

MARKET_TZ = pytz.timezone("America/New_York")


DECISION_DEBOUNCE = 120  # min seconds between event-driven decision cycles (2 min)


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
        self._mtf = MultiTimeframeAnalyzer(self.alpaca)

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
        self._current_regime: str            = "neutral"  # from bot08 regime state
        self._premarket_scores: dict[str, int] = {}       # symbol → 1-100 score from bot04
        self._daily_watchlist: list[str]     = []         # top 10 active from bot04 daily scan
        self._ai_consecutive_failures: int   = 0         # consecutive Sonnet JSON parse failures

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA,  self._on_market_data)
        await self.bus.subscribe(RedisConfig.CHANNEL_NEWS,         self._on_news)
        await self.bus.subscribe(RedisConfig.CHANNEL_MOMENTUM,     self._on_momentum)
        await self.bus.subscribe(RedisConfig.CHANNEL_OPTIONS_FLOW, self._on_options)
        asyncio.create_task(self.bus.listen())
        asyncio.create_task(self._morning_sector_task())
        await self._refresh_portfolio()
        self.log(
            f"Strategy Agent starting | portfolio=${self._portfolio_value:,.2f} | "
            f"subscribed to: market_data='{RedisConfig.CHANNEL_MARKET_DATA}' "
            f"news='{RedisConfig.CHANNEL_NEWS}' "
            f"momentum='{RedisConfig.CHANNEL_MOMENTUM}' "
            f"options='{RedisConfig.CHANNEL_OPTIONS_FLOW}'"
        )

    async def run(self):
        # Periodic fallback: if pub/sub goes silent, force a cycle every scan interval.
        # Event-driven triggers (momentum/market-data) run independently via debounce.
        while self.running:
            await asyncio.sleep(UniverseConfig.SCAN_INTERVAL_SECONDS)
            now  = datetime.now(timezone.utc)
            last = self._last_decision
            silent_secs = (now - last).total_seconds() if last else 9999
            if silent_secs >= UniverseConfig.SCAN_INTERVAL_SECONDS * 2:
                self.log(
                    f"Periodic fallback: no cycle in {silent_secs:.0f}s — "
                    "forcing decision cycle (pub/sub may be stale)"
                )
                try:
                    await self._decision_cycle()
                except Exception as e:
                    self.log(f"Periodic decision cycle error: {e}", "error")

    async def cleanup(self):
        self.log("Strategy Agent stopped")

    # ── Signal ingestion ───────────────────────────────────────────────────────

    async def _on_market_data(self, data: dict):
        sym = data.get("symbol")
        if sym:
            self._market_data[sym] = data
        # Only trigger a cycle when there are actual signals to analyse.
        # Firing on empty buffers consumes the debounce window before bot03's
        # signals arrive, then blocks the momentum-triggered cycle for 120 s.
        has_signals = bool(
            self._momentum_signals or self._news_signals or self._options_signals
        )
        if not has_signals:
            return
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
            self._ai_consecutive_failures = 0   # re-enable Claude each morning
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
            self._regime_scale    = float(regime_state.get("scale", 1.0))
            self._current_regime  = regime_state.get("regime", "neutral")
            if self._current_regime == "crash":
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

    def _regime_sl_tp(self) -> tuple[float, float]:
        """Return (stop_loss_pct, take_profit_pct) adjusted for current regime.
        Euphoria tightens stops so max loss on a $3k position stays around $10."""
        if self._current_regime == "euphoria":
            return 0.005, 0.01  # very tight — protect gains in overheated market
        return RiskConfig.STOP_LOSS_PCT, RiskConfig.TAKE_PROFIT_PCT  # 1.5%, 3%

    def _in_trading_window(self) -> bool:
        """True if current EST time is in the all-day trading window (9:35am–3:45pm)."""
        from datetime import time as dt_time
        t     = datetime.now(MARKET_TZ).time()
        open_ = dt_time(*TradingWindowConfig.OPEN)
        close = dt_time(*TradingWindowConfig.CLOSE)
        return open_ <= t <= close

    def _get_symbol_grade(self, sym: str) -> str:
        """Return the best momentum grade seen for a symbol from recent signals."""
        order = {"A+": 6, "A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        best, best_rank = "", 0
        for m in self._momentum_signals[-30:]:
            if m.get("symbol") == sym:
                g = m.get("grade", "")
                if order.get(g, 0) > best_rank:
                    best_rank, best = order.get(g, 0), g
        return best

    def _grade_position_size(self, grade: str) -> float:
        """Map grade → base position size in USD (MTF grades: A+=6, A=5, B=4, C=3)."""
        if grade in ("A", "A+"):
            return RiskConfig.GRADE_A_POSITION_USD   # $3,000
        if grade == "B":
            return RiskConfig.GRADE_B_POSITION_USD   # $2,000
        if grade == "C":
            return RiskConfig.GRADE_C_POSITION_USD   # $1,000
        return RiskConfig.MIN_SINGLE_POSITION_USD    # $1,500 floor

    def _safe_position_size(self, entry, stop_loss, grade: str = "") -> float:
        """
        Grade-based position sizing scaled by regime × CB multiplier.
          Grade A  → $3,000 base
          Grade B  → $2,000 base
          Default  → $1,500 floor
        Max loss per trade capped at RiskConfig.MAX_TRADE_LOSS_USD ($75).
        """
        combined_scale   = max(0.0, min(self._regime_scale * self._cb_scale, 1.0))
        base_usd         = self._grade_position_size(grade)
        max_position_usd = max(
            base_usd * combined_scale,
            RiskConfig.MIN_SINGLE_POSITION_USD * combined_scale,
        )
        max_risk_usd = RiskConfig.MAX_TRADE_LOSS_USD   # $75 hard cap

        entry     = float(entry or 0)
        stop_loss = float(stop_loss or 0)

        if entry <= 0 or stop_loss <= 0:
            return round(max_position_usd, 2)

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return round(max_position_usd, 2)

        # Verify the position doesn't risk more than the hard cap
        implied_shares = int(max_position_usd / entry)
        if implied_shares > 0 and implied_shares * stop_distance > max_risk_usd:
            implied_shares = int(max_risk_usd / stop_distance)

        position_usd = min(implied_shares * entry, max_position_usd)
        return round(max(position_usd, 0), 2)

    # ── Premarket scores ───────────────────────────────────────────────────────

    async def _load_premarket_scores(self):
        """Pull bot04's 8am premarket scores and daily watchlist into local cache."""
        try:
            scan = await self.bus.get_state("bot4:premarket_scan") or {}
            scored = scan.get("scored", [])
            self._premarket_scores = {s["symbol"]: s["score"] for s in scored}
        except Exception as e:
            self.log(f"Premarket score load error: {e}", "warning")
        try:
            wl = await self.bus.get_state("bot4:daily_watchlist") or {}
            top10 = wl.get("top_10_active", [])
            if top10:
                self._daily_watchlist = top10
                self.log(
                    f"Daily watchlist loaded: {top10} | "
                    f"sector_leaders={wl.get('sector_leaders', [])} | "
                    f"earnings={wl.get('earnings_plays', [])}"
                )
        except Exception as e:
            self.log(f"Daily watchlist load error: {e}", "warning")

    # ── Winner criteria gate ───────────────────────────────────────────────────

    def _check_winner_criteria(self, setup: dict) -> dict:
        """
        Evaluate the 6 winner criteria for a proposed setup.
        Returns {"passed": bool, "criteria_met": int, "reasons": [], "failures": []}

        Criteria:
          1. Premarket stock score >= 70/100
          2. News or momentum catalyst exists
          3. Momentum confirmed (3 of 5 checks pass)
          4. In the day's strongest sector (or bear ETF in bear regime)
          5. Market regime permits this trade direction
          6. AI confidence >= 80%
        """
        sym        = setup.get("symbol", "")
        confidence = float(setup.get("confidence") or 0)
        direction  = (setup.get("direction") or "long").lower()

        criteria_met = 0
        reasons  = []
        failures = []

        # ── Hard gate 1: Grade A or B only (C/D/F = skip) ────────────────────
        grade = setup.get("grade") or self._get_symbol_grade(sym)
        if grade and grade not in ("A", "A+", "B", ""):
            return {
                "passed": False, "criteria_met": 0, "stock_score": 0,
                "reasons": [], "failures": [f"grade_{grade}_blocked(need_A_or_B)"],
            }

        # ── Hard gate 2: Volume must be 2× average ────────────────────────────
        md  = self._market_data.get(sym, {})
        ind = md.get("indicators", {}) if md else {}
        vol_ratio = ind.get("volume_ratio")
        if vol_ratio is not None and vol_ratio < RiskConfig.VOLUME_CONFIRMATION_X:
            return {
                "passed": False, "criteria_met": 0, "stock_score": 0,
                "reasons": [], "failures": [
                    f"volume={vol_ratio:.1f}x<{RiskConfig.VOLUME_CONFIRMATION_X:.0f}x_required"
                ],
            }

        # ── 1. Premarket score ────────────────────────────────────────────────
        stock_score = self._premarket_scores.get(sym, 0)
        if stock_score == 0:
            # No premarket scan data yet — give benefit of the doubt
            criteria_met += 1
            reasons.append("score=unscored(pass)")
        elif stock_score >= RiskConfig.MIN_STOCK_SCORE:
            criteria_met += 1
            reasons.append(f"score={stock_score}/100")
        else:
            failures.append(f"score={stock_score}<{RiskConfig.MIN_STOCK_SCORE}")

        # ── 2. Catalyst (news bullish OR strong momentum grade A/B) ──────────
        has_catalyst = False
        catalyst_desc = ""
        for n in self._news_signals[-20:]:
            if sym in (n.get("symbols") or []) and n.get("sentiment") == "bullish":
                has_catalyst = True
                catalyst_desc = n.get("catalyst") or n.get("title", "")[:40] or "news_bullish"
                break
        if not has_catalyst:
            for m in self._momentum_signals[-20:]:
                if m.get("symbol") == sym and m.get("grade") in ("A", "B", "A+"):
                    has_catalyst = True
                    catalyst_desc = f"momentum_grade={m['grade']}"
                    break
        if not has_catalyst and self._strongest_sector:
            # Sector rotation is itself a valid catalyst
            if sym in self._sector_symbols:
                has_catalyst = True
                catalyst_desc = f"sector_rotation={self._strongest_sector}"
        if has_catalyst:
            criteria_met += 1
            reasons.append(f"catalyst:{catalyst_desc[:30]}")
            setup.setdefault("catalyst", catalyst_desc)
        else:
            failures.append("no_catalyst")

        # ── 3. Momentum confirmation (3 of 5 checks) ─────────────────────────
        md  = self._market_data.get(sym, {})
        ind = md.get("indicators", {}) if md else {}
        price = md.get("price", 0) or 0
        mom_checks = 0

        if price and ind.get("sma_20") and price > ind["sma_20"]:
            mom_checks += 1  # above 20-day MA
        if ind.get("volume_ratio") and ind["volume_ratio"] > 1.0:
            mom_checks += 1  # above average volume
        rsi = ind.get("rsi_14")
        if rsi and 45 <= rsi <= 70:
            mom_checks += 1  # RSI in bullish range
        macd = ind.get("macd") or {}
        if macd.get("histogram") and macd["histogram"] > 0:
            mom_checks += 1  # MACD bullish
        vwap = ind.get("vwap")
        if vwap and price and price > vwap:
            mom_checks += 1  # price above VWAP (up on the day)

        if mom_checks >= 3:
            criteria_met += 1
            reasons.append(f"momentum={mom_checks}/5")
        else:
            failures.append(f"momentum={mom_checks}/5<3")

        # ── 4. Right sector ───────────────────────────────────────────────────
        if self._current_regime in ("bear", "crash"):
            if sym in UniverseConfig.BEAR_ETFS:
                criteria_met += 1
                reasons.append("bear_etf_sector")
            else:
                failures.append("bear_regime_no_longs")
        elif not self._strongest_sector or sym in self._sector_symbols:
            criteria_met += 1
            reasons.append(f"sector={self._strongest_sector or 'any'}")
        elif sym in ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "SPY", "QQQ"):
            # Mega-caps allowed regardless of sector rotation
            criteria_met += 1
            reasons.append("mega_cap_pass")
        else:
            failures.append(f"wrong_sector(want={self._strongest_sector})")

        # ── 5. Regime allows this trade direction ─────────────────────────────
        if self._current_regime == "crash":
            failures.append("crash_all_blocked")
        elif self._current_regime == "bear":
            if direction == "long" and sym not in UniverseConfig.BEAR_ETFS:
                failures.append("bear_blocks_long_non_inverse")
            else:
                criteria_met += 1
                reasons.append("regime=bear_inverse_ok")
        elif self._current_regime == "neutral":
            # Neutral: only trade if score is high (85+) OR already has 3 other criteria
            if stock_score >= 85 or criteria_met >= 3:
                criteria_met += 1
                reasons.append(f"regime=neutral_ok(score={stock_score})")
            else:
                failures.append(f"neutral_requires_score85+(got={stock_score})")
        else:
            criteria_met += 1
            reasons.append(f"regime={self._current_regime}_ok")

        # ── 6. AI confidence ──────────────────────────────────────────────────
        if confidence >= RiskConfig.CONFIDENCE_THRESHOLD:
            criteria_met += 1
            reasons.append(f"conf={confidence:.0%}")
        else:
            failures.append(f"conf={confidence:.0%}<{RiskConfig.CONFIDENCE_THRESHOLD:.0%}")

        passed = criteria_met >= RiskConfig.MIN_WINNER_CRITERIA
        return {
            "passed":       passed,
            "criteria_met": criteria_met,
            "stock_score":  stock_score,
            "reasons":      reasons,
            "failures":     failures,
        }

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
        """Synthesise trade setups from live signals.
        Primary model: Haiku (fast, 30 s timeout).
        Retry model:   Sonnet (higher quality if Haiku parse fails).
        Never freezes: consecutive-failure bypass after 3 failures."""

        sl_pct, tp_pct = self._regime_sl_tp()

        # After 3 consecutive parse failures skip Claude — go straight to fallback
        if self._ai_consecutive_failures >= 3:
            self.log(
                f"Strategy AI bypassed: {self._ai_consecutive_failures} consecutive parse failures "
                "— using momentum fallback to protect trading cycle", "warning"
            )
            return self._momentum_fallback_setups(), "neutral", "ai_bypassed_consecutive_failures"

        sector_tag = f"sector={self._strongest_sector}" if self._strongest_sector else ""
        regime_tag = f"regime={self._current_regime} scale={self._regime_scale:.0%} cb={self._cb_scale:.0%}"

        # Bear/crash: restrict to inverse ETFs only
        bear_note = ""
        if self._current_regime in ("bear", "crash"):
            bear_note = f"BEAR REGIME: ONLY suggest inverse ETFs {UniverseConfig.BEAR_ETFS}. Long only on these."

        # Pass full signal context — Claude needs the actual price to generate a
        # valid entry_price. Never compress: stripping price_context caused stale prices.
        prompt = (
            f"You are an expert intraday trader. Analyse the signals below and output 1-3 high-conviction trade setups.\n"
            f"Rules:\n"
            f"  - Only Grade A (${RiskConfig.GRADE_A_POSITION_USD:,.0f} position) or Grade B (${RiskConfig.GRADE_B_POSITION_USD:,.0f} position) signals\n"
            f"  - Stop loss: {sl_pct*100:.1f}% | Take profit: {tp_pct*100:.1f}% | Minimum RR: 2:1\n"
            f"  - Grade A profit target >= ${RiskConfig.MIN_GRADE_A_PROFIT_USD:.0f} | Grade B >= ${RiskConfig.MIN_GRADE_B_PROFIT_USD:.0f}\n"
            f"  - Minimum confidence: {RiskConfig.CONFIDENCE_THRESHOLD:.0%}\n"
            f"  - entry_price MUST equal the current price shown in price_context for that symbol\n"
            f"  - Each setup MUST include a specific catalyst (earnings beat/upgrade/breakout/sector rotation)\n"
            + (f"  - Strongest sector today: {sector_tag}\n" if sector_tag else "")
            + f"  - {regime_tag}\n"
            + (f"  - {bear_note}\n" if bear_note else "")
            + f"\nFull market signals:\n{LLMRouter.j(signals)}\n\n"
            "Required JSON (fill in real values):\n"
            "{\"setups\":[{\"symbol\":\"TICKER\",\"direction\":\"long\","
            "\"entry_price\":0.0,\"confidence\":0.85,\"grade\":\"A\","
            "\"timeframe\":\"intraday\",\"thesis\":\"one sentence why\","
            "\"catalyst\":\"specific catalyst\","
            "\"required_confirmations\":[]}],"
            "\"market_bias\":\"neutral\",\"notes\":\"\"}"
        )

        # ── Primary: Haiku (fast, avoids Sonnet timeout issues) ─────────────────
        raw = await self._router.call(
            [{"role": "user", "content": prompt}],
            prefer="haiku",
            max_tokens=800,
            timeout=30.0,
        )
        if not raw:
            self._ai_consecutive_failures += 1
            self.log(
                f"Haiku empty/timeout (failure #{self._ai_consecutive_failures}) "
                "— trying Sonnet", "warning"
            )
            # One Sonnet attempt before giving up on AI entirely
            raw = await self._router.call(
                [{"role": "user", "content": prompt}],
                prefer="sonnet",
                max_tokens=800,
                timeout=30.0,
            )
        if not raw:
            self._ai_consecutive_failures += 1
            self.log(
                f"Sonnet also empty/timeout (failure #{self._ai_consecutive_failures}) "
                "— using momentum fallback", "warning"
            )
            return self._momentum_fallback_setups(), "neutral", "ai_timeout_momentum_fallback"

        parsed = self._parse_json_safe(raw)

        # ── Retry with Sonnet if Haiku parse failed ───────────────────────────
        if parsed is None:
            self.log(
                f"Haiku parse failed (failure #{self._ai_consecutive_failures + 1}) "
                "— retrying with Sonnet", "warning"
            )
            retry_prompt = (
                "Required format (fill in real values):\n"
                "{\"setups\":[{\"symbol\":\"TICKER\",\"direction\":\"long\","
                "\"entry_price\":0.0,\"confidence\":0.85,\"grade\":\"A\","
                "\"timeframe\":\"intraday\",\"thesis\":\"reason\","
                "\"catalyst\":\"catalyst\",\"required_confirmations\":[]}],"
                "\"market_bias\":\"neutral\",\"notes\":\"\"}"
            )
            raw2 = await self._router.call(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": raw},
                 {"role": "user", "content": retry_prompt}],
                prefer="sonnet",
                max_tokens=600,
                timeout=30.0,
            )
            parsed = self._parse_json_safe(raw2) if raw2 else None

        # ── Fallback: build setup from top momentum signal ────────────────────
        if parsed is None:
            self._ai_consecutive_failures += 1
            self.log(
                f"Parse failed after Haiku+Sonnet retry (failure #{self._ai_consecutive_failures}) "
                "— using momentum fallback", "warning"
            )
            fallback = self._momentum_fallback_setups()
            return fallback, "neutral", "parse_error_momentum_fallback"

        try:
            self._ai_consecutive_failures = 0   # reset streak — parse succeeded
            setups = parsed.get("setups", [])

            # ALWAYS overwrite stop/take-profit and position size — never trust Claude's values
            valid_setups = []
            for s in setups:
                entry     = float(s.get("entry_price") or 0)
                direction = (s.get("direction") or "long").lower()

                if entry <= 0:
                    self.log(f"Dropped setup for {s.get('symbol')}: entry_price=0", "warning")
                    continue

                sl_pct, tp_pct = self._regime_sl_tp()
                if direction == "long":
                    s["stop_loss"]   = round(entry * (1 - sl_pct), 4)
                    s["take_profit"] = round(entry * (1 + tp_pct), 4)
                else:
                    s["stop_loss"]   = round(entry * (1 + sl_pct), 4)
                    s["take_profit"] = round(entry * (1 - tp_pct), 4)

                ai_size = s.get("position_size_usd", "?")
                setup_grade = s.get("grade") or self._get_symbol_grade(s.get("symbol", ""))
                s["grade"]             = setup_grade
                s["position_size_usd"] = self._safe_position_size(
                    entry, s["stop_loss"], grade=setup_grade
                )

                if not s.get("stop_loss") or not s.get("take_profit"):
                    self.log(f"Dropped setup for {s.get('symbol')}: sl/tp still zero", "warning")
                    continue

                self.log(
                    f"Setup enforced: {s.get('symbol')} {direction} "
                    f"entry={entry} sl={s['stop_loss']} tp={s['take_profit']} "
                    f"size AI=${ai_size} → ${s['position_size_usd']} | "
                    f"catalyst={s.get('catalyst','none')[:40]}"
                )
                valid_setups.append(s)

            return valid_setups, parsed.get("market_bias", "neutral"), parsed.get("notes", "")
        except Exception as e:
            self.log(f"Strategy post-parse error: {e}", "error")
            return [], "neutral", ""

    @staticmethod
    def _parse_json_safe(raw: str) -> Optional[dict]:
        """
        Robustly parse a JSON object from a Claude response.
        Handles: markdown code fences, control characters, trailing commas.
        Returns the parsed dict, or None if all attempts fail.
        """
        if not raw:
            return None

        text = raw

        # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

        # 2. Strip ASCII control characters that are illegal in JSON strings
        #    (keeps \t \n \r which json.loads handles, removes \x00-\x08, \x0b-\x0c, \x0e-\x1f)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

        # 3. Strip JS-style comments
        text = re.sub(r"//[^\n]*", "", text)

        # 4. Extract the outermost { ... } block
        start = text.find("{")
        if start == -1:
            return None
        depth, in_str, escape, end = 0, False, False, -1
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
        if end == -1:
            return None
        text = text[start:end + 1]

        # 5. First parse attempt — strict=False allows literal control chars in strings
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            pass

        # 6. Second attempt after removing trailing commas and collapsing newlines
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = re.sub(r"[\n\r]+", " ", text)
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            return None

    def _momentum_fallback_setups(self) -> list[dict]:
        """
        Build up to one trade setup from the best available momentum signal.
        Used when Sonnet returns unparseable JSON so trading never fully stops.
        """
        sl_pct, tp_pct = self._regime_sl_tp()
        candidates = sorted(
            self._momentum_signals[-30:],
            key=lambda x: float(x.get("score") or 0),
            reverse=True,
        )
        for m in candidates:
            sym   = m.get("symbol", "")
            grade = m.get("grade", "")
            if not sym or grade not in ("A", "A+", "B", ""):
                continue
            md    = self._market_data.get(sym, {})
            price = float(md.get("price") or 0)
            if price <= 0:
                continue
            setup = {
                "symbol":      sym,
                "direction":   "long",
                "entry_price": price,
                "confidence":  float(m.get("score") or 0.70),
                "grade":       grade,
                "timeframe":   "intraday",
                "thesis":      m.get("reason", "momentum signal"),
                "catalyst":    f"momentum_grade_{grade}",
                "required_confirmations": [],
                "stop_loss":   round(price * (1 - sl_pct), 4),
                "take_profit": round(price * (1 + tp_pct), 4),
            }
            setup["position_size_usd"] = self._safe_position_size(
                price, setup["stop_loss"], grade=grade
            )
            self.log(
                f"Momentum fallback setup: {sym} entry={price} "
                f"sl={setup['stop_loss']} tp={setup['take_profit']} grade={grade}"
            )
            return [setup]
        return []

    # ── Decision cycle ─────────────────────────────────────────────────────────

    async def _decision_cycle(self):
        if not self._market_data:
            self.log("No market data yet")
            return

        await self._refresh_portfolio()
        await self._refresh_scales()
        await self._load_premarket_scores()

        # CB scale or market gate set to 0 → no new trades this cycle
        if self._cb_scale == 0.0:
            self.log("New trades blocked (circuit breaker or market gate). Skipping cycle.", "warning")
            return

        # Time window: only trade during the configured all-day window
        if not self._in_trading_window():
            t = datetime.now(MARKET_TZ).time()
            self.log(
                f"Outside trading window ({t.strftime('%H:%M')} ET) — "
                f"window: {TradingWindowConfig.OPEN_STR}–{TradingWindowConfig.CLOSE_STR}. Skipping."
            )
            return

        # Daily P&L gates: stop at -$150 loss or +$100 profit
        try:
            dashboard = await self.bus.get_state("bot8:dashboard") or {}
            daily_pnl = float(dashboard.get("daily_pnl", 0))
            if daily_pnl <= -RiskConfig.DAILY_LOSS_LIMIT_USD:
                self.log(
                    f"Daily loss limit hit: ${daily_pnl:.2f} — no new trades today",
                    "warning",
                )
                return
            if daily_pnl >= RiskConfig.DAILY_PROFIT_TARGET_USD:
                self.log(
                    f"Daily profit target hit: +${daily_pnl:.2f} — locking in gains, no new trades",
                    "warning",
                )
                return
        except Exception:
            pass

        # Grade A/B filter only — watchlist is a SOFT preference for Sonnet, not a hard gate.
        # Previously the watchlist was used as a hard filter here, silently dropping every
        # signal for symbols outside the 10-symbol top_10_active list.  That was the
        # primary cause of momentum=0/N even when bot03 had just published fresh signals.
        _ALLOWED_GRADES = {"A", "A+", "B", ""}
        filtered_momentum = [
            m for m in self._momentum_signals
            if not m.get("grade") or m.get("grade") in _ALLOWED_GRADES
        ]

        signals = self._aggregate_signals()
        # Sort by score; watchlist symbols float to top via score boost
        in_wl = set(self._daily_watchlist)
        signals["momentum"] = sorted(
            [s for s in filtered_momentum if s.get("direction") != "neutral"],
            key=lambda x: (1 if x.get("symbol") in in_wl else 0,
                           x.get("score") or 0.5),
            reverse=True,
        )[:10]

        self.log(
            f"Decision cycle | regime={self._current_regime} | window=OK | "
            f"momentum={len(signals['momentum'])}/{len(self._momentum_signals)} sent-to-Sonnet | "
            f"news={len(self._news_signals)} | watchlist={self._daily_watchlist[:5]}"
        )

        # Require at least one signal type — momentum alone is sufficient to trade
        if not signals["momentum"] and not signals["options"] and not signals["news"]:
            self.log(
                "No signals available yet — waiting for momentum/news. "
                "Will retry when bot03 publishes next scan or market data arrives."
            )
            return

        # ── Multi-Timeframe Analysis ───────────────────────────────────────────
        # Run MTF scoring on all momentum candidates; skip symbols with < 3
        # criteria (Grade D) to prevent entering on weak multi-TF alignment.
        if signals["momentum"]:
            candidates = list({m["symbol"] for m in signals["momentum"] if m.get("symbol")})
            current_prices = {
                sym: self._market_data[sym]["price"]
                for sym in candidates
                if sym in self._market_data and self._market_data[sym].get("price")
            }
            try:
                mtf_results = await self._mtf.analyze_batch(candidates, current_prices)
            except Exception as e:
                self.log(f"MTF analysis error: {e} — proceeding without MTF filter", "warning")
                mtf_results = {}

            if mtf_results:
                pre_filter = len(signals["momentum"])
                signals["momentum"] = [
                    m for m in signals["momentum"]
                    if mtf_results.get(m.get("symbol", ""), {}).get(
                        "criteria_count", RiskConfig.MTF_MIN_CRITERIA
                    ) >= RiskConfig.MTF_MIN_CRITERIA
                ]
                # Attach MTF grade to each signal so Claude + winner criteria can use it
                for m in signals["momentum"]:
                    mtf = mtf_results.get(m.get("symbol", ""), {})
                    m["mtf_grade"]      = mtf.get("grade", "")
                    m["mtf_criteria"]   = mtf.get("criteria_count", 0)
                    m["fvg_active"]     = mtf.get("fvg_active", False)
                    m["asp_setup"]      = mtf.get("asp_setup", False)
                    m["trend_summary"]  = (
                        f"D:{mtf.get('trend_daily','?')} "
                        f"4H:{mtf.get('trend_4h','?')} "
                        f"1H:{mtf.get('trend_1h','?')} "
                        f"15M:{mtf.get('trend_15m','?')}"
                    )
                    # Override grade with MTF grade when it is stronger
                    if mtf.get("grade") and not m.get("grade"):
                        m["grade"] = mtf["grade"]

                signals["mtf_analysis"] = {
                    sym: {
                        "grade":       v.get("grade"),
                        "criteria":    v.get("criteria_count"),
                        "trends":      f"D:{v.get('trend_daily','?')} 4H:{v.get('trend_4h','?')} 1H:{v.get('trend_1h','?')} 15M:{v.get('trend_15m','?')}",
                        "fvg":         v.get("fvg_active"),
                        "asp":         v.get("asp_setup"),
                    }
                    for sym, v in mtf_results.items()
                }
                self.log(
                    f"MTF filter | {pre_filter} → {len(signals['momentum'])} signals "
                    f"(kept ≥{RiskConfig.MTF_MIN_CRITERIA} criteria) | "
                    f"grades={[m.get('mtf_grade') for m in signals['momentum']]}"
                )
                if not signals["momentum"]:
                    self.log(
                        "MTF filter removed all momentum signals — no symbols with "
                        f"≥{RiskConfig.MTF_MIN_CRITERIA} aligned timeframes",
                        "warning",
                    )
                    # Fall through: options/news signals may still produce setups

        # In bear/crash regime, inject inverse ETF data into momentum signals
        if self._current_regime in ("bear", "crash"):
            for etf in UniverseConfig.BEAR_ETFS:
                md = self._market_data.get(etf, {})
                if md:
                    signals["momentum"].insert(0, {
                        "symbol":    etf,
                        "direction": "long",
                        "score":     0.8,
                        "grade":     "B",
                        "reason":    f"bear_regime_inverse_etf",
                    })

        result = await self._generate_trade_setups(signals)
        setups, market_bias, notes = result

        if not setups:
            self.log(f"No setups generated | market_bias={market_bias}")
        else:
            approved  = []
            rejected  = []
            for setup in setups:
                # Final hard cap on position size
                setup["position_size_usd"] = min(
                    float(setup.get("position_size_usd") or RiskConfig.MAX_TRADE_LOSS_USD),
                    RiskConfig.MAX_SINGLE_POSITION_USD,
                )

                # Winner criteria gate (min 4 of 6)
                result = self._check_winner_criteria(setup)
                setup["winner_score"]    = result["criteria_met"]
                setup["winner_reasons"]  = result["reasons"]
                setup["winner_failures"] = result["failures"]

                if result["passed"]:
                    approved.append(setup)
                    self.log(
                        f"APPROVED {setup.get('symbol')} {setup.get('direction')} | "
                        f"criteria={result['criteria_met']}/6 | "
                        f"catalyst={setup.get('catalyst','?')[:40]} | "
                        f"{','.join(result['reasons'])}"
                    )
                else:
                    rejected.append(setup.get("symbol"))
                    self.log(
                        f"REJECTED {setup.get('symbol')} | "
                        f"criteria={result['criteria_met']}/6 < {RiskConfig.MIN_WINNER_CRITERIA} | "
                        f"failed: {','.join(result['failures'])}",
                        "warning",
                    )

            if rejected:
                self.log(f"Winner criteria rejected: {rejected}")

            for setup in approved:
                await self.publish(RedisConfig.CHANNEL_STRATEGY, {
                    "type":        "trade_setup",
                    "market_bias": market_bias,
                    **setup,
                })
                self.log(
                    f"Trade setup published: {setup.get('symbol')} {setup.get('direction')} | "
                    f"conf={setup.get('confidence')} | "
                    f"WHY: {setup.get('catalyst') or setup.get('thesis', '')[:80]}"
                )
            setups = approved

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

        # Keep signals as carry-forward; bot03 scans every 5 min so we retain
        # 20 momentum signals to bridge the gap between scans.
        self._news_signals     = self._news_signals[-10:]
        self._momentum_signals = self._momentum_signals[-20:]
        self._options_signals  = self._options_signals[-10:]

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
