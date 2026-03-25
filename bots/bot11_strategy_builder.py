"""
Bot 11 – Strategy Builder
Model  : Opus 4.6  (overnight deep analysis)
Role   : Designs, evolves, and curates trading strategies overnight.
         • Reads backtest results from Bot 10
         • Reads live trading performance from Redis
         • Uses Opus to generate new strategy variants and improvements
         • Maintains a strategy library (persisted in Redis)
         • Submits new strategies to Bot 10 for backtesting
         • Promotes/demotes strategies based on live + backtest performance
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

import anthropic

from config import Models, RedisConfig, AnthropicConfig, RiskConfig
from shared.base_bot import BaseBot


class StrategyBuilder(BaseBot):
    """
    Bot 11 – Strategy Builder
    Overnight strategy research and generation with Opus 4.6.
    """

    BOT_ID = 11
    NAME   = "Strategy Builder"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.OPUS)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self._strategy_library: list[dict] = []
        self._backtest_results: list[dict] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_BACKTEST, self._on_backtest_result)
        asyncio.create_task(self.bus.listen())

        # Load existing strategy library
        saved = await self.bus.get_state("bot11:strategies")
        if saved:
            self._strategy_library = saved
            self.log(f"Loaded {len(self._strategy_library)} strategies from library")
        else:
            self._strategy_library = self._seed_strategies()
            await self._save_library()

        self.log(f"Strategy Builder starting | library={len(self._strategy_library)} strategies")

    async def run(self):
        while self.running:
            await self._wait_for_overnight()
            try:
                await self._overnight_cycle()
            except Exception as e:
                self.log(f"Overnight cycle error: {e}", "error")

    async def cleanup(self):
        await self._save_library()
        self.log("Strategy Builder stopped")

    # ── Scheduling ─────────────────────────────────────────────────────────────

    async def _wait_for_overnight(self):
        import pytz
        et     = pytz.timezone("America/New_York")
        now    = datetime.now(et)
        target = now.replace(hour=23, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        self.log(f"Next strategy build cycle in {wait/3600:.1f} hours")
        await asyncio.sleep(wait)

    # ── Backtest result handler ────────────────────────────────────────────────

    async def _on_backtest_result(self, data: dict):
        if data.get("type") != "backtest_result":
            return
        self._backtest_results.append(data)
        if len(self._backtest_results) > 50:
            self._backtest_results = self._backtest_results[-50:]

        # Update strategy performance in library
        strategy_name = data.get("strategy_name")
        avg_metrics   = data.get("avg_metrics", {})
        for s in self._strategy_library:
            if s.get("name") == strategy_name:
                s["last_backtest"]   = avg_metrics
                s["last_backtest_ts"] = datetime.utcnow().isoformat()
                s["status"]          = self._evaluate_strategy_status(avg_metrics)
                break
        await self._save_library()

    def _evaluate_strategy_status(self, metrics: dict) -> str:
        sharpe = metrics.get("sharpe_ratio", 0)
        ret    = metrics.get("total_return_pct", 0)
        dd     = metrics.get("max_drawdown_pct", 100)
        if sharpe >= 1.5 and ret >= 5 and dd <= 10:
            return "approved"
        elif sharpe >= 0.8 and ret >= 2:
            return "candidate"
        elif ret < 0 or dd > 20:
            return "retired"
        else:
            return "pending"

    # ── Seed strategies ────────────────────────────────────────────────────────

    def _seed_strategies(self) -> list[dict]:
        """Initial strategy library to bootstrap the system."""
        return [
            {
                "name":              "momentum_rsi_macd",
                "description":       "Trend-following using RSI and MACD confirmation",
                "entry_rules":       ["rsi_14 > 50", "rsi_14 < 70", "macd_histogram > 0", "price > sma_50"],
                "exit_rules":        ["rsi_14 > 75", "macd_histogram < 0"],
                "stop_loss_pct":     0.02,
                "take_profit_pct":   0.04,
                "position_size_pct": 0.10,
                "status":            "pending",
                "created_at":        datetime.utcnow().isoformat(),
            },
            {
                "name":              "breakout_volume",
                "description":       "Price breakout with volume confirmation",
                "entry_rules":       ["price > bb_upper", "volume_ratio > 1.5", "rsi_14 > 55"],
                "exit_rules":        ["price < bb_mid", "rsi_14 > 80"],
                "stop_loss_pct":     0.025,
                "take_profit_pct":   0.05,
                "position_size_pct": 0.08,
                "status":            "pending",
                "created_at":        datetime.utcnow().isoformat(),
            },
            {
                "name":              "mean_reversion_oversold",
                "description":       "Buy oversold dips with RSI divergence",
                "entry_rules":       ["rsi_14 < 30", "price > sma_200", "macd_histogram > macd_histogram"],
                "exit_rules":        ["rsi_14 > 50", "price < sma_50"],
                "stop_loss_pct":     0.03,
                "take_profit_pct":   0.06,
                "position_size_pct": 0.08,
                "status":            "pending",
                "created_at":        datetime.utcnow().isoformat(),
            },
        ]

    # ── Overnight cycle ────────────────────────────────────────────────────────

    async def _overnight_cycle(self):
        self.log("Starting overnight strategy build cycle")

        # Gather performance data
        live_perf   = await self.bus.get_state("bot6:stats") or {}
        risk_stats  = await self.bus.get_state("bot8:dashboard") or {}
        recent_bts  = self._backtest_results[-5:] if self._backtest_results else []

        # 1. Evolve existing strategies
        evolved = await self._evolve_strategies(recent_bts, live_perf)

        # 2. Generate new strategy ideas
        new_strategies = await self._generate_new_strategies(live_perf, risk_stats)

        # 3. Prune retired strategies
        self._strategy_library = [s for s in self._strategy_library
                                   if s.get("status") != "retired"]

        # 4. Add new candidates (avoid duplicates)
        existing_names = {s["name"] for s in self._strategy_library}
        added = 0
        for s in evolved + new_strategies:
            if s["name"] not in existing_names:
                self._strategy_library.append(s)
                existing_names.add(s["name"])
                added += 1

        await self._save_library()

        # 5. Submit top candidates to Bot 10 for backtesting
        candidates = [s for s in self._strategy_library
                      if s.get("status") in ("pending", "candidate")][:3]
        for strategy in candidates:
            await self.publish(RedisConfig.CHANNEL_BACKTEST, {
                "type":     "backtest_request",
                "strategy": strategy,
                "symbols":  ["AAPL", "NVDA", "SPY", "QQQ", "MSFT"],
                "days":     90,
            })
            self.log(f"Submitted for backtest: {strategy['name']}")

        self.log(
            f"Overnight cycle complete | "
            f"library={len(self._strategy_library)} | added={added} | "
            f"submitted={len(candidates)} for backtest"
        )

    # ── Strategy evolution ─────────────────────────────────────────────────────

    async def _evolve_strategies(self, backtest_results: list[dict],
                                  live_perf: dict) -> list[dict]:
        """Use Opus to evolve existing strategies based on performance data."""
        if not self._strategy_library:
            return []

        approved = [s for s in self._strategy_library if s.get("status") == "approved"]
        candidates = [s for s in self._strategy_library if s.get("status") == "candidate"]
        existing   = (approved + candidates)[:5]

        prompt = (
            "You are a quantitative strategy researcher. Evolve these trading strategies "
            "based on backtest performance.\n\n"
            f"EXISTING STRATEGIES:\n{json.dumps(existing, indent=2)}\n\n"
            f"RECENT BACKTEST RESULTS:\n{json.dumps(backtest_results, indent=2)}\n\n"
            f"LIVE TRADING PERFORMANCE:\n{json.dumps(live_perf, indent=2)}\n\n"
            "For each strategy that shows weakness, create an evolved version that:\n"
            "1. Addresses the specific failure mode (e.g. add filter for choppy markets)\n"
            "2. Tightens entry/exit conditions if losing too much\n"
            "3. Adds confirmation signals to reduce false signals\n"
            "4. Adjusts position sizing if drawdown is too high\n\n"
            "Rules for strategy format:\n"
            "  - entry_rules/exit_rules: simple comparisons like 'rsi_14 > 50'\n"
            "  - available indicators: rsi_14, macd_histogram, sma_20, sma_50, sma_200, "
            "bb_upper, bb_lower, bb_mid, atr_14, price, volume_ratio\n"
            "  - stop_loss_pct: 0.01 to 0.05\n"
            "  - take_profit_pct: must be > stop_loss_pct\n\n"
            "Respond ONLY with JSON:\n"
            "{\"evolved\": [{\"name\": \"strategy_name_v2\", \"description\": \"...\", "
            "\"entry_rules\": [], \"exit_rules\": [], \"stop_loss_pct\": 0.02, "
            "\"take_profit_pct\": 0.04, \"position_size_pct\": 0.10, "
            "\"parent\": \"original_name\", \"changes\": \"what changed and why\"}]}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            evolved = []
            for s in parsed.get("evolved", []):
                s["status"]     = "pending"
                s["created_at"] = datetime.utcnow().isoformat()
                evolved.append(s)
            self.log(f"Evolved {len(evolved)} strategies")
            return evolved
        except Exception as e:
            self.log(f"Strategy evolution error: {e}", "error")
            return []

    # ── New strategy generation ────────────────────────────────────────────────

    async def _generate_new_strategies(self, live_perf: dict,
                                        market_state: dict) -> list[dict]:
        """Use Opus to generate entirely new strategy ideas."""
        market_bias = market_state.get("commander_notes", "")
        session     = market_state.get("market_session", "unknown")

        prompt = (
            "You are a quantitative strategy researcher generating NEW trading strategy ideas.\n\n"
            f"Current market context: {market_bias}\n"
            f"Market session: {session}\n"
            f"Recent live performance: {json.dumps(live_perf, indent=2)}\n\n"
            "Generate 2 creative but realistic new strategies. Consider:\n"
            "• Volatility-based strategies (using ATR)\n"
            "• Sector rotation patterns\n"
            "• Opening range breakouts\n"
            "• Gap fill strategies\n"
            "• Multi-timeframe confluence\n\n"
            "Rules:\n"
            "  - entry_rules/exit_rules use indicators: rsi_14, macd_histogram, "
            "sma_20, sma_50, sma_200, bb_upper, bb_lower, bb_mid, atr_14, price\n"
            "  - Keep stop_loss_pct between 0.015 and 0.04\n"
            "  - take_profit_pct >= 1.5 * stop_loss_pct\n\n"
            "Respond ONLY with JSON:\n"
            "{\"strategies\": [{\"name\": \"unique_name\", \"description\": \"...\", "
            "\"rationale\": \"why this should work\", \"entry_rules\": [], "
            "\"exit_rules\": [], \"stop_loss_pct\": 0.02, \"take_profit_pct\": 0.04, "
            "\"position_size_pct\": 0.08}]}"
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
            new_strats = []
            for s in parsed.get("strategies", []):
                s["status"]     = "pending"
                s["created_at"] = datetime.utcnow().isoformat()
                new_strats.append(s)
            self.log(f"Generated {len(new_strats)} new strategies")
            return new_strats
        except Exception as e:
            self.log(f"Strategy generation error: {e}", "error")
            return []

    # ── Library persistence ────────────────────────────────────────────────────

    async def _save_library(self):
        await self.bus.set_state("bot11:strategies", self._strategy_library, ttl=86400 * 7)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = StrategyBuilder()
    asyncio.run(bot.start())
