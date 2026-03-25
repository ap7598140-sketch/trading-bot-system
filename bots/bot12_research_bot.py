"""
Bot 12 – Research Bot
Model  : Opus 4.6  (overnight deep analysis)
Role   : Deep market research and intelligence gathering.
         • Runs overnight to avoid competing with live trading
         • Researches macro conditions, sector themes, earnings calendar
         • Analyses correlations, market regimes, and regime changes
         • Builds a research knowledge base in Redis
         • Publishes research reports to CHANNEL_RESEARCH
         • Feeds insights to Bot 11 (Strategy Builder) and Bot 8 (Commander)
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import aiohttp

from config import Models, RedisConfig, AnthropicConfig, AlertConfig, UniverseConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient


class ResearchBot(BaseBot):
    """
    Bot 12 – Research Bot
    Overnight market intelligence with Opus 4.6.
    """

    BOT_ID = 12
    NAME   = "Research Bot"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.OPUS)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.alpaca = AlpacaClient()
        self._research_cache: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        self.log("Research Bot starting (Opus 4.6 – overnight mode)")

    async def run(self):
        while self.running:
            await self._wait_for_overnight()
            try:
                await self._overnight_research_cycle()
            except Exception as e:
                self.log(f"Research cycle error: {e}", "error")

    async def cleanup(self):
        self.log("Research Bot stopped")

    # ── Scheduling ─────────────────────────────────────────────────────────────

    async def _wait_for_overnight(self):
        import pytz
        et     = pytz.timezone("America/New_York")
        now    = datetime.now(et)
        # Run at 2 AM ET
        target = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        self.log(f"Next research cycle in {wait/3600:.1f} hours")
        await asyncio.sleep(wait)

    # ── Data gathering ─────────────────────────────────────────────────────────

    async def _fetch_historical_performance(self) -> dict:
        """Get 30-day performance for watchlist symbols."""
        from alpaca.data.timeframe import TimeFrame
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        loop  = asyncio.get_event_loop()
        symbols = UniverseConfig.WATCHLIST[:10]
        try:
            bars = await loop.run_in_executor(
                None,
                lambda: self.alpaca.get_bars(symbols, TimeFrame.Day, start),
            )
            perf = {}
            for sym, bar_list in bars.items():
                if len(bar_list) >= 2:
                    start_price = bar_list[0]["close"]
                    end_price   = bar_list[-1]["close"]
                    ret_30d     = (end_price - start_price) / start_price * 100
                    volumes     = [b["volume"] for b in bar_list]
                    import numpy as np
                    perf[sym] = {
                        "return_30d":   round(ret_30d, 2),
                        "start_price":  start_price,
                        "end_price":    end_price,
                        "avg_volume":   round(float(np.mean(volumes)), 0),
                        "price_range":  round((max(b["high"] for b in bar_list) -
                                               min(b["low"] for b in bar_list)) /
                                              start_price * 100, 2),
                    }
            return perf
        except Exception as e:
            self.log(f"Historical fetch error: {e}", "warning")
            return {}

    async def _fetch_recent_news_context(self) -> list[dict]:
        """Get recent news summaries stored by Bot 1."""
        news_raw = await self.bus.get_state("bot1:latest_summary")
        return [news_raw] if news_raw else []

    async def _fetch_options_context(self) -> dict:
        """Get recent options signals stored by Bot 2."""
        opts = {}
        for sym in ["SPY", "QQQ", "AAPL", "NVDA"]:
            state = await self.bus.get_state(f"bot2:last_scan")
            if state:
                opts = state
                break
        return opts

    # ── Research modules ───────────────────────────────────────────────────────

    async def _research_market_regime(self, perf_data: dict) -> dict:
        """Use Opus to classify current market regime and implications."""
        prompt = (
            "You are a macro strategist analysing market regime.\n\n"
            f"30-day performance data:\n{json.dumps(perf_data, indent=2)}\n\n"
            "Classify the current market regime:\n"
            "1. Trend: bull/bear/sideways/transition\n"
            "2. Volatility: low/normal/elevated/extreme\n"
            "3. Breadth: broad/narrow/deteriorating/improving\n"
            "4. Risk appetite: risk-on/risk-off/mixed\n"
            "5. Sector leadership: which sectors are outperforming and why\n"
            "6. Key risks to monitor in the next 2-4 weeks\n"
            "7. Recommended strategy style for this regime "
            "(momentum/mean-reversion/defensive/aggressive)\n\n"
            "Respond with JSON:\n"
            "{\"trend\": \"bull\", \"volatility\": \"normal\", \"breadth\": \"broad\", "
            "\"risk_appetite\": \"risk-on\", \"sector_leadership\": [\"tech\", \"energy\"], "
            "\"key_risks\": [\"string\"], \"recommended_style\": \"momentum\", "
            "\"regime_summary\": \"2-3 sentence summary\", \"confidence\": 0.8}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except Exception as e:
            self.log(f"Regime analysis error: {e}", "error")
            return {}

    async def _research_correlations(self, perf_data: dict) -> dict:
        """Analyse correlations between watchlist symbols."""
        prompt = (
            "Analyse correlations and relationships between these symbols based on recent performance.\n\n"
            f"Performance data:\n{json.dumps(perf_data, indent=2)}\n\n"
            "Identify:\n"
            "1. Highly correlated pairs (>0.8 correlation) – risk of concentration\n"
            "2. Uncorrelated/negatively correlated pairs – diversification opportunities\n"
            "3. Relative strength rankings (rank symbols 1=strongest)\n"
            "4. Which symbols are leading vs lagging the index\n"
            "5. Pair trade opportunities\n\n"
            "Respond with JSON:\n"
            "{\"high_correlation_pairs\": [[\"AAPL\", \"MSFT\"]], "
            "\"diversifiers\": [\"string\"], "
            "\"relative_strength\": {\"NVDA\": 1, \"AAPL\": 2}, "
            "\"leaders\": [\"NVDA\"], \"laggards\": [\"META\"], "
            "\"pair_trades\": [{\"long\": \"NVDA\", \"short\": \"AMD\", "
            "\"rationale\": \"string\"}]}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except Exception as e:
            self.log(f"Correlation analysis error: {e}", "error")
            return {}

    async def _research_watchlist_quality(self, perf_data: dict,
                                           regime: dict) -> dict:
        """
        Use Opus to recommend watchlist additions/removals and optimal coverage.
        """
        prompt = (
            "You are a portfolio researcher evaluating and improving a trading watchlist.\n\n"
            f"Current watchlist performance:\n{json.dumps(perf_data, indent=2)}\n\n"
            f"Market regime:\n{json.dumps(regime, indent=2)}\n\n"
            "Provide:\n"
            "1. Symbols to remove (low volume, poor liquidity, insufficient volatility)\n"
            "2. Suggested additions for the current regime (provide 5 ticker symbols with rationale)\n"
            "3. Optimal sector balance for the regime\n"
            "4. Expected alpha opportunities over next 2 weeks\n\n"
            "Respond with JSON:\n"
            "{\"remove\": [{\"symbol\": \"XYZ\", \"reason\": \"string\"}], "
            "\"add\": [{\"symbol\": \"ABC\", \"rationale\": \"string\", "
            "\"expected_catalyst\": \"string\"}], "
            "\"sector_weights\": {\"tech\": 0.30, \"finance\": 0.20}, "
            "\"alpha_opportunities\": [{\"symbol\": \"NVDA\", \"thesis\": \"string\", "
            "\"horizon\": \"2 weeks\"}]}"
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
            return json.loads(raw)
        except Exception as e:
            self.log(f"Watchlist research error: {e}", "error")
            return {}

    async def _generate_morning_brief(self, regime: dict, correlations: dict,
                                       watchlist_rec: dict, perf_data: dict) -> str:
        """Generate a morning briefing document for the operator."""
        prompt = (
            "Write a professional morning trading brief for an algorithmic trading system operator.\n\n"
            f"Market Regime Analysis:\n{json.dumps(regime, indent=2)}\n\n"
            f"Correlation Analysis:\n{json.dumps(correlations, indent=2)}\n\n"
            f"Watchlist Recommendations:\n{json.dumps(watchlist_rec, indent=2)}\n\n"
            f"Symbol Performance (30d):\n{json.dumps(perf_data, indent=2)}\n\n"
            "Format as a concise professional brief with sections:\n"
            "## Market Regime\n"
            "## Key Risks\n"
            "## Top Opportunities\n"
            "## Watchlist Changes Recommended\n"
            "## Strategy Recommendations\n"
            "Keep it under 400 words. Be specific with ticker symbols and numbers."
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            return response.content[0].text.strip()
        except Exception as e:
            self.log(f"Morning brief error: {e}", "error")
            return ""

    # ── Main overnight cycle ───────────────────────────────────────────────────

    async def _overnight_research_cycle(self):
        self.log("Starting overnight research cycle")

        # Gather data
        perf_data = await self._fetch_historical_performance()
        if not perf_data:
            self.log("No performance data available", "warning")
            return

        # Run research modules (sequentially to respect Opus rate limits)
        self.log("Analysing market regime…")
        regime = await self._research_market_regime(perf_data)

        self.log("Analysing correlations…")
        correlations = await self._research_correlations(perf_data)
        await asyncio.sleep(3)

        self.log("Evaluating watchlist…")
        watchlist_rec = await self._research_watchlist_quality(perf_data, regime)
        await asyncio.sleep(3)

        self.log("Generating morning brief…")
        morning_brief = await self._generate_morning_brief(
            regime, correlations, watchlist_rec, perf_data
        )

        # Build research report
        report = {
            "type":          "research_report",
            "date":          datetime.utcnow().strftime("%Y-%m-%d"),
            "regime":        regime,
            "correlations":  correlations,
            "watchlist_rec": watchlist_rec,
            "morning_brief": morning_brief,
            "perf_data":     perf_data,
            "timestamp":     datetime.utcnow().isoformat(),
        }

        # Save to Redis (persisted for 24h – available to all bots)
        await self.save_state("latest_report",   report,           ttl=86400)
        await self.save_state("market_regime",   regime,           ttl=86400)
        await self.save_state("morning_brief",   morning_brief,    ttl=86400)
        await self.save_state("watchlist_recs",  watchlist_rec,    ttl=86400)

        # Publish to bus
        await self.publish(RedisConfig.CHANNEL_RESEARCH, report)

        # Send morning brief as alert
        if morning_brief:
            await self.publish(RedisConfig.CHANNEL_ALERTS, {
                "type":    "morning_brief",
                "message": morning_brief[:1000],
            })

        self.log(
            f"Research cycle complete | regime={regime.get('trend', '?')} "
            f"{regime.get('volatility', '?')} | "
            f"style={regime.get('recommended_style', '?')}"
        )

        # Print morning brief to log
        if morning_brief:
            self.log(f"\n{'='*60}\nMORNING BRIEF\n{'='*60}\n{morning_brief}\n{'='*60}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = ResearchBot()
    asyncio.run(bot.start())
