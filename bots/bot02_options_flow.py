"""
Bot 2 – Options Flow Bot
Model  : Sonnet 4.6  (live trading decisions)
Role   : Monitors unusual options activity via Alpaca's options data feed.
         Detects large call/put sweeps, unusual open interest changes, and
         skew shifts that signal smart-money positioning.
         Publishes options flow signals to CHANNEL_OPTIONS_FLOW.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import aiohttp

from config import Models, RedisConfig, UniverseConfig, AnthropicConfig, AlpacaConfig
from shared.base_bot import BaseBot


POLL_INTERVAL    = 3600  # seconds between options scans (60 min)
SWEEP_THRESHOLD  = 500   # contracts for a "large" sweep
OI_CHANGE_THRESH = 0.20  # 20% OI change = unusual


class OptionsFlowBot(BaseBot):
    """
    Bot 2 – Options Flow Bot
    Uses Sonnet 4.6 to interpret options flow anomalies in market context.
    """

    BOT_ID = 2
    NAME   = "Options Flow Bot"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.SONNET)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self._prev_oi: dict[str, dict] = {}    # symbol -> {strike: oi}
        self._market_context: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        # Subscribe to market data for price context
        await self.bus.subscribe(RedisConfig.CHANNEL_MARKET_DATA, self._on_market_data)
        asyncio.create_task(self.bus.listen())
        self.log("Options Flow Bot starting")

    async def run(self):
        while self.running:
            try:
                await self._options_cycle()
            except Exception as e:
                self.log(f"Options cycle error: {e}", "error")
            await asyncio.sleep(POLL_INTERVAL)

    async def cleanup(self):
        self.log("Options Flow Bot stopped")

    # ── Market data context ────────────────────────────────────────────────────

    async def _on_market_data(self, data: dict):
        sym = data.get("symbol")
        if sym:
            self._market_context[sym] = {
                "price":   data.get("price"),
                "rsi":     data.get("indicators", {}).get("rsi_14"),
                "vwap":    data.get("indicators", {}).get("vwap"),
                "vol_ratio": data.get("indicators", {}).get("volume_ratio"),
            }

    # ── Options data fetch ─────────────────────────────────────────────────────

    async def _fetch_options_chain(self, symbol: str) -> list[dict]:
        """Fetch options chain snapshot from Alpaca."""
        url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
        params = {"limit": 100, "type": "all"}
        headers = {
            "APCA-API-KEY-ID":     AlpacaConfig.API_KEY,
            "APCA-API-SECRET-KEY": AlpacaConfig.SECRET_KEY,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_snapshots(data.get("snapshots", {}))
                elif resp.status == 404:
                    return []   # Options not available for this symbol
                else:
                    self.log(f"Options fetch {symbol} status {resp.status}", "warning")
                    return []

    def _parse_snapshots(self, snapshots: dict) -> list[dict]:
        contracts = []
        for contract_sym, snap in snapshots.items():
            greeks = snap.get("greeks", {})
            quote  = snap.get("latestQuote", {})
            trade  = snap.get("latestTrade", {})
            detail = snap.get("details", {})
            contracts.append({
                "contract":     contract_sym,
                "type":         detail.get("type", ""),          # call / put
                "strike":       float(detail.get("strike_price", 0)),
                "expiry":       detail.get("expiration_date", ""),
                "open_interest": int(snap.get("openInterest", 0)),
                "volume":       int(snap.get("volume", 0)),
                "iv":           float(greeks.get("impliedVolatility", 0)),
                "delta":        float(greeks.get("delta", 0)),
                "gamma":        float(greeks.get("gamma", 0)),
                "bid":          float(quote.get("bp", 0)),
                "ask":          float(quote.get("ap", 0)),
                "last":         float(trade.get("p", 0)),
            })
        return contracts

    # ── Flow analysis ──────────────────────────────────────────────────────────

    def _detect_unusual_activity(self, symbol: str,
                                  chain: list[dict]) -> list[dict]:
        alerts = []
        prev   = self._prev_oi.get(symbol, {})

        for c in chain:
            contract = c["contract"]
            oi       = c["open_interest"]
            vol      = c["volume"]
            prev_oi  = prev.get(contract, oi)

            # 1. Volume spike: volume > 2x open interest
            if oi > 0 and vol > oi * 2:
                alerts.append({
                    "type":     "volume_spike",
                    "contract": contract,
                    "option_type": c["type"],
                    "strike":   c["strike"],
                    "expiry":   c["expiry"],
                    "volume":   vol,
                    "oi":       oi,
                    "ratio":    round(vol / oi, 1),
                    "iv":       c["iv"],
                    "delta":    c["delta"],
                })

            # 2. Large sweep
            if vol >= SWEEP_THRESHOLD:
                alerts.append({
                    "type":     "large_sweep",
                    "contract": contract,
                    "option_type": c["type"],
                    "strike":   c["strike"],
                    "expiry":   c["expiry"],
                    "volume":   vol,
                    "premium_paid": round(vol * c["ask"] * 100, 0),
                    "delta":    c["delta"],
                    "iv":       c["iv"],
                })

            # 3. OI change
            if prev_oi > 0:
                oi_change = (oi - prev_oi) / prev_oi
                if abs(oi_change) >= OI_CHANGE_THRESH:
                    alerts.append({
                        "type":       "oi_change",
                        "contract":   contract,
                        "option_type": c["type"],
                        "strike":     c["strike"],
                        "expiry":     c["expiry"],
                        "oi_prev":    prev_oi,
                        "oi_now":     oi,
                        "pct_change": round(oi_change * 100, 1),
                    })

        # Update prev OI
        self._prev_oi[symbol] = {c["contract"]: c["open_interest"] for c in chain}

        # 4. Put/Call ratio
        calls = sum(c["volume"] for c in chain if c["type"] == "call")
        puts  = sum(c["volume"] for c in chain if c["type"] == "put")
        if calls + puts > 0:
            pc_ratio = puts / calls if calls > 0 else 999
            if pc_ratio > 2.0:
                alerts.append({"type": "put_call_extreme_bearish", "ratio": round(pc_ratio, 2)})
            elif pc_ratio < 0.3:
                alerts.append({"type": "put_call_extreme_bullish", "ratio": round(pc_ratio, 2)})

        return alerts

    # ── AI interpretation ──────────────────────────────────────────────────────

    async def _ai_interpret_flow(self, symbol: str,
                                  alerts: list[dict]) -> Optional[dict]:
        """Use Sonnet to interpret the options flow in full market context."""
        if not alerts:
            return None

        ctx = self._market_context.get(symbol, {})
        prompt = (
            f"You are an options flow analyst interpreting smart-money signals for {symbol}.\n\n"
            f"Current price: ${ctx.get('price', 'N/A')}\n"
            f"RSI-14: {ctx.get('rsi', 'N/A')}\n"
            f"VWAP: {ctx.get('vwap', 'N/A')}\n"
            f"Volume ratio: {ctx.get('vol_ratio', 'N/A')}x\n\n"
            f"Unusual options activity detected:\n{json.dumps(alerts, indent=2)}\n\n"
            "Analyse the flow and determine:\n"
            "1. Overall directional bias (bullish/bearish/neutral)\n"
            "2. Confidence level (0-1)\n"
            "3. Time horizon (intraday/swing/monthly)\n"
            "4. Recommended action (buy_calls/buy_puts/sell_calls/sell_puts/none)\n"
            "5. Key insight in one sentence\n\n"
            "Respond ONLY with JSON:\n"
            "{\"bias\": \"bullish\", \"confidence\": 0.8, \"horizon\": \"swing\", "
            "\"action\": \"buy_calls\", \"insight\": \"Large institutional call sweep suggests...\", "
            "\"target_strike\": 150, \"target_expiry\": \"2025-01-17\"}"
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
            return json.loads(raw)
        except Exception as e:
            self.log(f"AI interpret error for {symbol}: {e}", "warning")
            return None

    # ── Main cycle ─────────────────────────────────────────────────────────────

    async def _options_cycle(self):
        # Focus on highest-conviction symbols (large-cap, most liquid options)
        priority_symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META"]
        active_symbols   = [s for s in priority_symbols if s in UniverseConfig.WATCHLIST]

        for symbol in active_symbols:
            try:
                chain  = await self._fetch_options_chain(symbol)
                if not chain:
                    continue

                alerts = self._detect_unusual_activity(symbol, chain)
                if not alerts:
                    continue

                # High-signal alerts only (>2 alerts = significant)
                if len(alerts) < 2 and not any(a["type"] == "large_sweep" for a in alerts):
                    continue

                interpretation = await self._ai_interpret_flow(symbol, alerts)
                if not interpretation:
                    continue

                # Only publish if confidence >= 0.6
                if interpretation.get("confidence", 0) < 0.6:
                    continue

                signal = {
                    "type":           "options_flow_signal",
                    "symbol":         symbol,
                    "alerts":         alerts,
                    "bias":           interpretation.get("bias"),
                    "confidence":     interpretation.get("confidence"),
                    "horizon":        interpretation.get("horizon"),
                    "action":         interpretation.get("action"),
                    "insight":        interpretation.get("insight"),
                    "target_strike":  interpretation.get("target_strike"),
                    "target_expiry":  interpretation.get("target_expiry"),
                    "alert_count":    len(alerts),
                }

                await self.publish(RedisConfig.CHANNEL_OPTIONS_FLOW, signal)
                self.log(
                    f"Options signal: {symbol} | {interpretation.get('bias')} | "
                    f"conf={interpretation.get('confidence')} | {interpretation.get('insight')}"
                )

                # Small delay between symbols to respect rate limits
                await asyncio.sleep(2)

            except Exception as e:
                self.log(f"Error processing {symbol}: {e}", "warning")

        # Save summary
        await self.save_state("last_scan", {
            "symbols_scanned": len(active_symbols),
            "timestamp": datetime.utcnow().isoformat(),
        }, ttl=120)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = OptionsFlowBot()
    asyncio.run(bot.start())
