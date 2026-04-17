"""
Bot 10 – Backtesting Bot
Model  : Opus 4.6  (overnight deep analysis)
Role   : Runs comprehensive strategy backtests on historical data.
         • Triggered by Bot 11 (Strategy Builder) or manual command
         • Fetches historical OHLCV data from Alpaca
         • Simulates strategy execution with realistic slippage/commission
         • Computes full performance metrics (Sharpe, Sortino, max drawdown, etc.)
         • Uses Opus to deeply interpret results and suggest improvements
         • Publishes results to CHANNEL_BACKTEST
         Runs overnight to avoid consuming live trading resources.
"""

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

from alpaca.data.timeframe import TimeFrame

from config import Models, RedisConfig, AnthropicConfig
from shared.base_bot import BaseBot
from shared.alpaca_client import AlpacaClient
from shared.walk_forward import WalkForwardEngine
from shared.regime_detector import RegimeDetector


# Simulation parameters
COMMISSION_PER_SHARE = 0.005   # $0.005 per share
SLIPPAGE_BPS         = 5       # 5 basis points slippage


class BacktestingBot(BaseBot):
    """
    Bot 10 – Backtesting Bot
    Deep overnight analysis with Opus 4.6.
    """

    BOT_ID = 10
    NAME   = "Backtesting Bot"

    def __init__(self):
        super().__init__(self.BOT_ID, self.NAME, Models.OPUS)
        self.client = anthropic.Anthropic(api_key=AnthropicConfig.API_KEY)
        self.alpaca = AlpacaClient()
        self._results_history: list[dict] = []
        self._regime   = RegimeDetector()
        self._wf_engine = WalkForwardEngine(regime_detector=self._regime)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self):
        await self.bus.subscribe(RedisConfig.CHANNEL_BACKTEST, self._on_backtest_request)
        asyncio.create_task(self.bus.listen())
        self.log("Backtesting Bot starting (Opus 4.6 – overnight mode)")

    async def run(self):
        # Scheduled: run nightly at midnight
        while self.running:
            await self._wait_for_midnight()
            try:
                await self._nightly_backtest_run()
            except Exception as e:
                self.log(f"Nightly backtest error: {e}", "error")

    async def cleanup(self):
        self.log("Backtesting Bot stopped")

    # ── Scheduling ─────────────────────────────────────────────────────────────

    async def _wait_for_midnight(self):
        import pytz
        et  = pytz.timezone("America/New_York")
        now = datetime.now(et)
        target = now.replace(hour=0, minute=5, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        self.log(f"Next backtest run in {wait/3600:.1f} hours")
        await asyncio.sleep(wait)

    # ── On-demand backtest request ─────────────────────────────────────────────

    async def _on_backtest_request(self, data: dict):
        if data.get("type") != "backtest_request":
            return
        strategy = data.get("strategy", {})
        symbols  = data.get("symbols", ["SPY"])
        days     = data.get("days", 90)
        self.log(f"On-demand backtest: {strategy.get('name', 'unnamed')} | {symbols} | {days}d")
        await self._run_backtest(strategy, symbols, days)

    # ── Nightly run ────────────────────────────────────────────────────────────

    async def _nightly_backtest_run(self):
        self.log("Starting nightly backtest run (walk-forward + standard)")

        from config import UniverseConfig
        symbols_wf = UniverseConfig.WATCHLIST[:5]   # walk-forward on top 5

        # Train regime detector on SPY before walk-forward
        await self._train_regime_for_wf(symbols_wf)

        # Walk-forward run
        await self._run_walk_forward(symbols_wf)

        # Standard per-strategy backtests (existing behaviour)
        strategies_raw = await self.bus.get_state("bot11:strategies")
        if not strategies_raw:
            strategies_raw = [self._default_strategy()]

        for strategy in strategies_raw:
            await self._run_backtest(strategy, UniverseConfig.WATCHLIST[:8], days=90)
            await asyncio.sleep(5)

    async def _train_regime_for_wf(self, symbols: list[str]):
        """Train HMM on SPY before walk-forward to enable regime breakdown."""
        try:
            start = (datetime.now(timezone.utc) - timedelta(days=510)).strftime("%Y-%m-%d")
            loop  = asyncio.get_event_loop()
            bars  = await loop.run_in_executor(
                None, lambda: self.alpaca.get_bars(["SPY"], TimeFrame.Day, start)
            )
            spy_bars = bars.get("SPY", [])
            if len(spy_bars) < 60:
                return
            closes  = np.array([b["close"]  for b in spy_bars], dtype=float)
            volumes = np.array([b["volume"] for b in spy_bars], dtype=float)
            ok = await loop.run_in_executor(
                None, lambda: self._regime.train(closes, volumes)
            )
            if ok:
                self.log("Walk-forward: regime detector trained")
        except Exception as e:
            self.log(f"Walk-forward regime training error: {e}", "warning")

    async def _run_walk_forward(self, symbols: list[str]):
        """Run walk-forward engine on each symbol and publish results."""
        strategy = self._default_strategy()
        self.log(f"Walk-forward backtest | symbols={symbols}")

        for sym in symbols:
            try:
                all_dfs = await self._fetch_historical_data([sym], days=550)
                df = all_dfs.get(sym)
                if df is None or len(df) < 420:
                    self.log(f"  {sym}: insufficient data for walk-forward ({len(df) if df is not None else 0} bars)")
                    continue
                df = self._add_indicators(df)

                def strategy_fn(train_df: pd.DataFrame, oos_df: pd.DataFrame) -> list[dict]:
                    result = self._simulate_strategy(oos_df, strategy)
                    return result.get("trades", [])

                wf_result = self._wf_engine.run(df, strategy_fn, symbol=sym)
                agg   = wf_result.get("aggregate", {})
                bench = wf_result.get("benchmarks", {})
                self.log(
                    f"  {sym} WF: return={agg.get('total_return_pct_mean', 0):.1f}% "
                    f"sharpe={agg.get('sharpe_ratio_mean', 0):.2f} "
                    f"vs BaH={bench.get('buy_and_hold_return_pct', 0):.1f}%"
                )
                wf_result["type"]      = "walk_forward_result"
                wf_result["timestamp"] = datetime.utcnow().isoformat()
                await self.save_state(f"wf_{sym}", wf_result, ttl=86400)
                await self.publish(RedisConfig.CHANNEL_BACKTEST, wf_result)
            except Exception as e:
                self.log(f"  {sym} walk-forward error: {e}", "warning")

    def _default_strategy(self) -> dict:
        return {
            "name":         "momentum_rsi_macd",
            "entry_rules":  ["rsi_14 > 50", "rsi_14 < 70", "macd_histogram > 0", "price > sma_50"],
            "exit_rules":   ["rsi_14 > 75", "macd_histogram < 0", "price < sma_20"],
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "position_size_pct": 0.10,
        }

    # ── Data fetching ──────────────────────────────────────────────────────────

    async def _fetch_historical_data(self, symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        loop  = asyncio.get_event_loop()

        bars = await loop.run_in_executor(
            None,
            lambda: self.alpaca.get_bars(symbols, TimeFrame.Day, start),
        )

        dfs = {}
        for sym, bar_list in bars.items():
            if not bar_list:
                self.log(f"  {sym}: no bar data, skipping")
                continue
            df = pd.DataFrame(bar_list)
            if "close" not in df.columns or len(df) < 30:
                self.log(f"  {sym}: insufficient data ({len(df)} bars), skipping")
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            dfs[sym] = df
        return dfs

    # ── Indicator calculation ──────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]           = ema12 - ema26
        df["macd_signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # SMAs
        df["sma_20"]  = df["close"].rolling(20).mean()
        df["sma_50"]  = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()

        # Bollinger
        df["bb_mid"]   = df["close"].rolling(20).mean()
        bb_std         = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std

        # ATR
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        return df.dropna()

    # ── Strategy simulation ────────────────────────────────────────────────────

    def _simulate_strategy(self, df: pd.DataFrame, strategy: dict,
                            initial_capital: float = 10000.0) -> dict:
        """
        Simple vectorised backtest simulation.
        Returns trades list and equity curve.
        """
        stop_pct   = strategy.get("stop_loss_pct", 0.02)
        target_pct = strategy.get("take_profit_pct", 0.04)
        size_pct   = strategy.get("position_size_pct", 0.10)

        if df is None or len(df) == 0 or "close" not in df.columns:
            return {"trades": [], "equity": [initial_capital], "final_capital": initial_capital}

        try:
            return self._simulate_strategy_inner(df, strategy, initial_capital,
                                                  stop_pct, target_pct, size_pct)
        except Exception as e:
            return {"trades": [], "equity": [initial_capital], "final_capital": initial_capital,
                    "error": str(e)}

    def _simulate_strategy_inner(self, df, strategy, initial_capital,
                                  stop_pct, target_pct, size_pct):
        capital    = initial_capital
        position   = 0     # shares held
        entry_price = 0.0
        trades     = []
        equity     = [capital]
        in_trade   = False

        for i, (ts, row) in enumerate(df.iterrows()):
            price = row["close"]

            if not in_trade:
                # Entry signal evaluation
                entry_signal = self._eval_entry(row, strategy)
                if entry_signal:
                    size_usd  = capital * size_pct
                    shares    = max(1, int(size_usd / price))
                    slip      = price * (SLIPPAGE_BPS / 10000)
                    exec_price = price + slip
                    cost       = shares * exec_price + shares * COMMISSION_PER_SHARE
                    if cost <= capital:
                        capital     -= cost
                        position     = shares
                        entry_price  = exec_price
                        in_trade     = True
                        trades.append({
                            "entry_date": str(ts.date()),
                            "entry":      round(exec_price, 2),
                            "shares":     shares,
                        })
            else:
                # Exit signal evaluation
                pnl_pct = (price - entry_price) / entry_price

                exit_signal = (
                    pnl_pct <= -stop_pct or         # stop loss
                    pnl_pct >= target_pct or         # take profit
                    self._eval_exit(row, strategy)   # strategy exit rule
                )

                if exit_signal or i == len(df) - 1:
                    slip       = price * (SLIPPAGE_BPS / 10000)
                    exec_price = price - slip
                    proceeds   = position * exec_price - position * COMMISSION_PER_SHARE
                    capital   += proceeds
                    pnl        = proceeds - (trades[-1]["shares"] * trades[-1]["entry"])
                    pnl_pct_r  = pnl / (trades[-1]["shares"] * trades[-1]["entry"])

                    trades[-1].update({
                        "exit_date": str(ts.date()),
                        "exit":      round(exec_price, 2),
                        "pnl":       round(pnl, 2),
                        "pnl_pct":   round(pnl_pct_r * 100, 2),
                        "reason":    "stop_loss" if pnl_pct <= -stop_pct
                                     else "take_profit" if pnl_pct >= target_pct
                                     else "signal_exit",
                    })
                    position = 0
                    in_trade = False

            equity.append(capital + (position * price if in_trade else 0))

        last_price = df["close"].iloc[-1] if len(df) > 0 else 0.0
        return {"trades": trades, "equity": equity, "final_capital": capital + position * last_price}

    def _eval_entry(self, row, strategy: dict) -> bool:
        rules = strategy.get("entry_rules", [])
        for rule in rules:
            try:
                if not self._eval_rule(row, rule):
                    return False
            except Exception:
                pass
        return len(rules) > 0

    def _eval_exit(self, row, strategy: dict) -> bool:
        rules = strategy.get("exit_rules", [])
        for rule in rules:
            try:
                if self._eval_rule(row, rule):
                    return True
            except Exception:
                pass
        return False

    def _eval_rule(self, row, rule: str) -> bool:
        """Evaluate a simple rule string like 'rsi_14 > 50'."""
        parts = rule.strip().split()
        if len(parts) == 3:
            col, op, val = parts
            lhs = row.get(col)
            rhs = float(val)
            if lhs is None:
                return False
            if op == ">":  return lhs > rhs
            if op == "<":  return lhs < rhs
            if op == ">=": return lhs >= rhs
            if op == "<=": return lhs <= rhs
            if op == "==": return abs(lhs - rhs) < 0.001
        return False

    # ── Performance metrics ────────────────────────────────────────────────────

    def _calc_metrics(self, trades: list[dict], equity: list[float],
                       initial: float) -> dict:
        if not trades:
            return {"error": "no_trades"}

        total_return  = (equity[-1] - initial) / initial * 100
        pnl_list      = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
        winners       = [p for p in pnl_list if p > 0]
        losers        = [p for p in pnl_list if p < 0]
        win_rate      = len(winners) / len(pnl_list) * 100 if pnl_list else 0

        # Sharpe (daily returns from equity curve)
        eq_arr = np.array(equity)
        daily_returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252)
                  if np.std(daily_returns) > 0 else 0)

        # Sortino
        neg_returns  = daily_returns[daily_returns < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 0
        sortino      = (np.mean(daily_returns) / downside_std * math.sqrt(252)
                        if downside_std > 0 else 0)

        # Max drawdown
        peak     = eq_arr[0] if len(eq_arr) > 0 else 0.0
        max_dd   = 0.0
        for v in eq_arr:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        avg_win  = sum(winners) / len(winners) if winners else 0
        avg_loss = sum(losers)  / len(losers)  if losers  else 0
        profit_factor = abs(sum(winners) / sum(losers)) if losers else float("inf")

        return {
            "total_trades":    len(trades),
            "win_rate_pct":    round(win_rate, 1),
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio":    round(sharpe, 3),
            "sortino_ratio":   round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_win_pct":     round(avg_win, 2),
            "avg_loss_pct":    round(avg_loss, 2),
            "profit_factor":   round(profit_factor, 3),
            "final_capital":   round(equity[-1], 2),
        }

    # ── AI interpretation ──────────────────────────────────────────────────────

    async def _ai_interpret_results(self, strategy: dict, metrics: dict,
                                     trades: list[dict]) -> str:
        """Opus gives deep interpretation and concrete improvement suggestions."""
        recent_trades = trades[-10:] if len(trades) > 10 else trades

        prompt = (
            "You are a quantitative analyst reviewing a trading strategy backtest. "
            "Provide a thorough, expert analysis.\n\n"
            f"STRATEGY:\n{json.dumps(strategy, indent=2)}\n\n"
            f"PERFORMANCE METRICS:\n{json.dumps(metrics, indent=2)}\n\n"
            f"RECENT TRADES (sample):\n{json.dumps(recent_trades, indent=2)}\n\n"
            "Provide:\n"
            "1. Overall performance assessment (strengths and weaknesses)\n"
            "2. Statistical edge analysis (is the win rate / profit factor sustainable?)\n"
            "3. Risk-adjusted return quality (Sharpe/Sortino interpretation)\n"
            "4. Drawdown analysis (is max DD acceptable?)\n"
            "5. 3-5 specific, actionable improvements with expected impact\n"
            "6. Whether to deploy this strategy live (yes/no/conditional)\n\n"
            "Be specific and quantitative. Reference the actual numbers."
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
            return response.content[0].text.strip()
        except Exception as e:
            self.log(f"Opus interpretation error: {e}", "error")
            return "Interpretation failed."

    # ── Main backtest runner ───────────────────────────────────────────────────

    async def _run_backtest(self, strategy: dict, symbols: list[str], days: int):
        name = strategy.get("name", "unnamed")
        self.log(f"Running backtest: {name} | {symbols} | {days}d")

        # Fetch data
        try:
            all_dfs = await self._fetch_historical_data(symbols, days)
        except Exception as e:
            self.log(f"Data fetch failed: {e}", "error")
            return

        # Run per-symbol backtests
        all_metrics = {}
        all_trades  = []
        for sym, df in all_dfs.items():
            try:
                df_ind  = self._add_indicators(df)
                result  = self._simulate_strategy(df_ind, strategy)
                metrics = self._calc_metrics(result["trades"], result["equity"], 10000.0)
                all_metrics[sym] = metrics
                all_trades.extend(result["trades"])
                self.log(f"  {sym}: {metrics.get('total_return_pct', 0):.1f}% | "
                         f"SR={metrics.get('sharpe_ratio', 0):.2f} | "
                         f"DD={metrics.get('max_drawdown_pct', 0):.1f}%")
            except Exception as e:
                self.log(f"  {sym} backtest error: {e}", "warning")

        if not all_metrics:
            return

        # Aggregate across symbols
        avg_metrics = {
            k: round(sum(m.get(k, 0) for m in all_metrics.values()) / len(all_metrics), 3)
            for k in ["total_return_pct", "sharpe_ratio", "sortino_ratio",
                      "max_drawdown_pct", "win_rate_pct", "profit_factor"]
        }

        # Opus deep analysis
        self.log(f"Running Opus analysis for {name}…")
        interpretation = await self._ai_interpret_results(strategy, avg_metrics, all_trades)

        result_record = {
            "type":            "backtest_result",
            "strategy_name":   name,
            "symbols":         symbols,
            "days":            days,
            "per_symbol":      all_metrics,
            "avg_metrics":     avg_metrics,
            "interpretation":  interpretation,
            "timestamp":       datetime.utcnow().isoformat(),
        }

        self._results_history.append(result_record)
        await self.save_state("latest_result", result_record, ttl=86400)
        await self.publish(RedisConfig.CHANNEL_BACKTEST, result_record)

        self.log(
            f"Backtest complete: {name} | "
            f"avg_return={avg_metrics.get('total_return_pct')}% | "
            f"sharpe={avg_metrics.get('sharpe_ratio')} | "
            f"max_dd={avg_metrics.get('max_drawdown_pct')}%"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    bot = BacktestingBot()
    asyncio.run(bot.start())
