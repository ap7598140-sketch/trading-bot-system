"""
Walk-Forward Backtesting Engine — zero API calls, pure pandas/numpy.

Rolling windows:
  • Train: 252 trading days (1 year)
  • OOS:   126 trading days (6 months)

Benchmarks: buy-and-hold, 200-day SMA trend follow, random entry.
Stress test: inject 3 random 10–15% crash events into OOS data.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd


TRAIN_WINDOW = 252
OOS_WINDOW   = 126


class WalkForwardEngine:
    """
    Run rolling walk-forward backtests with benchmark comparisons,
    regime performance breakdown, and crash stress testing.
    """

    def __init__(self, regime_detector=None):
        self._rd = regime_detector  # optional RegimeDetector instance

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, strategy_fn: Callable, symbol: str = "") -> dict:
        """
        df:          DataFrame with 'close' column (and optionally 'volume').
                     Index should be datetime-like and sorted ascending.
        strategy_fn: callable(train_df, oos_df) → list[dict]
                     Each trade dict must have: pnl_pct, optionally confidence, regime.
        Returns aggregated walk-forward results with benchmarks and stress test.
        """
        df = df.sort_index().copy()
        if "close" not in df.columns:
            return {"error": "missing_close_column"}

        windows = self._split_windows(df)
        if not windows:
            return {"error": "insufficient_data_for_walk_forward", "windows": []}

        window_results = []
        for i, (train_df, oos_df) in enumerate(windows):
            try:
                trades  = strategy_fn(train_df, oos_df)
                metrics = self._window_metrics(trades)
                regime_bd = {}
                if self._rd and self._rd._trained:
                    regime_bd = self._regime_breakdown(trades)
                window_results.append({
                    "window":           i + 1,
                    "oos_start":        str(oos_df.index[0].date()),
                    "oos_end":          str(oos_df.index[-1].date()),
                    "trades":           len(trades),
                    "metrics":          metrics,
                    "regime_breakdown": regime_bd,
                })
            except Exception as e:
                window_results.append({"window": i + 1, "error": str(e)})

        valid = [w for w in window_results if "metrics" in w and "error" not in w.get("metrics", {})]
        agg   = self._aggregate(valid)

        full_oos    = df.iloc[TRAIN_WINDOW:]
        benchmarks  = self._benchmarks(full_oos)
        stress      = self._stress_test(df, strategy_fn)

        return {
            "symbol":     symbol,
            "windows":    window_results,
            "aggregate":  agg,
            "benchmarks": benchmarks,
            "stress_test": stress,
        }

    # ── Window splitting ───────────────────────────────────────────────────────

    def _split_windows(self, df: pd.DataFrame) -> list[tuple]:
        windows = []
        start   = 0
        while start + TRAIN_WINDOW + OOS_WINDOW <= len(df):
            train = df.iloc[start:start + TRAIN_WINDOW]
            oos   = df.iloc[start + TRAIN_WINDOW:start + TRAIN_WINDOW + OOS_WINDOW]
            windows.append((train, oos))
            start += OOS_WINDOW  # step forward by one OOS window
        return windows

    # ── Per-window metrics ─────────────────────────────────────────────────────

    def _window_metrics(self, trades: list[dict]) -> dict:
        if not trades:
            return {"error": "no_trades"}

        pnls     = [float(t.get("pnl_pct", 0)) for t in trades]
        winners  = [p for p in pnls if p > 0]
        losers   = [p for p in pnls if p < 0]
        win_rate = len(winners) / len(pnls) * 100 if pnls else 0

        eq = [1.0]
        for p in pnls:
            eq.append(eq[-1] * (1 + p / 100))

        eq_arr = np.array(eq)
        daily  = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-9)
        sharpe = float(np.mean(daily) / np.std(daily) * np.sqrt(252)) if np.std(daily) > 0 else 0.0

        peak   = float(eq_arr[0])
        max_dd = 0.0
        for v in eq_arr:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Confidence bucket breakdown
        high_conf = [t for t in trades if float(t.get("confidence", 0.5)) >= 0.70]
        low_conf  = [t for t in trades if float(t.get("confidence", 0.5)) <  0.70]
        hc_wins   = [t for t in high_conf if float(t.get("pnl_pct", 0)) > 0]
        lc_wins   = [t for t in low_conf  if float(t.get("pnl_pct", 0)) > 0]

        return {
            "total_trades":     len(trades),
            "total_return_pct": round(float(np.sum(pnls)), 2),
            "win_rate_pct":     round(win_rate, 1),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_win_pct":      round(float(np.mean(winners)), 2) if winners else 0,
            "avg_loss_pct":     round(float(np.mean(losers)),  2) if losers  else 0,
            "high_conf_trades": len(high_conf),
            "high_conf_win_pct": round(len(hc_wins) / len(high_conf) * 100, 1) if high_conf else 0,
            "low_conf_trades":  len(low_conf),
            "low_conf_win_pct": round(len(lc_wins)  / len(low_conf)  * 100, 1) if low_conf  else 0,
        }

    def _aggregate(self, valid: list[dict]) -> dict:
        if not valid:
            return {}
        keys = ["total_return_pct", "win_rate_pct", "sharpe_ratio", "max_drawdown_pct"]
        agg  = {}
        for k in keys:
            vals = [w["metrics"].get(k, 0) for w in valid if isinstance(w.get("metrics"), dict)]
            if vals:
                agg[k + "_mean"] = round(float(np.mean(vals)), 3)
                agg[k + "_std"]  = round(float(np.std(vals)),  3)
        agg["windows_run"] = len(valid)
        return agg

    # ── Regime breakdown ───────────────────────────────────────────────────────

    @staticmethod
    def _regime_breakdown(trades: list[dict]) -> dict:
        bd: dict[str, list] = {}
        for t in trades:
            r = t.get("regime", "unknown")
            bd.setdefault(r, []).append(float(t.get("pnl_pct", 0)))
        result = {}
        for regime, pnls in bd.items():
            wins = [p for p in pnls if p > 0]
            result[regime] = {
                "trades":   len(pnls),
                "avg_pnl":  round(float(np.mean(pnls)), 2) if pnls else 0,
                "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            }
        return result

    # ── Benchmarks ─────────────────────────────────────────────────────────────

    @staticmethod
    def _benchmarks(df: pd.DataFrame) -> dict:
        if len(df) < 10:
            return {}
        closes = df["close"].values.astype(float)

        # 1. Buy-and-hold
        bah = (closes[-1] - closes[0]) / closes[0] * 100

        # 2. 200-day SMA trend following
        sma200    = pd.Series(closes).rolling(200, min_periods=1).mean().values
        sma_pnls  = []
        in_t      = False
        entry_px  = 0.0
        for i in range(len(closes)):
            if not in_t and closes[i] > sma200[i]:
                entry_px, in_t = closes[i], True
            elif in_t and closes[i] < sma200[i]:
                sma_pnls.append((closes[i] - entry_px) / entry_px * 100)
                in_t = False
        if in_t:
            sma_pnls.append((closes[-1] - entry_px) / entry_px * 100)
        sma_total = float(np.sum(sma_pnls)) if sma_pnls else 0.0

        # 3. Random entry with same 1%SL / 2%TP (100 monte-carlo runs)
        rng = random.Random(42)
        rand_runs = []
        n_trades  = max(1, len(closes) // 10)
        for _ in range(100):
            run_pnl = 0.0
            for _ in range(n_trades):
                idx = rng.randint(0, max(0, len(closes) - 12))
                ep  = closes[idx]
                pnl = (closes[min(idx + 10, len(closes) - 1)] - ep) / ep * 100
                for j in range(1, min(11, len(closes) - idx)):
                    cp = closes[idx + j]
                    if cp <= ep * 0.99:
                        pnl = -1.0
                        break
                    if cp >= ep * 1.02:
                        pnl = 2.0
                        break
                run_pnl += pnl
            rand_runs.append(run_pnl)

        return {
            "buy_and_hold_return_pct":  round(bah, 2),
            "sma200_trend_return_pct":  round(sma_total, 2),
            "random_entry_avg_pct":     round(float(np.mean(rand_runs)), 2),
            "random_entry_std_pct":     round(float(np.std(rand_runs)),  2),
        }

    # ── Stress test: inject crash events ──────────────────────────────────────

    def _stress_test(self, df: pd.DataFrame, strategy_fn: Callable) -> dict:
        """Inject 3 random 10–15% crash events in the OOS region and re-run."""
        if len(df) < TRAIN_WINDOW + OOS_WINDOW + 20:
            return {}
        try:
            df_c    = df.copy()
            closes  = df_c["close"].values.astype(float)
            rng     = random.Random(99)
            n_crash = 3
            indices = rng.sample(range(TRAIN_WINDOW, len(closes) - 20), n_crash)
            for idx in indices:
                drop = rng.uniform(0.10, 0.15)
                span = min(5, len(closes) - idx)
                for j in range(span):
                    closes[idx + j] *= (1 - drop / span)
            df_c["close"] = closes

            windows = self._split_windows(df_c)
            all_trades: list[dict] = []
            for train_df, oos_df in windows:
                try:
                    all_trades.extend(strategy_fn(train_df, oos_df))
                except Exception:
                    pass

            metrics = self._window_metrics(all_trades) if all_trades else {"error": "no_trades"}
            return {
                "crashes_injected": n_crash,
                "crash_drop_range": "10–15%",
                "metrics":          metrics,
            }
        except Exception as e:
            return {"error": str(e)}
