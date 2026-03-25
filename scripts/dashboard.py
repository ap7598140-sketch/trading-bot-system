"""
Simple terminal dashboard – polls Redis and prints system state.
Run: python scripts/dashboard.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis.asyncio as aioredis
from config import RedisConfig


async def main():
    r = await aioredis.from_url(
        f"redis://{RedisConfig.HOST}:{RedisConfig.PORT}/{RedisConfig.DB}",
        decode_responses=True,
    )

    while True:
        os.system("clear")
        print(f"{'='*60}")
        print(f"  AI Trading Bot Dashboard  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # Master Commander dashboard
        raw = await r.get("bot8:dashboard")
        if raw:
            d = json.loads(raw)
            session = d.get("market_session", "?")
            halted  = d.get("system_halted", False)
            pv      = d.get("portfolio_value", 0)
            pnl     = d.get("daily_pnl", 0)
            pnl_pct = d.get("daily_pnl_pct", 0)

            status_str = "🔴 HALTED" if halted else "🟢 LIVE"
            print(f"\n  {status_str}  |  Session: {session.upper()}")
            print(f"  Portfolio: ${pv:,.2f}  |  Daily P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

            bot_health = d.get("bot_health", {})
            alive  = sum(1 for s in bot_health.values() if s == "alive")
            dead   = sum(1 for s in bot_health.values() if s == "dead")
            unseen = sum(1 for s in bot_health.values() if s in ("never_seen", "unknown"))
            print(f"  Bots: {alive} alive | {dead} dead | {unseen} unseen")

            if d.get("commander_notes"):
                print(f"\n  Commander: {d['commander_notes'][:80]}")

        # Risk stats
        raw = await r.get("bot6:stats")
        if raw:
            d = json.loads(raw)
            print(f"\n  Risk: {d.get('approved_today', 0)} approved | "
                  f"{d.get('rejected_today', 0)} rejected | "
                  f"{d.get('open_positions', 0)} open positions")

        # Momentum leaderboard
        raw = await r.get("bot3:leaderboard")
        if raw:
            d = json.loads(raw)
            bull = d.get("top_bullish", [])
            bear = d.get("top_bearish", [])
            print(f"\n  Momentum Top Bullish:  {' | '.join(f\"{s['sym']} {s['grade']}\" for s in bull)}")
            print(f"  Momentum Top Bearish:  {' | '.join(f\"{s['sym']} {s['grade']}\" for s in bear)}")

        # News summary
        raw = await r.get("bot1:latest_summary")
        if raw:
            d = json.loads(raw)
            mood = d.get("market_mood", "?")
            bull = d.get("bullish", 0)
            bear = d.get("bearish", 0)
            print(f"\n  News: {mood.upper()} | bull={bull} bear={bear}")

        # Execution summary
        raw = await r.get("bot7:summary")
        if raw:
            d = json.loads(raw)
            print(f"\n  Execution: {d.get('executions_today', 0)} trades | "
                  f"{d.get('pending_orders', 0)} pending | "
                  f"halted={d.get('trading_halted', False)}")

        print(f"\n{'='*60}")
        print("  Press Ctrl+C to exit")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDashboard exited")
