"""
Circuit Breaker — pure Python, zero API calls.

Levels:
  0  normal       — continue, full size
  1  caution      — down 2% today   → halve all position sizes
  2  danger       — down 3% today   → close ALL positions immediately
  3  weekly_limit — down 5% week    → quarter all position sizes
  4  lockout      — down 10% peak   → stop bot, write LOCKFILE.lock
  5  emergency    — down 15% peak   → emergency exit + urgent alert
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os

from config import BASE_DIR

LOCKFILE = os.path.join(BASE_DIR, "LOCKFILE.lock")


@dataclass
class CBState:
    level: int
    action: str          # continue | halve_sizes | close_all | quarter_sizes | lockout | emergency_exit
    reason: str
    position_scale: float
    trading_allowed: bool


class CircuitBreaker:
    """
    Stateless check: call check() every command cycle.
    Maintains peak and week-start values internally.
    Writes LOCKFILE.lock on levels 4 and 5 (requires manual deletion to resume).
    """

    def __init__(self):
        self._peak_value: float = 0.0
        self._week_start_value: float = 0.0
        self._week_start_set: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_peak(self, portfolio_value: float):
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

    def set_week_start(self, value: float):
        """Call once at the start of each trading week."""
        if not self._week_start_set or value <= 0:
            self._week_start_value = value
            self._week_start_set = True

    def check(self, portfolio_value: float, session_start_value: float) -> CBState:
        """
        Evaluate all circuit breaker levels and return the highest triggered.
        CB has VETO power over all bots — callers must honour trading_allowed.
        """
        self.update_peak(portfolio_value)

        if session_start_value <= 0 or portfolio_value <= 0:
            return CBState(0, "continue", "no_data", 1.0, True)

        daily_pct = (portfolio_value - session_start_value) / session_start_value

        peak_pct = (
            (portfolio_value - self._peak_value) / self._peak_value
            if self._peak_value > 0 else 0.0
        )

        weekly_pct = (
            (portfolio_value - self._week_start_value) / self._week_start_value
            if self._week_start_value > 0 else 0.0
        )

        # Level 5: down 15% from peak → emergency exit
        if peak_pct <= -0.15:
            self._write_lockfile(f"Level 5: down {peak_pct*100:.1f}% from peak")
            return CBState(
                5, "emergency_exit",
                f"Down {peak_pct*100:.1f}% from peak (threshold: -15%)",
                0.0, False,
            )

        # Level 4: down 10% from peak → lockout
        if peak_pct <= -0.10:
            self._write_lockfile(f"Level 4: down {peak_pct*100:.1f}% from peak")
            return CBState(
                4, "lockout",
                f"Down {peak_pct*100:.1f}% from peak (threshold: -10%)",
                0.0, False,
            )

        # Level 3: down 5% this week → 25% position size
        if weekly_pct <= -0.05:
            return CBState(
                3, "quarter_sizes",
                f"Down {weekly_pct*100:.1f}% this week (threshold: -5%)",
                0.25, True,
            )

        # Level 2: down 3% today → close all
        if daily_pct <= -0.03:
            return CBState(
                2, "close_all",
                f"Down {daily_pct*100:.1f}% today (threshold: -3%)",
                0.0, False,
            )

        # Level 1: down 2% today → halve sizes
        if daily_pct <= -0.02:
            return CBState(
                1, "halve_sizes",
                f"Down {daily_pct*100:.1f}% today (threshold: -2%)",
                0.5, True,
            )

        return CBState(0, "continue", "normal", 1.0, True)

    @staticmethod
    def is_locked() -> bool:
        """True if LOCKFILE.lock exists (manual deletion required to resume)."""
        return os.path.exists(LOCKFILE)

    @staticmethod
    def _write_lockfile(reason: str):
        if not os.path.exists(LOCKFILE):
            with open(LOCKFILE, "w") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} | {reason}\n")
