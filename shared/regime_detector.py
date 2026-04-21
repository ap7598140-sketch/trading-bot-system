"""
HMM Regime Detector — zero API calls, pure hmmlearn + numpy.

5 regimes: crash, bear, neutral, bull, euphoria
Features:  log return, 5-day rolling volatility, volume ratio vs 20-day avg

Falls back gracefully to "neutral" if hmmlearn is not installed.
"""

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

STABILITY_BARS = 3    # regime must persist this many consecutive bars before acting
FLICKER_WINDOW = 20   # look-back window for flicker detection
FLICKER_THRESH = 4    # changes in FLICKER_WINDOW → uncertain (halve position sizes)


class RegimeDetector:
    """
    Train on 2 years of daily data; predict with forward algorithm (no lookahead bias).
    Auto-selects best N regimes (3–7) by BIC score.
    """

    def __init__(self):
        self._model = None
        self._n_regimes: int = 5
        self._regime_map: dict[int, str] = {}  # HMM state index → regime name
        self._history: list[str] = []          # raw per-bar predictions
        self._last_stable: str = "neutral"     # stability-filtered regime
        self._trained: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Fit the HMM on historical data. Returns True on success."""
        if not HMM_AVAILABLE:
            return False
        if len(closes) < 60:
            return False
        try:
            X = self._build_features(closes, volumes)
            if len(X) < 30:
                return False
            n = self._auto_select_n_regimes(X)
            self._n_regimes = n
            model = GaussianHMM(
                n_components=n, covariance_type="diag",
                n_iter=100, random_state=42,
            )
            model.fit(X)
            self._model = model
            self._regime_map = self._label_regimes(model, n)
            self._trained = True
            return True
        except Exception:
            return False

    def predict(self, closes: np.ndarray, volumes: np.ndarray) -> str:
        """
        Return current regime name (stability-filtered).
        Uses forward algorithm — no future data ever accessed.
        """
        if not self._trained or self._model is None:
            return "neutral"
        try:
            X = self._build_features(closes, volumes)
            if len(X) == 0:
                return self._last_stable
            state = self._model.predict(X)[-1]
            regime = self._regime_map.get(int(state), "neutral")
            self._history.append(regime)
            if len(self._history) > FLICKER_WINDOW:
                self._history = self._history[-FLICKER_WINDOW:]
            # Stability filter: only update _last_stable when last N bars agree
            if len(self._history) >= STABILITY_BARS:
                recent = self._history[-STABILITY_BARS:]
                if all(r == recent[0] for r in recent):
                    self._last_stable = recent[0]
            return self._last_stable
        except Exception:
            return self._last_stable

    def is_uncertain(self) -> bool:
        """True if 4+ regime flickers detected in last 20 bars."""
        if len(self._history) < 2:
            return False
        changes = sum(
            1 for i in range(1, len(self._history))
            if self._history[i] != self._history[i - 1]
        )
        return changes >= FLICKER_THRESH

    def allocation_scale(self) -> float:
        """
        Position size multiplier based on current stable regime.
        Halved when regime is flickering (uncertain).

        euphoria → 0.70  (tighten stops)
        bull     → 0.95  (1.25× leverage max)
        neutral  → 0.60  (no leverage)
        bear     → 0.50  (reduced size, still trade)
        crash    → 0.00  (close everything)
        """
        scales = {
            "euphoria": 0.70,
            "bull":     0.95,
            "neutral":  0.60,
            "bear":     0.50,
            "crash":    0.00,
        }
        scale = scales.get(self._last_stable, 0.60)
        if self.is_uncertain():
            scale *= 0.50
        return round(scale, 4)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_features(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        closes  = np.asarray(closes,  dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        n = len(closes)
        if n < 6:
            return np.empty((0, 3))

        log_rets = np.diff(np.log(np.maximum(closes, 1e-9)))

        # 5-day rolling volatility of log returns
        rolling_vol = np.array([
            log_rets[max(0, i - 4):i + 1].std()
            for i in range(len(log_rets))
        ])

        # Volume ratio vs 20-day rolling average
        vol_ratio = np.ones(len(log_rets))
        for i in range(len(log_rets)):
            window = volumes[max(0, i - 19):i + 2]  # i+2 because volumes is one longer
            avg = window[:-1].mean() if len(window) > 1 else volumes[i]
            vol_ratio[i] = volumes[i + 1] / avg if avg > 0 else 1.0

        min_len = min(len(log_rets), len(rolling_vol), len(vol_ratio))
        return np.column_stack([
            log_rets[:min_len],
            rolling_vol[:min_len],
            vol_ratio[:min_len],
        ])

    @staticmethod
    def _auto_select_n_regimes(X: np.ndarray, n_min: int = 3, n_max: int = 7) -> int:
        """Test n=3..7 regimes, pick best by BIC score (lower = better)."""
        best_n, best_bic = n_min, np.inf
        for n in range(n_min, n_max + 1):
            try:
                m = GaussianHMM(
                    n_components=n, covariance_type="diag",
                    n_iter=100, random_state=42,
                )
                m.fit(X)
                ll = m.score(X)
                # BIC: -2*logL + params*log(N)
                n_params = n * n + 2 * n * X.shape[1]
                bic = -2 * ll + n_params * np.log(len(X))
                if bic < best_bic:
                    best_bic, best_n = bic, n
            except Exception:
                continue
        return best_n

    @staticmethod
    def _label_regimes(model, n: int) -> dict[int, str]:
        """
        Sort HMM states by mean log return (first feature), lowest → highest.
        Map to regime names ordered from most bearish to most bullish.

        Bear threshold: a state is only labeled "bear" if its mean daily log
        return is below -0.004 (~-0.4%/day, ~-2%/week). States above that
        threshold but still negative are relabeled "neutral" so sideways/
        mildly-negative markets don't block all trading.
        """
        BEAR_THRESHOLD = -0.004   # mean daily log return required to be called "bear"

        means = [model.means_[i][0] for i in range(n)]
        order = np.argsort(means)  # ascending: most bearish first

        name_sets = {
            7: ["crash", "crash", "bear", "neutral", "neutral", "bull", "euphoria"],
            6: ["crash", "bear", "bear", "neutral", "bull", "euphoria"],
            5: ["crash", "bear", "neutral", "bull", "euphoria"],
            4: ["crash", "bear", "bull", "euphoria"],
            3: ["bear", "neutral", "bull"],
            2: ["bear", "bull"],
        }
        names = name_sets.get(n, ["neutral"] * n)
        mapping = {int(order[i]): names[i] for i in range(n)}

        # Relabel "bear" states that don't meet the minimum negativity threshold
        for state_idx, label in mapping.items():
            if label == "bear" and model.means_[state_idx][0] > BEAR_THRESHOLD:
                mapping[state_idx] = "neutral"

        return mapping
