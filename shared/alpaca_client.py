"""
Shared Alpaca client wrapper used by Data Agent, Execution Agent, and Risk Agent.
"""

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from typing import Optional

from config import AlpacaConfig
from loguru import logger


class AlpacaClient:
    """Thin wrapper around alpaca-py SDK."""

    @staticmethod
    def _mask(key: str) -> str:
        """Return a masked credential string for safe logging."""
        if not key:
            return "(not set)"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("forbidden", "unauthorized", "403", "401",
                                      "invalid key", "x-api-key", "authentication"))

    def __init__(self):
        self.trading = TradingClient(
            api_key=AlpacaConfig.API_KEY,
            secret_key=AlpacaConfig.SECRET_KEY,
            paper=AlpacaConfig.PAPER,
        )
        self.data = StockHistoricalDataClient(
            api_key=AlpacaConfig.API_KEY,
            secret_key=AlpacaConfig.SECRET_KEY,
        )
        logger.info(
            f"AlpacaClient initialised | paper={AlpacaConfig.PAPER} | "
            f"api_key={self._mask(AlpacaConfig.API_KEY)} | "
            f"secret_key={self._mask(AlpacaConfig.SECRET_KEY)}"
        )

    # ── Account ────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        try:
            acct = self.trading.get_account()
        except Exception as e:
            if self._is_auth_error(e):
                raise RuntimeError(
                    f"Alpaca authentication failed — check your credentials.\n"
                    f"  ALPACA_API_KEY    = {self._mask(AlpacaConfig.API_KEY)}\n"
                    f"  ALPACA_SECRET_KEY = {self._mask(AlpacaConfig.SECRET_KEY)}\n"
                    f"  BASE_URL          = {AlpacaConfig.BASE_URL}\n"
                    f"  Original error: {e}"
                ) from e
            raise
        return {
            "equity":          float(acct.equity),
            "cash":            float(acct.cash),
            "buying_power":    float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "day_trade_count": acct.daytrade_count,
        }

    def get_positions(self) -> list[dict]:
        positions = self.trading.get_all_positions()
        return [
            {
                "symbol":      p.symbol,
                "qty":         float(p.qty),
                "avg_entry":   float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "market_value":  float(p.market_value),
            }
            for p in positions
        ]

    def get_orders(self) -> list[dict]:
        orders = self.trading.get_orders()
        return [
            {
                "id":       str(o.id),
                "symbol":   o.symbol,
                "qty":      float(o.qty) if o.qty else 0,
                "side":     o.side.value,
                "type":     o.type.value,
                "status":   o.status.value,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
            }
            for o in orders
        ]

    # ── Market data ────────────────────────────────────────────────────────────

    def get_live_price(self, symbol: str, side: str = "buy") -> Optional[float]:
        """
        Fetch the real-time best price for one symbol from Alpaca.
        Returns ask for buy orders, bid for sell orders, None on failure.
        This is the ONLY safe price source for order sizing.
        """
        try:
            req    = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data.get_stock_latest_quote(req)
            q      = quotes.get(symbol)
            if q is None:
                return None
            if side == "sell":
                p = float(q.bid_price) if q.bid_price else None
            else:
                p = float(q.ask_price) if q.ask_price else None
            # Fall back to last trade price if quote is absent
            if not p or p <= 0:
                treq   = StockLatestTradeRequest(symbol_or_symbols=[symbol])
                trades = self.data.get_stock_latest_trade(treq)
                t      = trades.get(symbol)
                p      = float(t.price) if t and t.price else None
            return p
        except Exception:
            return None

    def get_latest_quotes(self, symbols: list[str]) -> dict:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self.data.get_stock_latest_quote(req)
        result = {}
        for sym, q in quotes.items():
            result[sym] = {
                "ask":       float(q.ask_price),
                "bid":       float(q.bid_price),
                "ask_size":  float(q.ask_size),
                "bid_size":  float(q.bid_size),
            }
        return result

    def get_bars(self, symbols: list[str], timeframe: TimeFrame,
                 start: str, end: str = None) -> dict:
        from datetime import datetime, timezone
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        try:
            bars = self.data.get_stock_bars(req)
        except Exception as e:
            if self._is_auth_error(e):
                raise RuntimeError(
                    f"Alpaca data authentication failed — check your credentials.\n"
                    f"  ALPACA_API_KEY    = {self._mask(AlpacaConfig.API_KEY)}\n"
                    f"  ALPACA_SECRET_KEY = {self._mask(AlpacaConfig.SECRET_KEY)}\n"
                    f"  Original error: {e}"
                ) from e
            raise
        result = {}
        bar_data = bars.data if hasattr(bars, "data") else bars
        for sym, bar_list in bar_data.items():
            result[sym] = [
                {
                    "timestamp": b.timestamp.isoformat(),
                    "open":   float(b.open),
                    "high":   float(b.high),
                    "low":    float(b.low),
                    "close":  float(b.close),
                    "volume": float(b.volume),
                    "vwap":   float(b.vwap) if b.vwap else None,
                }
                for b in bar_list
            ]
        return result

    # ── Orders ─────────────────────────────────────────────────────────────────

    def submit_market_order(self, symbol: str, qty: float,
                            side: str, time_in_force: str = "day") -> dict:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC,
        )
        order = self.trading.submit_order(req)
        return {"id": str(order.id), "status": order.status.value, "symbol": symbol}

    def cancel_order(self, order_id: str):
        self.trading.cancel_order_by_id(order_id)

    def cancel_all_orders(self):
        self.trading.cancel_orders()

    def close_position(self, symbol: str):
        self.trading.close_position(symbol)

    def close_all_positions(self):
        self.trading.close_all_positions(cancel_orders=True)
