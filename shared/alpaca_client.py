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
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from config import AlpacaConfig
from loguru import logger


class AlpacaClient:
    """Thin wrapper around alpaca-py SDK."""

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
        logger.info(f"AlpacaClient initialised | paper={AlpacaConfig.PAPER}")

    # ── Account ────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        acct = self.trading.get_account()
        return {
            "equity":        float(acct.equity),
            "cash":          float(acct.cash),
            "buying_power":  float(acct.buying_power),
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
        bars = self.data.get_stock_bars(req)
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
