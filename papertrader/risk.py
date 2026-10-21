"""Risk layer. Strategies propose; this layer disposes. A strategy can never bypass it."""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal as D
from .money import d, q8_down

ZERO = D(0)


@dataclass
class RiskConfig:
    max_position_pct: float = 10.0      # % of equity in one position
    max_exposure_pct: float = 50.0      # % of equity in all positions
    max_cash_per_trade: float = 1000.0
    max_open_positions: int = 3
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0      # sticky halt until reset_halt()
    min_score: float = 0.6
    max_spread_bps: float = 30.0
    max_slippage_bps: float = 25.0
    max_data_age_s: float = 15.0
    min_notional: float = 10.0
    cooldown_after_losses: int = 3      # 0 disables
    cooldown_s: float = 3600.0
    stop_loss_pct: float | None = 3.0
    take_profit_pct: float | None = 6.0
    trailing_stop_pct: float | None = None


@dataclass
class Verdict:
    allowed: bool
    qty: D = ZERO
    notional: D = ZERO
    reasons: list = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        self.peak = None
        self.day = None
        self.day_start = None
        self.halt = None

    def update(self, equity, now):
        """Feed current equity (must be a complete valuation). Maintains peak, daily baseline and drawdown halt."""
        if equity is None:
            return
        equity = d(equity)
        day = int(now // 86400)
        if day != self.day:
            self.day, self.day_start = day, equity
        if self.peak is None or equity > self.peak:
            self.peak = equity
        dd = (self.peak - equity) / self.peak * 100
        if dd >= self.cfg.max_drawdown_pct and not self.halt:
            self.halt = f"max drawdown {dd:.2f}% >= {self.cfg.max_drawdown_pct}% - new entries blocked until reset_halt()"

    def reset_halt(self, equity=None):
        self.halt = None
        if equity is not None:
            self.peak = d(equity)

    def evaluate_entry(self, ledger, quote, score, summary, now) -> Verdict:
        c, why = self.cfg, []
        if summary is None or not summary["complete"]:
            return Verdict(False, reasons=["cannot value portfolio (missing/stale mark prices) - failing safe"])
        equity = summary["equity"]
        self.update(equity, now)
        if self.halt: why.append(self.halt)
        if self.day_start and (self.day_start - equity) / self.day_start * 100 >= c.max_daily_loss_pct:
            why.append(f"daily loss limit {c.max_daily_loss_pct}% reached")
        age = now - quote.ts
        if age > c.max_data_age_s: why.append(f"market data stale ({age:.0f}s > {c.max_data_age_s}s)")
        if quote.spread_bps > d(c.max_spread_bps): why.append(f"spread {float(quote.spread_bps):.1f}bps > {c.max_spread_bps}")
        if score < c.min_score: why.append(f"score {score:.2f} < min {c.min_score}")
        held = [p for p in summary["positions"]]
        if any(p["symbol"] == quote.symbol for p in held): why.append("already holding this symbol")
        if len(held) >= c.max_open_positions: why.append(f"max open positions ({c.max_open_positions}) reached")
        if c.cooldown_after_losses:
            recent = ledger.trades(limit=c.cooldown_after_losses)
            if len(recent) == c.cooldown_after_losses and all(float(t["net_pnl"]) < 0 for t in recent) \
                    and now - recent[-1]["exit_ts"] < c.cooldown_s:
                why.append(f"cooldown after {c.cooldown_after_losses} consecutive losses")
        eq = d(equity)
        room = eq * d(c.max_exposure_pct) / 100 - summary["market_value"] - sum(
            (o["reserved"] for o in ledger.open_orders() if o["side"] == "buy"), ZERO)
        notional = min(eq * d(c.max_position_pct) / 100, d(c.max_cash_per_trade), summary["cash"] / D("1.01"), room)
        if notional < d(c.min_notional):
            why.append(f"allowed size ${float(max(notional, ZERO)):.2f} below minimum ${c.min_notional} (exposure/cash limits)")
        if why:
            return Verdict(False, reasons=why)
        qty = q8_down(notional / quote.ask)
        return Verdict(True, qty=qty, notional=notional, reasons=["all risk checks passed"])

    def exit_reason(self, pos, quote):
        """Stop-loss / take-profit / trailing stop evaluated on the bid."""
        c = self.cfg
        if pos["qty"] <= 0:
            return None
        entry = pos["gross_cost"] / pos["qty"]
        b = quote.bid
        if c.stop_loss_pct and b <= entry * d(1 - c.stop_loss_pct / 100): return "stop_loss"
        if c.take_profit_pct and b >= entry * d(1 + c.take_profit_pct / 100): return "take_profit"
        if c.trailing_stop_pct and pos["peak_price"] and b <= pos["peak_price"] * d(1 - c.trailing_stop_pct / 100): return "trailing_stop"
        return None

    def intrabar_exit(self, pos, o, h, l):
        """Backtest helper: did this candle's range hit a protective level? Stop wins ties (conservative). Gap-aware."""
        c = self.cfg
        if pos["qty"] <= 0:
            return None
        entry = float(pos["gross_cost"] / pos["qty"])
        peak = float(pos["peak_price"]) if pos["peak_price"] else entry
        stops = []
        if c.stop_loss_pct: stops.append(("stop_loss", entry * (1 - c.stop_loss_pct / 100)))
        if c.trailing_stop_pct: stops.append(("trailing_stop", peak * (1 - c.trailing_stop_pct / 100)))
        hit = [(n, lv) for n, lv in stops if l <= lv]
        if hit:
            n, lv = max(hit, key=lambda x: x[1])
            return n, min(o, lv)
        if c.take_profit_pct:
            lv = entry * (1 + c.take_profit_pct / 100)
            if h >= lv:
                return "take_profit", max(o, lv)
        return None
