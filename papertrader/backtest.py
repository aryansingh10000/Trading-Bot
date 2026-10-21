"""Backtester built on the SAME ledger/risk/engine code as live paper trading.

No look-ahead: at bar k the strategy sees only candles[:k] (completed bars) and executes at the OPEN of bar k
(plus spread and slippage). Stops/targets are checked against bar k's range afterwards; stop wins ties; gaps fill at the
worse of open/level. Limitations are returned with every result."""
from __future__ import annotations
import math
from decimal import Decimal as D
from .analytics import perf_stats, max_drawdown_pct
from .engine import PaperBot
from .ledger import PaperLedger, Quote, ExecConfig
from .money import d
from .risk import RiskConfig, RiskManager
from .strategies import Candle, make_strategy

LIMITATIONS = [
    "Backtest, not live results. Past performance does not predict future returns.",
    "Candles carry no order-book depth: fills use an assumed spread plus configured slippage, which may be optimistic in fast markets.",
    "Only market orders are simulated in backtests; stop/take-profit fills are approximated from candle highs/lows.",
    "Synthetic or exchange-supplied history may have gaps/revisions; only pairs that exist today are usually available (survivorship bias).",
]


class DataError(Exception):
    pass


def validate_candles(candles, interval_s):
    """Raise on corrupt data; return list of gap start timestamps (missing candles)."""
    gaps = []
    for i, c in enumerate(candles):
        vals = (c.o, c.h, c.l, c.c, c.v)
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vals) or min(c.o, c.h, c.l, c.c) <= 0 or c.v < 0:
            raise DataError(f"invalid candle values at index {i}")
        if c.h < max(c.o, c.c, c.l) or c.l > min(c.o, c.c, c.h):
            raise DataError(f"inconsistent OHLC at index {i}")
        if i:
            dt = c.ts - candles[i - 1].ts
            if dt <= 0:
                raise DataError(f"timestamps not strictly increasing at index {i}")
            if dt != interval_s:
                gaps.append(candles[i - 1].ts)
    return gaps


def chronological_split(n, warmup=100, train=0.6, val=0.2):
    usable = n - warmup
    if usable < 30:
        raise DataError("not enough candles to split")
    a = warmup + int(usable * train)
    b = a + int(usable * val)
    return {"train": (warmup, a), "validation": (a, b), "test": (b, n)}


def _quote(sym, price, ts, half):
    p = d(price)
    return Quote(sym, p * (1 - half), p * (1 + half), ts)


def run_backtest(data, strategy, params=None, risk=None, execution=None, start=None, end=None, initial=10000,
                 spread_bps=8.0, interval_s=3600, allow_gaps=False, liquidate_at_end=True):
    syms = sorted(data)
    common = set.intersection(*[{c.ts for c in data[s]} for s in syms])
    ser = {s: [c for c in data[s] if c.ts in common] for s in syms}
    gaps = {}
    for s in syms:
        g = validate_candles(ser[s], interval_s)
        if g:
            gaps[s] = len(g)
    if gaps and not allow_gaps:
        raise DataError(f"missing candles detected {gaps}; pass allow_gaps=True to proceed knowingly")
    n = len(ser[syms[0]])
    start = 1 if start is None else max(1, start)
    end = n if end is None else min(n, end)
    if end - start < 2:
        raise DataError("empty backtest range")
    half = D(str(spread_bps)) / 20000
    strat = make_strategy(strategy, params)
    rm = RiskManager(risk or RiskConfig())
    L = PaperLedger(":memory:", initial, execution or ExecConfig())
    bot = PaperBot(L, strat, rm, "auto", "running")
    curve, times = [], []
    lb = strat.lookback
    for k in range(start, end):
        ts = ser[syms[0]][k].ts
        for s in syms:
            bot.marks[s] = (d(ser[s][k].o) * (1 - half), ts)
        for s in syms:
            hist = ser[s][max(0, k - lb):k]                       # completed bars only
            bot.step(s, hist, _quote(s, ser[s][k].o, ts, half), ts, ts)
        for s in syms:                                              # intrabar protective exits on bar k
            c = ser[s][k]
            pos = L.position(s)
            ex = rm.intrabar_exit(pos, c.o, c.h, c.l)
            if ex:
                why, px = ex
                t2 = ts + interval_s / 2
                L.close_position(s, Quote(s, d(px), d(px) * (1 + 2 * half), t2), t2, f"intrabar:{s}:{why}:{int(ts)}", why, strat.name)
            L.update_peak(s, c.h)
        close_marks = {s: d(ser[s][k].c) * (1 - half) for s in syms}
        sm = L.summary(close_marks)
        rm.update(sm["equity"], ts + interval_s)
        L.snapshot(close_marks, ts + interval_s)
        curve.append(float(sm["equity"])); times.append(ts + interval_s)
    if liquidate_at_end:
        last = end - 1
        tl = ser[syms[0]][last].ts + interval_s
        for s in syms:
            L.close_position(s, _quote(s, ser[s][last].c, tl, half), tl, f"final:{s}", "end_of_backtest", strat.name)
        marks = {s: d(ser[s][last].c) * (1 - half) for s in syms}
        curve[-1] = float(L.summary(marks)["equity"])
    trades = L.trades(limit=10**9)
    # equal-weight buy-and-hold benchmark, same fees/spread/slippage, over the same window
    fee = float(L.cfg.taker_fee_bps) / 10000; slip = float(L.cfg.slippage_bps) / 10000; hs = float(half)
    units = {s: (initial / len(syms)) / (1 + fee) / (ser[s][start].o * (1 + hs) * (1 + slip)) for s in syms}
    bh = [sum(units[s] * ser[s][k].c * (1 - hs) for s in syms) for k in range(start, end)]
    bh[-1] *= (1 - fee)
    return {
        "strategy": strategy, "params": strat.params, "bars": [start, end],
        "metrics": perf_stats(curve, trades, initial, interval_s),
        "benchmark": {"return_pct": (bh[-1] - initial) / initial * 100, "max_drawdown_pct": max_drawdown_pct(bh), "curve": bh},
        "equity_curve": list(zip(times, curve)), "trades": trades,
        "data_gaps": gaps, "invariant_problems": L.check_invariants(),
        "limitations": LIMITATIONS,
    }
