import math


def max_drawdown_pct(curve):
    peak, mdd = None, 0.0
    for e in curve:
        peak = e if peak is None or e > peak else peak
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak * 100)
    return mdd


def perf_stats(curve, trades, initial, interval_s=3600):
    """curve: list of equity floats (one per bar). trades: ledger trade dicts."""
    initial = float(initial)
    pnl = [float(t["net_pnl"]) for t in trades]
    wins, losses = [x for x in pnl if x > 0], [x for x in pnl if x <= 0]
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve)) if curve[i - 1] > 0]
    sharpe = None
    if len(rets) >= 30:
        m = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
        if sd > 0:
            sharpe = m / sd * math.sqrt(365 * 86400 / interval_s)
    gl = -sum(losses)
    final = curve[-1] if curve else initial
    return {
        "total_return_pct": (final - initial) / initial * 100,
        "max_drawdown_pct": max_drawdown_pct(curve),
        "sharpe_annualised": sharpe,                 # None when < 30 bars: not enough data to be meaningful
        "trades": len(pnl),
        "win_rate_pct": (len(wins) / len(pnl) * 100) if pnl else None,
        "avg_win": (sum(wins) / len(wins)) if wins else None,
        "avg_loss": (sum(losses) / len(losses)) if losses else None,
        "largest_win": max(wins) if wins else None,
        "largest_loss": min(losses) if losses else None,
        "profit_factor": (sum(wins) / gl) if gl > 0 else None,
        "avg_hold_s": (sum(t["hold_s"] for t in trades) / len(trades)) if trades else None,
        "fees_paid": sum(float(t["fees"]) for t in trades),
        "gross_pnl_before_fees": sum(pnl) + sum(float(t["fees"]) for t in trades),
        "net_pnl": sum(pnl),
    }
