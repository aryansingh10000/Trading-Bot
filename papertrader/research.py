"""Controlled strategy-research pipeline. It can only RECOMMEND a candidate for forward testing.
Promotion to the active strategy is never automatic and always needs the user's approval."""
from __future__ import annotations
import itertools
from .backtest import run_backtest, chronological_split


def propose_candidates(base_params, grid, max_candidates=40):
    keys = sorted(grid)
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = {**base_params, **dict(zip(keys, combo))}
        if p != base_params:
            out.append(p)
    return out[:max_candidates]


def _score(m):
    """Risk-adjusted ranking: return per unit of drawdown, ignoring strategies with too few trades."""
    return m["total_return_pct"] / max(m["max_drawdown_pct"], 1.0)


def evaluate(data, strategy, base_params, grid, risk=None, execution=None, interval_s=3600, warmup=100,
             min_trades=10, max_drawdown_pct=15.0, top_k=3, **bt):
    n = len(next(iter(data.values())))
    sp = chronological_split(n, warmup)
    run = lambda params, seg: run_backtest(data, strategy, params, risk, execution, *sp[seg], interval_s=interval_s, **bt)
    base = {seg: run(base_params, seg) for seg in sp}
    cands = propose_candidates(base_params, grid)
    scored = []
    for p in cands:                                      # selection uses TRAIN data only
        r = run(p, "train")
        if r["metrics"]["trades"] >= min_trades:
            scored.append((_score(r["metrics"]), p, r))
    scored.sort(key=lambda x: -x[0])
    report = {"strategy": strategy, "candidates_tried": len(cands), "selection_pool": len(scored),
              "multiple_testing_note": f"{len(cands)} configurations were tried; the best of many random-ish trials looks good by chance, so only out-of-sample results count.",
              "baseline": {s: base[s]["metrics"] | {"buy_hold_return_pct": base[s]["benchmark"]["return_pct"]} for s in sp},
              "finalists": [], "recommendation": None}
    for _, p, tr in scored[:top_k]:
        val, test = run(p, "validation"), run(p, "test")
        mv, mt = val["metrics"], test["metrics"]
        fails = []
        if mv["trades"] < min_trades: fails.append(f"validation trades {mv['trades']} < {min_trades}")
        if mv["max_drawdown_pct"] > max_drawdown_pct: fails.append("validation drawdown too high")
        if mt["max_drawdown_pct"] > max_drawdown_pct: fails.append("test drawdown too high")
        if mv["total_return_pct"] <= 0 or mt["total_return_pct"] <= 0: fails.append("not profitable net of costs out-of-sample")
        if mv["total_return_pct"] <= base["validation"]["metrics"]["total_return_pct"]: fails.append("does not beat baseline on validation")
        if mt["total_return_pct"] <= base["test"]["metrics"]["total_return_pct"]: fails.append("does not beat baseline on test")
        tr_ret = tr["metrics"]["total_return_pct"]
        if tr_ret > 0 and mv["total_return_pct"] < 0.25 * tr_ret: fails.append("validation return collapsed vs train (overfitting signal)")
        report["finalists"].append({"params": p, "train": tr["metrics"], "validation": mv, "test": mt, "failures": fails,
                                    "passed": not fails, "status": "recommend_forward_test" if not fails else "rejected"})
    ok = [f for f in report["finalists"] if f["passed"]]
    if ok:
        report["recommendation"] = {"params": ok[0]["params"], "next_step": "forward-test with virtual money against the baseline; "
                                    "activation requires manual approval"}
    return report
