"""Offline CLI. Usage:
  python -m papertrader.cli backtest --strategy macd [--csv file.csv] [--fee-bps 40 --slippage-bps 5]
  python -m papertrader.cli research --strategy ma_crossover
  python -m papertrader.cli sample-data --out ./sample_data
Without --csv it uses SYNTHETIC candles (not real market data)."""
import argparse, json, os
from .backtest import run_backtest
from .ledger import ExecConfig
from .research import evaluate
from .sample_data import generate_candles, write_csv, read_csv
from .strategies import REGISTRY

GRIDS = {"ma_crossover": {"short": [5, 9, 12], "long": [21, 34, 55]}, "rsi": {"oversold": [25, 30, 35], "overbought": [65, 70, 75]},
         "macd": {"fast": [8, 12], "slow": [21, 26, 35]}, "trend": {"fast": [10, 20], "slow": [40, 50, 80]},
         "combined": {"enter_score": [0.65, 0.7, 0.8], "max_atr_pct": [2.0, 3.0]}}


def main():
    ap = argparse.ArgumentParser(prog="papertrader")
    ap.add_argument("cmd", choices=["backtest", "research", "sample-data"])
    ap.add_argument("--strategy", default="ma_crossover", choices=sorted(REGISTRY))
    ap.add_argument("--csv", nargs="*")
    ap.add_argument("--fee-bps", type=float, default=40); ap.add_argument("--slippage-bps", type=float, default=5)
    ap.add_argument("--out", default="sample_data")
    a = ap.parse_args()
    if a.cmd == "sample-data":
        os.makedirs(a.out, exist_ok=True)
        for i, seed in enumerate((1, 2, 3), 1):
            write_csv(f"{a.out}/SYN{i}-USD_1h.csv", generate_candles(2000, seed, 100 / i))
        return print(f"wrote SYNTHETIC candles to {a.out}/ (for offline testing only)")
    data = {os.path.basename(p).split("_")[0]: read_csv(p) for p in a.csv} if a.csv else \
        {"SYN1-USD": generate_candles(2000, 1), "SYN2-USD": generate_candles(2000, 2, 50)}
    ex = ExecConfig(taker_fee_bps=a.fee_bps, slippage_bps=a.slippage_bps)
    if a.cmd == "backtest":
        r = run_backtest(data, a.strategy, execution=ex)
        m = r["metrics"]
        print(json.dumps({"strategy": a.strategy, "params": r["params"], "metrics": m, "buy_and_hold_return_pct": r["benchmark"]["return_pct"],
                          "data_gaps": r["data_gaps"], "accounting_problems": r["invariant_problems"]}, indent=2, default=str))
        print("\nLIMITATIONS:\n- " + "\n- ".join(r["limitations"]))
    else:
        rep = evaluate(data, a.strategy, REGISTRY[a.strategy].defaults, GRIDS[a.strategy], execution=ex)
        print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
