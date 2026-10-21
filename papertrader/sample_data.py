"""Deterministic SYNTHETIC candles for offline testing. NOT real market data; results on it mean nothing financially."""
import csv, math, random
from .strategies import Candle


def generate_candles(n=2000, seed=1, start_price=100.0, interval_s=3600, start_ts=1_700_000_000):
    rnd = random.Random(seed)
    price, ts = start_price, start_ts - start_ts % interval_s
    out, regime, left = [], (0.0, 0.004), 0
    for _ in range(n):
        if left <= 0:
            regime = rnd.choice([(0.0006, 0.004), (-0.0006, 0.006), (0.0, 0.003), (0.0003, 0.008)])
            left = rnd.randint(50, 250)
        left -= 1
        o = price
        path = [o]
        for _ in range(4):
            path.append(path[-1] * math.exp(rnd.gauss(regime[0] / 4, regime[1] / 2)))
        c = path[-1]
        h = max(path) * (1 + abs(rnd.gauss(0, 0.0007)))
        l = min(path) * (1 - abs(rnd.gauss(0, 0.0007)))
        out.append(Candle(float(ts), o, h, l, c, rnd.uniform(50, 500)))
        price, ts = c, ts + interval_s
    return out


def write_csv(path, candles):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ts", "open", "high", "low", "close", "volume"])
        w.writerows([[c.ts, c.o, c.h, c.l, c.c, c.v] for c in candles])


def read_csv(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return [Candle(float(x["ts"]), float(x["open"]), float(x["high"]), float(x["low"]), float(x["close"]), float(x["volume"])) for x in r]
