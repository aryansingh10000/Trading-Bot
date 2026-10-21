"""Modular strategies. A strategy sees ONLY the candles it is given (completed bars) and returns a Signal.
Long-only: 'buy' = enter, 'sell' = exit. The engine ignores buys when holding and sells when flat."""
from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass, field
from . import indicators as ind


class Candle(NamedTuple):
    ts: float
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class Signal:
    action: str = "hold"            # buy | sell | hold
    score: float = 0.0              # 0..1 confidence used by the risk layer
    reason: str = ""
    values: dict = field(default_factory=dict)


def _last(x, k=1):
    return x[-k] if len(x) >= k else None


class Strategy:
    name = ""
    description = ""
    defaults: dict = {}
    lookback = 400

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"unknown parameters for {self.name}: {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        self.validate()

    def validate(self):
        pass

    def window(self, candles):
        return candles[-self.lookback:]

    def signal(self, candles) -> Signal:
        raise NotImplementedError

    def warm(self, why="warming up"):
        return Signal("hold", 0.0, why)


class MACrossover(Strategy):
    name = "ma_crossover"
    description = "Enter when the short moving average crosses above the long one; exit on the opposite cross."
    defaults = {"short": 9, "long": 21, "ma_type": "ema", "min_gap_pct": 0.0}

    def validate(self):
        p = self.params
        if not (0 < p["short"] < p["long"]):
            raise ValueError("need 0 < short < long")
        if p["ma_type"] not in ("ema", "sma"):
            raise ValueError("ma_type must be ema or sma")

    def signal(self, candles):
        p = self.params
        cl = [c.c for c in self.window(candles)]
        fn = ind.ema if p["ma_type"] == "ema" else ind.sma
        s, l = fn(cl, p["short"]), fn(cl, p["long"])
        if len(cl) < p["long"] + 2 or l[-2] is None:
            return self.warm()
        gap = (s[-1] - l[-1]) / l[-1] * 100
        v = {"short_ma": s[-1], "long_ma": l[-1], "gap_pct": gap}
        if s[-2] <= l[-2] and s[-1] > l[-1] and gap >= p["min_gap_pct"]:
            return Signal("buy", min(1.0, 0.6 + gap * 10), f"{p['short']}MA crossed above {p['long']}MA (gap {gap:.3f}%)", v)
        if s[-2] >= l[-2] and s[-1] < l[-1]:
            return Signal("sell", 0.6, f"{p['short']}MA crossed below {p['long']}MA", v)
        return Signal("hold", 0.0, "no crossover", v)


class RSIStrategy(Strategy):
    name = "rsi"
    description = "Buy when RSI recovers up through the oversold line; sell when it exceeds overbought. " \
                  "Optional trend filter skips dip-buys while price is far below its long average (RSI fails in strong downtrends)."
    defaults = {"period": 14, "oversold": 30, "overbought": 70, "trend_filter": True, "trend_ma": 100, "max_below_trend_pct": 8.0}

    def validate(self):
        p = self.params
        if not (0 < p["oversold"] < p["overbought"] < 100):
            raise ValueError("need 0 < oversold < overbought < 100")

    def signal(self, candles):
        p = self.params
        cl = [c.c for c in self.window(candles)]
        r = ind.rsi(cl, p["period"])
        if len(cl) < p["period"] + 3 or r[-2] is None:
            return self.warm()
        v = {"rsi": r[-1], "rsi_prev": r[-2]}
        if r[-1] >= p["overbought"]:
            return Signal("sell", 0.6, f"RSI {r[-1]:.1f} >= overbought {p['overbought']}", v)
        if r[-2] < p["oversold"] <= r[-1]:
            if p["trend_filter"]:
                tm = ind.sma(cl, p["trend_ma"])[-1]
                if tm is not None:
                    below = (tm - cl[-1]) / tm * 100
                    v["below_trend_pct"] = below
                    if below > p["max_below_trend_pct"]:
                        return Signal("hold", 0.0, f"oversold bounce skipped: price {below:.1f}% below {p['trend_ma']}MA (downtrend)", v)
            return Signal("buy", min(1.0, 0.6 + (p["oversold"] - r[-2]) / 50), f"RSI recovered through {p['oversold']} (was {r[-2]:.1f})", v)
        return Signal("hold", 0.0, "RSI neutral", v)


class MACDStrategy(Strategy):
    name = "macd"
    description = "Enter when MACD crosses above its signal line with a rising histogram; exit on the opposite cross."
    defaults = {"fast": 12, "slow": 26, "signal": 9, "require_above_zero": False}

    def validate(self):
        if not (0 < self.params["fast"] < self.params["slow"]):
            raise ValueError("need 0 < fast < slow")

    def signal(self, candles):
        p = self.params
        cl = [c.c for c in self.window(candles)]
        m, sg, h = ind.macd(cl, p["fast"], p["slow"], p["signal"])
        if len(cl) < p["slow"] + p["signal"] + 2 or sg[-2] is None:
            return self.warm()
        v = {"macd": m[-1], "signal": sg[-1], "hist": h[-1]}
        if m[-2] <= sg[-2] and m[-1] > sg[-1] and h[-1] > h[-2] and (not p["require_above_zero"] or m[-1] > 0):
            return Signal("buy", min(1.0, 0.6 + abs(h[-1]) / cl[-1] * 200), "MACD crossed above signal, histogram rising", v)
        if m[-2] >= sg[-2] and m[-1] < sg[-1]:
            return Signal("sell", 0.6, "MACD crossed below signal", v)
        return Signal("hold", 0.0, "no MACD cross", v)


class TrendFollowing(Strategy):
    name = "trend"
    description = "Enter when price > fast EMA > slow EMA, momentum (ROC) is positive enough and structure shows higher highs and higher lows; exit when price falls below the slow EMA."
    defaults = {"fast": 20, "slow": 50, "roc_n": 10, "min_roc_pct": 1.0, "structure_n": 20}

    def validate(self):
        if not (0 < self.params["fast"] < self.params["slow"]):
            raise ValueError("need 0 < fast < slow")

    def signal(self, candles):
        p = self.params
        w = self.window(candles)
        cl, hi, lo = [c.c for c in w], [c.h for c in w], [c.l for c in w]
        n = p["structure_n"]
        if len(cl) < max(p["slow"], 2 * n, p["roc_n"]) + 2:
            return self.warm()
        f, s_ = ind.ema(cl, p["fast"])[-1], ind.ema(cl, p["slow"])[-1]
        r = ind.roc(cl, p["roc_n"])[-1]
        hh = max(hi[-n:]) > max(hi[-2 * n:-n])
        hl = min(lo[-n:]) > min(lo[-2 * n:-n])
        v = {"fast_ema": f, "slow_ema": s_, "roc_pct": r, "higher_highs": hh, "higher_lows": hl}
        if cl[-1] < s_:
            return Signal("sell", 0.6, "price below slow EMA", v)
        if cl[-1] > f > s_ and r >= p["min_roc_pct"] and hh and hl:
            return Signal("buy", min(1.0, 0.6 + r / 20), f"uptrend: price>EMA{p['fast']}>EMA{p['slow']}, ROC {r:.2f}%, higher highs/lows", v)
        return Signal("hold", 0.0, "trend conditions not met", v)


class Combined(Strategy):
    name = "combined"
    description = "Scores trend (0.35), momentum (0.25), RSI zone (0.20) and calm volatility (0.20). Enter at/above enter_score, exit at/below exit_score."
    defaults = {"fast": 12, "slow": 26, "rsi_n": 14, "atr_n": 14, "enter_score": 0.7, "exit_score": 0.3, "max_atr_pct": 3.0}

    def validate(self):
        p = self.params
        if not (0 <= p["exit_score"] < p["enter_score"] <= 1) or not (0 < p["fast"] < p["slow"]):
            raise ValueError("need 0<=exit_score<enter_score<=1 and fast<slow")

    def signal(self, candles):
        p = self.params
        w = self.window(candles)
        cl, hi, lo = [c.c for c in w], [c.h for c in w], [c.l for c in w]
        if len(cl) < p["slow"] + 12:
            return self.warm()
        f, s_ = ind.ema(cl, p["fast"])[-1], ind.ema(cl, p["slow"])[-1]
        _, _, h = ind.macd(cl)
        r = ind.rsi(cl, p["rsi_n"])
        a = ind.atr(hi, lo, cl, p["atr_n"])[-1]
        if h[-2] is None or r[-2] is None or a is None:
            return self.warm()
        atr_pct = a / cl[-1] * 100
        parts = {}
        parts["trend"] = 0.35 if cl[-1] > f > s_ else (0.15 if f > s_ else 0.0)
        parts["momentum"] = 0.25 if (h[-1] > 0 and h[-1] > h[-2]) else (0.12 if h[-1] > 0 else 0.0)
        parts["rsi"] = 0.2 if 45 <= r[-1] <= 65 else (0.1 if 30 < r[-1] < 45 and r[-1] > r[-2] else 0.0)
        parts["volatility"] = 0.2 if atr_pct <= p["max_atr_pct"] else 0.0
        score = round(sum(parts.values()), 4)
        v = {"score_parts": parts, "atr_pct": atr_pct, "rsi": r[-1]}
        if score >= p["enter_score"]:
            return Signal("buy", score, f"combined score {score:.2f} >= {p['enter_score']} {parts}", v)
        if score <= p["exit_score"]:
            return Signal("sell", max(score, 0.3), f"combined score {score:.2f} <= {p['exit_score']}", v)
        return Signal("hold", score, f"combined score {score:.2f} between thresholds", v)


REGISTRY = {c.name: c for c in (MACrossover, RSIStrategy, MACDStrategy, TrendFollowing, Combined)}


def make_strategy(name, params=None):
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
