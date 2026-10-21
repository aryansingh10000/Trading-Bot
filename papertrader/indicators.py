"""Pure-Python indicators. Every value at index i depends only on inputs[0..i] (no look-ahead)."""
import math


def sma(v, n):
    out = [None] * len(v)
    s = 0.0
    for i, x in enumerate(v):
        s += x
        if i >= n:
            s -= v[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(v, n):
    out = [None] * len(v)
    k = 2.0 / (n + 1)
    e = None
    for i, x in enumerate(v):
        if i == n - 1:
            e = sum(v[:n]) / n
            out[i] = e
        elif i >= n:
            e = x * k + e * (1 - k)
            out[i] = e
    return out


def _rs(ag, al):
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def rsi(v, n=14):
    out = [None] * len(v)
    if len(v) <= n:
        return out
    g = [max(v[i] - v[i - 1], 0.0) for i in range(1, len(v))]
    l = [max(v[i - 1] - v[i], 0.0) for i in range(1, len(v))]
    ag, al = sum(g[:n]) / n, sum(l[:n]) / n
    out[n] = _rs(ag, al)
    for i in range(n, len(g)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        out[i + 1] = _rs(ag, al)
    return out


def macd(v, fast=12, slow=26, signal=9):
    ef, es = ema(v, fast), ema(v, slow)
    line = [None if (a is None or b is None) else a - b for a, b in zip(ef, es)]
    idx = [i for i, x in enumerate(line) if x is not None]
    sig = [None] * len(v)
    hist = [None] * len(v)
    if idx:
        compact = ema([line[i] for i in idx], signal)
        for j, i in enumerate(idx):
            sig[i] = compact[j]
            if compact[j] is not None:
                hist[i] = line[i] - compact[j]
    return line, sig, hist


def bollinger(v, n=20, k=2.0):
    mid = sma(v, n)
    up, lo = [None] * len(v), [None] * len(v)
    for i in range(n - 1, len(v)):
        w = v[i - n + 1:i + 1]
        sd = math.sqrt(sum((x - mid[i]) ** 2 for x in w) / n)
        up[i], lo[i] = mid[i] + k * sd, mid[i] - k * sd
    return up, mid, lo


def atr(h, l, c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    a = sum(tr[1:n + 1]) / n
    out[n] = a
    for i in range(n + 1, len(c)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def roc(v, n):
    return [None if i < n or v[i - n] == 0 else (v[i] / v[i - n] - 1) * 100 for i in range(len(v))]


def volatility(v, n=30):
    """Std-dev of log returns over the last n bars (per-bar, not annualised)."""
    out = [None] * len(v)
    r = [None] + [math.log(v[i] / v[i - 1]) for i in range(1, len(v))]
    for i in range(n, len(v)):
        w = r[i - n + 1:i + 1]
        m = sum(w) / n
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / (n - 1))
    return out
