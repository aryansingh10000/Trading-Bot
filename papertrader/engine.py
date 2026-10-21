"""PaperBot: one decision step per symbol per bar. Used identically by live trading and by the backtester."""
from __future__ import annotations
from .ledger import PaperLedger, Quote
from .risk import RiskManager
from .strategies import Strategy


class PaperBot:
    def __init__(self, ledger: PaperLedger, strategy: Strategy, risk: RiskManager, mode="auto", state="stopped"):
        self.ledger, self.strategy, self.risk = ledger, strategy, risk
        self.mode = mode          # auto | manual (manual: bot only manages exits, never opens)
        self.state = state        # stopped | running | paused (paused: no NEW entries, exits continue)
        self.marks = {}           # symbol -> (bid Decimal, ts)

    def start(self): self.state = "running"
    def pause(self): self.state = "paused"
    def resume(self): self.state = "running"
    def stop(self): self.state = "stopped"

    def fresh_marks(self, now):
        return {k: b for k, (b, ts) in self.marks.items() if now - ts <= self.risk.cfg.max_data_age_s}

    def step(self, symbol, candles, quote: Quote, now, bar_ts, apply_exits=True):
        L, st = self.ledger, self.strategy
        out = {"symbol": symbol, "action": "none", "executed": False}
        self.marks[symbol] = (quote.bid, quote.ts)
        if self.state == "stopped":
            return out
        L.on_quote(quote, now)
        pos = L.position(symbol)
        if apply_exits and pos["qty"] > 0:
            why = self.risk.exit_reason(pos, quote)
            if why:
                return self._exit(symbol, quote, now, bar_ts, why, out)
        sig = st.signal(candles)
        out.update(action=sig.action, score=sig.score, reason=sig.reason)
        if sig.action == "sell" and pos["qty"] > 0:
            return self._exit(symbol, quote, now, bar_ts, f"strategy: {sig.reason}", out, sig)
        if sig.action == "buy":
            if pos["qty"] > 0:
                return out
            blocked = None
            if self.state != "running": blocked = [f"bot is {self.state}"]
            elif self.mode != "auto": blocked = ["manual mode: bot does not open positions"]
            if blocked is None:
                sm = L.summary(self.fresh_marks(now))
                v = self.risk.evaluate_entry(L, quote, sig.score, sm, now)
                if v.allowed:
                    cid = f"{st.name}:{symbol}:{int(bar_ts)}:buy"
                    o = L.submit_order(cid, symbol, "buy", v.qty, quote, "market", now=now, strategy=st.name, reason=sig.reason,
                                       max_slippage_bps=self.risk.cfg.max_slippage_bps)
                    ok = o["status"] in ("FILLED", "PARTIAL", "PARTIAL_CANCELLED")
                    L.log_decision(now, symbol, st.name, "buy", sig.score, ok, cid,
                                   {"signal": sig.reason, "values": sig.values, "risk": v.reasons, "size": v.notional,
                                    "order_status": o["status"], "reject": o["reject_reason"]})
                    out.update(executed=ok, order=o)
                    return out
                blocked = v.reasons
            L.log_decision(now, symbol, st.name, "buy_blocked", sig.score, False, None,
                           {"signal": sig.reason, "values": sig.values, "blocked_by": blocked})
            out["blocked_by"] = blocked
        return out

    def _exit(self, symbol, quote, now, bar_ts, why, out, sig=None):
        cid = f"exit:{self.strategy.name}:{symbol}:{int(bar_ts)}"
        o = self.ledger.close_position(symbol, quote, now, cid, why, self.strategy.name)
        ok = bool(o) and o["status"] in ("FILLED", "PARTIAL", "PARTIAL_CANCELLED")
        self.ledger.log_decision(now, symbol, self.strategy.name, "exit", sig.score if sig else 1.0, ok, cid,
                                 {"reason": why, "order_status": o and o["status"], "reject": o and o["reject_reason"]})
        out.update(action="exit", reason=why, executed=ok, order=o)
        return out
