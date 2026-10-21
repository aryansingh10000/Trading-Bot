"""Paper ledger: Decimal accounting, simulated order execution, SQLite persistence.

Accounting rules (long-only spot):
  equity = available cash + reserved cash + market value of positions (marked at BID)
  position.cost_basis includes buy fees; realized P&L is net of buy and sell fees.
  Invariant: cash + reserved + cost_basis == initial + realized   (checked by check_invariants)
Every state change happens inside one BEGIN IMMEDIATE transaction. client_order_id is UNIQUE,
so retries/restarts can never create a duplicate order or fill.
"""
from __future__ import annotations
import sqlite3, threading, time
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal as D
from typing import Optional

from .money import d, q8, s

ZERO = D(0)


class AccountingError(Exception):
    pass


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: D
    ask: D
    ts: float
    bid_size: Optional[D] = None
    ask_size: Optional[D] = None

    @staticmethod
    def of(symbol, bid, ask, ts, bid_size=None, ask_size=None):
        return Quote(symbol, d(bid), d(ask), float(ts),
                     None if bid_size is None else d(bid_size),
                     None if ask_size is None else d(ask_size))

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self):
        return (self.ask - self.bid) / self.mid * 10000


@dataclass
class ExecConfig:
    maker_fee_bps: D = D("25")
    taker_fee_bps: D = D("40")
    slippage_bps: D = D("5")          # base slippage applied to taker fills
    impact_bps: D = D("10")           # extra bps if order consumes the whole visible book level
    max_quote_age_s: float = 30.0
    allow_partial: bool = True

    def __post_init__(self):
        for f in ("maker_fee_bps", "taker_fee_bps", "slippage_bps", "impact_bps"):
            setattr(self, f, d(getattr(self, f)))


SCHEMA = """
CREATE TABLE IF NOT EXISTS account(id INTEGER PRIMARY KEY CHECK(id=1), initial TEXT NOT NULL,
  cash TEXT NOT NULL, reserved TEXT NOT NULL, epoch INTEGER NOT NULL, created_ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, qty TEXT NOT NULL, reserved_qty TEXT NOT NULL,
  cost_basis TEXT NOT NULL, gross_cost TEXT NOT NULL, realized TEXT NOT NULL, peak_price TEXT,
  opened_ts REAL, strategy TEXT);
CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY AUTOINCREMENT, cid TEXT NOT NULL UNIQUE, ts REAL NOT NULL,
  symbol TEXT NOT NULL, side TEXT NOT NULL CHECK(side IN('buy','sell')), type TEXT NOT NULL CHECK(type IN('market','limit')),
  qty TEXT NOT NULL, limit_price TEXT, filled_qty TEXT NOT NULL DEFAULT '0', filled_value TEXT NOT NULL DEFAULT '0',
  status TEXT NOT NULL, reserved TEXT NOT NULL DEFAULT '0', rpu TEXT NOT NULL DEFAULT '0', expires_ts REAL,
  strategy TEXT, reason TEXT, reject_reason TEXT, max_slip TEXT);
CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL REFERENCES orders(id),
  ts REAL NOT NULL, symbol TEXT, side TEXT, qty TEXT, price TEXT, fee TEXT, liquidity TEXT, slippage_bps TEXT, spread_bps TEXT);
CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, strategy TEXT, entry_ts REAL, exit_ts REAL,
  qty TEXT, entry_price TEXT, exit_price TEXT, fees TEXT, net_pnl TEXT, hold_s REAL, reason TEXT, order_id INTEGER);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, epoch INTEGER, kind TEXT, symbol TEXT, ref TEXT,
  dcash TEXT, dreserved TEXT, cash_after TEXT, reserved_after TEXT);
CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, epoch INTEGER, equity TEXT, cash TEXT,
  reserved TEXT, market_value TEXT, realized TEXT, unrealized TEXT);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, symbol TEXT, strategy TEXT, action TEXT,
  score REAL, executed INTEGER, order_cid TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_orders_open ON orders(status, symbol);
"""
OPEN = ("NEW", "OPEN", "PARTIAL")


class PaperLedger:
    def __init__(self, path=":memory:", initial_balance="10000", config: ExecConfig | None = None):
        self.cfg = config or ExecConfig()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        if self.db.execute("SELECT COUNT(*) FROM account").fetchone()[0] == 0:
            self._new_account(d(initial_balance), time.time(), 1)

    # ---------- plumbing ----------
    @contextmanager
    def _tx(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _new_account(self, initial, now, epoch):
        if initial <= 0:
            raise ValueError("initial balance must be positive")
        with self._tx():
            self.db.execute("INSERT OR REPLACE INTO account VALUES(1,?,?,?,?,?)", (s(initial), s(initial), "0", epoch, now))
            self.db.execute("INSERT INTO audit(ts,epoch,kind,dcash,dreserved,cash_after,reserved_after) VALUES(?,?,?,?,?,?,?)",
                            (now, epoch, "init", s(initial), "0", s(initial), "0"))

    def _acct(self):
        r = self.db.execute("SELECT * FROM account").fetchone()
        return d(r["initial"]), d(r["cash"]), d(r["reserved"]), r["epoch"]

    def _adjust(self, dcash, dres, kind, symbol, ref, now):
        r = self.db.execute("SELECT cash,reserved,epoch FROM account").fetchone()
        cash, res = d(r["cash"]) + dcash, d(r["reserved"]) + dres
        if cash < 0 or res < 0:
            raise AccountingError(f"{kind} would make balances negative (cash={cash}, reserved={res})")
        self.db.execute("UPDATE account SET cash=?,reserved=?", (s(cash), s(res)))
        self.db.execute("INSERT INTO audit(ts,epoch,kind,symbol,ref,dcash,dreserved,cash_after,reserved_after) VALUES(?,?,?,?,?,?,?,?,?)",
                        (now, r["epoch"], kind, symbol, ref, s(dcash), s(dres), s(cash), s(res)))

    def _pos(self, symbol):
        r = self.db.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if not r:
            return dict(symbol=symbol, qty=ZERO, reserved_qty=ZERO, cost_basis=ZERO, gross_cost=ZERO, realized=ZERO,
                        peak_price=None, opened_ts=None, strategy=None)
        return dict(symbol=symbol, qty=d(r["qty"]), reserved_qty=d(r["reserved_qty"]), cost_basis=d(r["cost_basis"]),
                    gross_cost=d(r["gross_cost"]), realized=d(r["realized"]),
                    peak_price=None if r["peak_price"] is None else d(r["peak_price"]),
                    opened_ts=r["opened_ts"], strategy=r["strategy"])

    def _save_pos(self, p):
        self.db.execute("INSERT OR REPLACE INTO positions VALUES(?,?,?,?,?,?,?,?,?)",
                        (p["symbol"], s(p["qty"]), s(p["reserved_qty"]), s(p["cost_basis"]), s(p["gross_cost"]), s(p["realized"]),
                         None if p["peak_price"] is None else s(p["peak_price"]), p["opened_ts"], p["strategy"]))

    def _order(self, where, args):
        r = self.db.execute(f"SELECT * FROM orders WHERE {where}", args).fetchone()
        return self._o(r) if r else None

    @staticmethod
    def _o(r):
        o = dict(r)
        for k in ("qty", "filled_qty", "filled_value", "reserved", "rpu"):
            o[k] = d(o[k])
        o["limit_price"] = None if o["limit_price"] is None else d(o["limit_price"])
        o["max_slip"] = None if o["max_slip"] is None else d(o["max_slip"])
        return o

    def _save_order(self, o):
        self.db.execute("UPDATE orders SET filled_qty=?,filled_value=?,status=?,reserved=?,reject_reason=? WHERE id=?",
                        (s(o["filled_qty"]), s(o["filled_value"]), o["status"], s(o["reserved"]), o["reject_reason"], o["id"]))

    # ---------- public reads ----------
    def get_order(self, cid):
        with self.lock:
            return self._order("cid=?", (cid,))

    def open_orders(self, symbol=None):
        with self.lock:
            q = "SELECT * FROM orders WHERE status IN ('NEW','OPEN','PARTIAL')" + (" AND symbol=?" if symbol else "") + " ORDER BY id"
            return [self._o(r) for r in self.db.execute(q, (symbol,) if symbol else ())]

    def position(self, symbol):
        with self.lock:
            return self._pos(symbol)

    def positions(self):
        with self.lock:
            return [self._pos(r["symbol"]) for r in self.db.execute("SELECT symbol FROM positions WHERE CAST(qty AS REAL) > 0")]

    def trades(self, limit=1000, symbol=None, strategy=None):
        with self.lock:
            q, a = "SELECT * FROM trades WHERE 1=1", []
            if symbol:
                q += " AND symbol=?"; a.append(symbol)
            if strategy:
                q += " AND strategy=?"; a.append(strategy)
            q += " ORDER BY id DESC LIMIT ?"; a.append(limit)
            return [dict(r) for r in self.db.execute(q, a)][::-1]

    def fills(self, order_cid):
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT f.* FROM fills f JOIN orders o ON o.id=f.order_id WHERE o.cid=? ORDER BY f.id", (order_cid,))]

    def snapshots(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM snapshots ORDER BY id")]

    def decisions(self, limit=200):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))]

    def summary(self, marks: dict):
        """marks: symbol -> Decimal price (use BID). Missing marks => equity None (we never invent prices)."""
        with self.lock:
            initial, cash, res, _ = self._acct()
            mv = unreal = ZERO
            missing, rows = [], []
            realized = sum((d(r["realized"]) for r in self.db.execute("SELECT realized FROM positions")), ZERO)
            for p in self.positions():
                m = marks.get(p["symbol"])
                if m is None:
                    missing.append(p["symbol"]); continue
                v = q8(p["qty"] * d(m))
                mv += v; unreal += v - p["cost_basis"]
                rows.append(dict(symbol=p["symbol"], qty=p["qty"], avg_entry=p["gross_cost"] / p["qty"], mark=d(m),
                                 market_value=v, unrealized=v - p["cost_basis"]))
            complete = not missing
            equity = cash + res + mv if complete else None
            return dict(initial=initial, cash=cash, reserved=res, market_value=mv, equity=equity, realized=realized,
                        unrealized=unreal, complete=complete, missing_marks=missing, positions=rows,
                        net_pnl=None if equity is None else equity - initial,
                        return_pct=None if equity is None else (equity - initial) / initial * 100,
                        exposure=None if not equity else mv / equity)

    def snapshot(self, marks, now=None):
        now = now or time.time()
        with self._tx():
            sm = self.summary(marks)
            if not sm["complete"]:
                return None
            self.db.execute("INSERT INTO snapshots(ts,epoch,equity,cash,reserved,market_value,realized,unrealized) VALUES(?,?,?,?,?,?,?,?)",
                            (now, self._acct()[3], s(sm["equity"]), s(sm["cash"]), s(sm["reserved"]), s(sm["market_value"]),
                             s(sm["realized"]), s(sm["unrealized"])))
            return sm

    def log_decision(self, now, symbol, strategy, action, score, executed, order_cid, detail):
        import json
        with self._tx():
            self.db.execute("INSERT INTO decisions(ts,symbol,strategy,action,score,executed,order_cid,detail) VALUES(?,?,?,?,?,?,?,?)",
                            (now, symbol, strategy, action, score, int(executed), order_cid, json.dumps(detail, default=str)))

    # ---------- order entry ----------
    def submit_order(self, cid, symbol, side, qty, quote: Quote, type="market", limit_price=None, now=None,
                     expires_in=None, strategy="manual", reason="", max_slippage_bps=None):
        now = time.time() if now is None else now
        if side not in ("buy", "sell") or type not in ("market", "limit") or not cid:
            raise ValueError("invalid side/type/client_order_id")
        qty = q8(qty)
        if qty <= 0:
            raise ValueError("qty must be positive")
        lp = None
        if type == "limit":
            if limit_price is None or d(limit_price) <= 0:
                raise ValueError("limit order needs a positive limit_price")
            lp = q8(limit_price)
        with self._tx():
            existing = self._order("cid=?", (cid,))
            if existing:
                return existing            # idempotent: same client_order_id never executes twice
            self.db.execute("INSERT INTO orders(cid,ts,symbol,side,type,qty,limit_price,status,expires_ts,strategy,reason,max_slip) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, now, symbol, side, type, s(qty), None if lp is None else s(lp), "NEW",
                             None if expires_in is None else now + expires_in, strategy, reason,
                             None if max_slippage_bps is None else s(d(max_slippage_bps))))
            o = self._order("cid=?", (cid,))
            self._place(o, quote, now)
            return self._order("cid=?", (cid,))

    def _reject(self, o, why):
        o["status"], o["reject_reason"] = "REJECTED", why
        self._save_order(o)

    def _quote_problem(self, q: Quote, symbol, now):
        if q is None or q.symbol != symbol:
            return "no quote for symbol"
        if q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
            return "invalid or crossed quote"
        if now - q.ts > self.cfg.max_quote_age_s:
            return f"stale quote ({now - q.ts:.0f}s old)"
        if now - q.ts < -5:
            return "quote timestamp is in the future"
        return None

    def _place(self, o, q, now):
        why = self._quote_problem(q, o["symbol"], now)
        if why:
            return self._reject(o, why)
        pos = self._pos(o["symbol"])
        if o["side"] == "sell":
            if o["qty"] > pos["qty"] - pos["reserved_qty"]:
                return self._reject(o, "insufficient position quantity")
            if o["type"] == "limit":
                pos["reserved_qty"] += o["qty"]; o["reserved"] = o["qty"]; self._save_pos(pos)
        elif o["type"] == "limit":
            rpu = o["limit_price"] * (1 + self.cfg.taker_fee_bps / 10000)
            need = q8(o["qty"] * rpu)
            if need > self._acct()[1]:
                return self._reject(o, "insufficient available cash")
            self._adjust(-need, need, "reserve", o["symbol"], o["cid"], now)
            o["reserved"] = need
            self.db.execute("UPDATE orders SET rpu=? WHERE id=?", (s(rpu), o["id"]))
            o["rpu"] = rpu
        marketable = o["type"] == "market" or (o["side"] == "buy" and q.ask <= o["limit_price"]) or \
                     (o["side"] == "sell" and q.bid >= o["limit_price"])
        if marketable:
            plan = self._plan(o, q, "taker")
            if plan is None:
                if o["type"] == "market":
                    o["status"], o["reject_reason"] = "CANCELLED", "no liquidity"
                    return self._save_order(o)
            else:
                if o["type"] == "market":
                    if o["max_slip"] is not None and plan["slip"] > o["max_slip"]:
                        return self._reject(o, f"expected slippage {plan['slip']:.1f}bps exceeds limit")
                    if o["side"] == "buy" and plan["notional"] + plan["fee"] > self._acct()[1]:
                        return self._reject(o, "insufficient available cash")
                self._commit(o, plan, q, now, "taker")
        if o["type"] == "market":      # market orders are immediate-or-cancel
            if o["status"] in OPEN:
                o["status"] = "PARTIAL_CANCELLED" if o["filled_qty"] > 0 else "CANCELLED"
            self._save_order(o)
        else:
            if o["status"] == "NEW":
                o["status"] = "OPEN"
            self._save_order(o)

    def _plan(self, o, q, liquidity):
        rem = o["qty"] - o["filled_qty"]
        buy = o["side"] == "buy"
        top = q.ask_size if buy else q.bid_size
        fq = rem if top is None else min(rem, top)
        if fq <= 0 or (fq < rem and not self.cfg.allow_partial):
            return None
        if o["side"] == "sell":
            pos = self._pos(o["symbol"])
            free = pos["qty"] - pos["reserved_qty"] + o["reserved"]
            fq = min(fq, free)
            if fq <= 0:
                return None
        if liquidity == "taker":
            base = q.ask if buy else q.bid
            slip = self.cfg.slippage_bps + (self.cfg.impact_bps * fq / top if top else ZERO)
            price = q8(base * (1 + slip / 10000 if buy else 1 - slip / 10000))
            if o["type"] == "limit":
                price = min(price, o["limit_price"]) if buy else max(price, o["limit_price"])
            fee_bps = self.cfg.taker_fee_bps
        else:
            price, slip, fee_bps = o["limit_price"], ZERO, self.cfg.maker_fee_bps
        notional = q8(fq * price)
        return dict(qty=fq, price=price, notional=notional, fee=q8(notional * fee_bps / 10000), slip=slip, spread=q.spread_bps)

    def _commit(self, o, p, q, now, liquidity):
        fq, price, notional, fee = p["qty"], p["price"], p["notional"], p["fee"]
        pos = self._pos(o["symbol"])
        complete = o["filled_qty"] + fq >= o["qty"]
        if o["side"] == "buy":
            if o["reserved"] > 0:
                rel = o["reserved"] if complete else min(o["reserved"], q8(fq * o["rpu"]))
                self._adjust(rel, -rel, "release", o["symbol"], o["cid"], now)
                o["reserved"] -= rel
            cost = notional + fee
            self._adjust(-cost, ZERO, "buy_fill", o["symbol"], o["cid"], now)
            if pos["qty"] == 0:
                pos["opened_ts"], pos["strategy"], pos["peak_price"] = now, o["strategy"], price
            pos["qty"] += fq; pos["cost_basis"] += cost; pos["gross_cost"] += notional
            pos["peak_price"] = max(pos["peak_price"] or price, price)
            self._save_pos(pos)
        else:
            if o["reserved"] > 0:
                pos["reserved_qty"] -= fq; o["reserved"] -= fq
            full = fq == pos["qty"]
            basis = pos["cost_basis"] if full else q8(pos["cost_basis"] * fq / pos["qty"])
            gross = pos["gross_cost"] if full else q8(pos["gross_cost"] * fq / pos["qty"])
            net = notional - fee - basis
            self._adjust(notional - fee, ZERO, "sell_fill", o["symbol"], o["cid"], now)
            self.db.execute("INSERT INTO trades(symbol,strategy,entry_ts,exit_ts,qty,entry_price,exit_price,fees,net_pnl,hold_s,reason,order_id) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (o["symbol"], pos["strategy"], pos["opened_ts"], now, s(fq), s(gross / fq), s(price),
                             s((basis - gross) + fee), s(net), now - (pos["opened_ts"] or now), o["reason"], o["id"]))
            pos["qty"] -= fq; pos["cost_basis"] -= basis; pos["gross_cost"] -= gross; pos["realized"] += net
            if pos["qty"] == 0:
                pos["cost_basis"] = pos["gross_cost"] = ZERO; pos["peak_price"] = pos["opened_ts"] = None
            self._save_pos(pos)
        self.db.execute("INSERT INTO fills(order_id,ts,symbol,side,qty,price,fee,liquidity,slippage_bps,spread_bps) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (o["id"], now, o["symbol"], o["side"], s(fq), s(price), s(fee), liquidity, s(p["slip"]), s(p["spread"])))
        o["filled_qty"] += fq; o["filled_value"] += notional
        o["status"] = "FILLED" if complete else "PARTIAL"
        self._save_order(o)

    # ---------- lifecycle ----------
    def _cancel(self, o, now, status="CANCELLED", why=None):
        if o["side"] == "buy" and o["reserved"] > 0:
            self._adjust(o["reserved"], -o["reserved"], "release", o["symbol"], o["cid"], now)
        elif o["side"] == "sell" and o["reserved"] > 0:
            pos = self._pos(o["symbol"]); pos["reserved_qty"] -= o["reserved"]; self._save_pos(pos)
        o["reserved"] = ZERO
        o["status"] = "PARTIAL_CANCELLED" if (o["filled_qty"] > 0 and status == "CANCELLED") else status
        o["reject_reason"] = why
        self._save_order(o)

    def cancel_order(self, cid, now=None):
        now = time.time() if now is None else now
        with self._tx():
            o = self._order("cid=?", (cid,))
            if o and o["status"] in OPEN:
                self._cancel(o, now)
            return self._order("cid=?", (cid,))

    def on_quote(self, q: Quote, now=None):
        """Advance pending limit orders, expire old ones, track position peak (for trailing stops)."""
        now = time.time() if now is None else now
        events = []
        if self._quote_problem(q, q.symbol, now):
            return events
        with self._tx():
            pos = self._pos(q.symbol)
            if pos["qty"] > 0 and (pos["peak_price"] is None or q.bid > pos["peak_price"]):
                pos["peak_price"] = q.bid; self._save_pos(pos)
            for o in self.open_orders(q.symbol):
                if o["expires_ts"] is not None and now >= o["expires_ts"]:
                    self._cancel(o, now, "EXPIRED"); events.append((o["cid"], "expired")); continue
                if o["type"] != "limit":
                    continue
                hit = (o["side"] == "buy" and q.ask <= o["limit_price"]) or (o["side"] == "sell" and q.bid >= o["limit_price"])
                if hit:
                    p = self._plan(o, q, "maker")
                    if p:
                        self._commit(o, p, q, now, "maker"); events.append((o["cid"], "fill"))
        return events

    def update_peak(self, symbol, price):
        """Raise the position's high-water mark (used by trailing stops; backtests feed candle highs)."""
        with self._tx():
            pos = self._pos(symbol)
            if pos["qty"] > 0 and (pos["peak_price"] is None or d(price) > pos["peak_price"]):
                pos["peak_price"] = d(price); self._save_pos(pos)

    def close_position(self, symbol, quote, now=None, cid=None, reason="close", strategy="manual"):
        now = time.time() if now is None else now
        with self._tx():
            for o in self.open_orders(symbol):
                if o["side"] == "sell":
                    self._cancel(o, now)
            pos = self._pos(symbol)
            if pos["qty"] <= 0:
                return None
            cid = cid or f"close:{symbol}:{now}"
            existing = self._order("cid=?", (cid,))
            if existing:
                return existing
            self.db.execute("INSERT INTO orders(cid,ts,symbol,side,type,qty,status,strategy,reason) VALUES(?,?,?,?,?,?,?,?,?)",
                            (cid, now, symbol, "sell", "market", s(pos["qty"]), "NEW", strategy, reason))
            o = self._order("cid=?", (cid,))
            self._place(o, quote, now)
            return self._order("cid=?", (cid,))

    def reset(self, confirm: str, initial_balance=None, now=None):
        if confirm != "RESET":
            raise ValueError('reset requires confirm="RESET"')
        now = time.time() if now is None else now
        with self.lock:
            initial, _, _, epoch = self._acct()
            with self._tx():
                for t in ("fills", "trades", "snapshots", "decisions", "orders", "positions"):
                    self.db.execute(f"DELETE FROM {t}")
            self._new_account(d(initial_balance) if initial_balance is not None else initial, now, epoch + 1)

    # ---------- integrity ----------
    def check_invariants(self, marks=None):
        problems = []
        with self.lock:
            initial, cash, res, epoch = self._acct()
            if cash < 0: problems.append("negative cash")
            if res < 0: problems.append("negative reserved")
            if sum((o["reserved"] for o in self.open_orders() if o["side"] == "buy"), ZERO) != res:
                problems.append("reserved cash != sum of open buy-order reservations")
            realized = ZERO; basis = ZERO
            for r in self.db.execute("SELECT symbol FROM positions"):
                p = self._pos(r["symbol"])
                realized += p["realized"]; basis += p["cost_basis"]
                if p["qty"] < 0 or p["reserved_qty"] < 0 or p["reserved_qty"] > p["qty"]:
                    problems.append(f"bad quantities for {p['symbol']}")
                if sum((o["reserved"] for o in self.open_orders(p["symbol"]) if o["side"] == "sell"), ZERO) != p["reserved_qty"]:
                    problems.append(f"reserved qty mismatch for {p['symbol']}")
            a = self.db.execute("SELECT dcash,dreserved FROM audit WHERE epoch=?", (epoch,)).fetchall()
            if sum((d(x[0]) + d(x[1]) for x in a), ZERO) != cash + res:
                problems.append("audit trail does not reconcile with balances")
            if sum((d(t["net_pnl"]) for t in self.db.execute("SELECT net_pnl FROM trades")), ZERO) != realized:
                problems.append("trade P&L != position realized P&L")
            if cash + res + basis != initial + realized:
                problems.append("cash + reserved + cost basis != initial + realized")
        return problems
