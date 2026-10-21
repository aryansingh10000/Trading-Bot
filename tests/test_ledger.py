import os, tempfile, threading, unittest
from decimal import Decimal as D
from papertrader.ledger import PaperLedger, Quote, ExecConfig

NOW = 1_700_000_000.0
def Q(bid="100", ask="100.1", ts=NOW, bs=None, asz=None, sym="X"):
    return Quote.of(sym, bid, ask, ts, bs, asz)
ZERO_COST = ExecConfig(maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0, impact_bps=0)


class LedgerTests(unittest.TestCase):
    def ok(self, L, marks=None):
        self.assertEqual(L.check_invariants(marks), [])

    def test_init_defaults(self):
        L = PaperLedger()
        s = L.summary({})
        self.assertEqual((s["initial"], s["cash"], s["reserved"], s["equity"]), (D(10000),) * 2 + (D(0), D(10000)))
        self.ok(L)

    def test_buy_sell_accounting_exact(self):
        L = PaperLedger(initial_balance="10000", config=ExecConfig(maker_fee_bps=25, taker_fee_bps=40, slippage_bps=10, impact_bps=0))
        o = L.submit_order("b1", "X", "buy", "10", Q("100", "100"), now=NOW)
        # price = 100*1.001 = 100.1 ; notional 1001 ; fee 0.4% = 4.004
        self.assertEqual(o["status"], "FILLED")
        f = L.fills("b1")[0]
        self.assertEqual((D(f["price"]), D(f["fee"])), (D("100.1"), D("4.004")))
        s = L.summary({"X": D("100")})
        self.assertEqual(s["cash"], D("10000") - D("1001") - D("4.004"))
        self.assertEqual(s["unrealized"], D("1000") - D("1005.004"))           # fees are in cost basis
        o = L.submit_order("s1", "X", "sell", "10", Q("110", "110"), now=NOW)  # 110*0.999=109.89 ; notional 1098.9 ; fee 4.3956
        t = L.trades()[0]
        self.assertEqual(D(t["net_pnl"]), D("1098.9") - D("4.3956") - D("1005.004"))
        self.assertEqual(D(t["fees"]), D("4.004") + D("4.3956"))
        s = L.summary({})
        self.assertEqual(s["equity"], D("10000") + D(t["net_pnl"]))
        self.assertEqual(s["realized"], D(t["net_pnl"]))
        self.ok(L)

    def test_spread_and_slippage_make_round_trip_lose(self):
        L = PaperLedger()
        L.submit_order("b", "X", "buy", "1", Q("100", "101"), now=NOW)
        L.submit_order("s", "X", "sell", "1", Q("100", "101"), now=NOW)
        self.assertLess(D(L.trades()[0]["net_pnl"]), 0)

    def test_realized_and_unrealized_separate_partial_close(self):
        L = PaperLedger(config=ZERO_COST)
        L.submit_order("b", "X", "buy", "10", Q("100", "100"), now=NOW)
        L.submit_order("s", "X", "sell", "4", Q("110", "110"), now=NOW)
        s = L.summary({"X": D("120")})
        self.assertEqual(s["realized"], D("40"))
        self.assertEqual(s["unrealized"], D("120"))        # 6 * (120-100)
        self.assertEqual(s["equity"], D("10000") + 40 + 120)
        self.ok(L, {"X": D(120)})

    def test_partial_fill_market_is_ioc(self):
        L = PaperLedger(config=ZERO_COST)
        o = L.submit_order("b", "X", "buy", "10", Q("100", "100", asz="3"), now=NOW)
        self.assertEqual((o["status"], o["filled_qty"]), ("PARTIAL_CANCELLED", D(3)))
        self.assertEqual(L.position("X")["qty"], D(3))
        self.ok(L)

    def test_no_partial_when_disabled(self):
        L = PaperLedger(config=ExecConfig(allow_partial=False))
        o = L.submit_order("b", "X", "buy", "10", Q(asz="3"), now=NOW)
        self.assertEqual(o["status"], "CANCELLED")
        self.assertEqual(L.position("X")["qty"], 0)

    def test_limit_order_reserves_cash_then_fills_as_maker(self):
        L = PaperLedger(config=ExecConfig(maker_fee_bps=10, taker_fee_bps=40, slippage_bps=5, impact_bps=0))
        o = L.submit_order("l", "X", "buy", "10", Q("100", "101"), "limit", limit_price="95", now=NOW)
        self.assertEqual(o["status"], "OPEN")
        s = L.summary({})
        self.assertEqual(s["reserved"], D("950") * D("1.004"))
        self.assertEqual(s["cash"] + s["reserved"], D(10000))     # no double counting
        self.assertEqual(L.on_quote(Q("96", "97"), NOW + 1), [])  # not hit
        self.assertEqual(L.on_quote(Q("94", "94.5"), NOW + 2), [("l", "fill")])
        o = L.get_order("l")
        self.assertEqual(o["status"], "FILLED")
        self.assertEqual(L.fills("l")[0]["liquidity"], "maker")
        s = L.summary({"X": D("94")})
        self.assertEqual(s["reserved"], 0)
        self.assertEqual(s["cash"], D(10000) - D("950") - D("0.95"))
        self.ok(L, {"X": D(94)})

    def test_limit_partial_fills_over_time_keep_reservation_consistent(self):
        L = PaperLedger(config=ExecConfig(maker_fee_bps=10, taker_fee_bps=40, slippage_bps=0, impact_bps=0))
        L.submit_order("l", "X", "buy", "10", Q("100", "101"), "limit", limit_price="95", now=NOW)
        L.on_quote(Q("90", "94", asz="4"), NOW + 1)
        self.assertEqual(L.get_order("l")["status"], "PARTIAL"); self.ok(L)
        L.on_quote(Q("90", "94", asz="6"), NOW + 2)
        self.assertEqual(L.get_order("l")["status"], "FILLED")
        self.assertEqual(L.summary({})["reserved"], 0); self.ok(L)

    def test_marketable_limit_fills_immediately_capped_at_limit(self):
        L = PaperLedger()
        o = L.submit_order("l", "X", "buy", "1", Q("100", "100"), "limit", limit_price="100.02", now=NOW)
        self.assertEqual(o["status"], "FILLED")
        self.assertLessEqual(D(L.fills("l")[0]["price"]), D("100.02")); self.ok(L)

    def test_cancel_and_expire_release_reservations(self):
        L = PaperLedger()
        L.submit_order("a", "X", "buy", "5", Q(), "limit", limit_price="90", now=NOW)
        L.submit_order("b", "X", "buy", "5", Q(), "limit", limit_price="90", now=NOW, expires_in=60)
        L.cancel_order("a", NOW + 1)
        L.on_quote(Q(ts=NOW + 61), NOW + 61)
        self.assertEqual(L.get_order("a")["status"], "CANCELLED")
        self.assertEqual(L.get_order("b")["status"], "EXPIRED")
        s = L.summary({}); self.assertEqual((s["reserved"], s["cash"]), (0, D(10000))); self.ok(L)

    def test_sell_limit_reserves_position_and_blocks_oversell(self):
        L = PaperLedger(config=ZERO_COST)
        L.submit_order("b", "X", "buy", "10", Q("100", "100"), now=NOW)
        L.submit_order("sl", "X", "sell", "8", Q("100", "100"), "limit", limit_price="120", now=NOW)
        o = L.submit_order("s2", "X", "sell", "5", Q("100", "100"), now=NOW)
        self.assertEqual(o["status"], "REJECTED")
        self.assertEqual(L.position("X")["reserved_qty"], D(8)); self.ok(L)
        L.on_quote(Q("121", "122"), NOW + 1)
        self.assertEqual(L.position("X")["qty"], D(2)); self.ok(L)

    def test_cannot_buy_more_than_cash_or_sell_what_we_dont_have(self):
        L = PaperLedger(initial_balance="1000")
        self.assertEqual(L.submit_order("b", "X", "buy", "100", Q("100", "100"), now=NOW)["status"], "REJECTED")
        self.assertEqual(L.submit_order("l", "X", "buy", "100", Q("100", "100"), "limit", limit_price="50", now=NOW)["status"], "REJECTED")
        self.assertEqual(L.submit_order("s", "X", "sell", "1", Q(), now=NOW)["status"], "REJECTED")
        self.assertEqual(L.summary({})["cash"], D(1000)); self.ok(L)

    def test_stale_invalid_quotes_rejected(self):
        L = PaperLedger()
        self.assertIn("stale", L.submit_order("a", "X", "buy", "1", Q(ts=NOW - 500), now=NOW)["reject_reason"])
        self.assertIn("crossed", L.submit_order("b", "X", "buy", "1", Q("101", "100"), now=NOW)["reject_reason"])
        self.assertIn("invalid", L.submit_order("c", "X", "buy", "1", Q("0", "100"), now=NOW)["reject_reason"])
        self.assertIn("no quote", L.submit_order("d", "X", "buy", "1", None, now=NOW)["reject_reason"])
        self.assertEqual(L.positions(), []); self.ok(L)

    def test_max_slippage_guard(self):
        L = PaperLedger()
        o = L.submit_order("a", "X", "buy", "10", Q(asz="10"), now=NOW, max_slippage_bps="12")  # 5 + 10 impact = 15
        self.assertEqual(o["status"], "REJECTED")

    def test_invalid_inputs_raise(self):
        L = PaperLedger()
        for kw in (dict(qty="0"), dict(qty="-1"), dict(side="short"), dict(type="limit")):
            a = dict(cid="z", symbol="X", side="buy", qty="1", quote=Q()); a.update(kw)
            if "type" in kw: a["type"] = "limit"
            with self.assertRaises(ValueError):
                L.submit_order(a.pop("cid"), a.pop("symbol"), a.pop("side"), a.pop("qty"), a.pop("quote"), now=NOW, **a)

    def test_duplicate_client_order_id_is_idempotent(self):
        L = PaperLedger()
        a = L.submit_order("dup", "X", "buy", "1", Q(), now=NOW)
        b = L.submit_order("dup", "X", "buy", "1", Q(), now=NOW + 5)
        self.assertEqual(a["id"], b["id"]); self.assertEqual(L.position("X")["qty"], D(1)); self.assertEqual(len(L.fills("dup")), 1)

    def test_concurrent_same_cid_executes_once(self):
        L = PaperLedger(); res = []
        def go(): res.append(L.submit_order("same", "X", "buy", "1", Q(), now=NOW)["id"])
        ts = [threading.Thread(target=go) for _ in range(16)]; [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(set(res)), 1); self.assertEqual(L.position("X")["qty"], D(1)); self.ok(L)

    def test_concurrent_buys_never_overdraw(self):
        L = PaperLedger(initial_balance="1000", config=ZERO_COST)
        def go(i): L.submit_order(f"c{i}", "X", "buy", "3", Q("100", "100"), now=NOW)
        ts = [threading.Thread(target=go, args=(i,)) for i in range(20)]; [t.start() for t in ts]; [t.join() for t in ts]
        s = L.summary({"X": D(100)})
        self.assertGreaterEqual(s["cash"], 0)
        self.assertEqual(L.position("X")["qty"], D(9))      # 3 fills of qty 3 = $900; the 4th ($300) exceeds the remaining $100
        self.ok(L, {"X": D(100)})

    def test_invariants_hold_through_random_sequence(self):
        import random
        r = random.Random(7); L = PaperLedger(config=ExecConfig(maker_fee_bps=10, taker_fee_bps=30, slippage_bps=5, impact_bps=5))
        px = 100.0; t = NOW
        for i in range(400):
            px *= 1 + r.uniform(-0.01, 0.01); t += 10
            q = Q(f"{px:.4f}", f"{px*1.0005:.4f}", t, bs=str(r.uniform(1, 10)), asz=str(r.uniform(1, 10)))
            L.on_quote(q, t)
            a = r.random()
            if a < .3: L.submit_order(f"o{i}", "X", "buy", f"{r.uniform(.1, 6):.4f}", q, now=t)
            elif a < .5: L.submit_order(f"o{i}", "X", "sell", f"{r.uniform(.1, 6):.4f}", q, now=t)
            elif a < .7: L.submit_order(f"o{i}", "X", r.choice(["buy", "sell"]), f"{r.uniform(.1, 4):.4f}", q, "limit", limit_price=f"{px*r.uniform(.98, 1.02):.4f}", now=t, expires_in=100)
            elif a < .75 and L.open_orders(): L.cancel_order(L.open_orders()[0]["cid"], t)
            elif a < .8: L.close_position("X", q, t, f"cl{i}")
            self.assertEqual(L.check_invariants({"X": q.bid}), [], msg=f"step {i}")

    def test_persistence_and_restart_recovery(self):
        p = os.path.join(tempfile.mkdtemp(), "t.db")
        L = PaperLedger(p, "5000", ZERO_COST)
        L.submit_order("keep", "X", "buy", "5", Q("100", "100"), now=NOW)
        L.submit_order("lim", "X", "buy", "5", Q("100", "100"), "limit", limit_price="80", now=NOW)
        L.db.close()
        L2 = PaperLedger(p, "999999", ZERO_COST)                # initial arg ignored for existing account
        s = L2.summary({"X": D(100)})
        self.assertEqual((s["initial"], L2.position("X")["qty"], len(L2.open_orders())), (D(5000), D(5), 1))
        again = L2.submit_order("keep", "X", "buy", "5", Q("100", "100"), now=NOW + 99)   # replay after restart
        self.assertEqual(again["status"], "FILLED"); self.assertEqual(L2.position("X")["qty"], D(5))
        self.assertEqual(L2.check_invariants({"X": D(100)}), [])

    def test_missing_mark_means_no_equity_not_invented_price(self):
        L = PaperLedger(); L.submit_order("b", "X", "buy", "1", Q(), now=NOW)
        s = L.summary({}); self.assertIsNone(s["equity"]); self.assertEqual(s["missing_marks"], ["X"])
        self.assertIsNone(L.snapshot({}))

    def test_reset_requires_confirmation_and_clears(self):
        L = PaperLedger(); L.submit_order("b", "X", "buy", "1", Q(), now=NOW)
        with self.assertRaises(ValueError): L.reset("yes")
        self.assertEqual(L.position("X")["qty"], 1)
        L.reset("RESET", "2500")
        s = L.summary({}); self.assertEqual((s["initial"], s["cash"], L.positions()), (D(2500), D(2500), [])); self.ok(L)


if __name__ == "__main__":
    unittest.main()
