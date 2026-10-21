import math, unittest
from decimal import Decimal as D
from papertrader.backtest import run_backtest, validate_candles, chronological_split, DataError
from papertrader.engine import PaperBot
from papertrader.ledger import PaperLedger, Quote, ExecConfig
from papertrader.research import evaluate
from papertrader.risk import RiskConfig, RiskManager
from papertrader.sample_data import generate_candles
from papertrader.strategies import Candle, make_strategy, REGISTRY

NOW = 1_700_000_000.0


def series(prices, step=3600):
    return [Candle(NOW + i * step, p, p * 1.001, p * .999, p, 1.0) for i, p in enumerate(prices)]


class StrategyTests(unittest.TestCase):
    def test_ma_cross_buy_and_sell(self):
        up = series([100] * 30 + [100 + i for i in range(1, 15)])
        sigs = [make_strategy("ma_crossover", {"short": 3, "long": 8}).signal(up[:i]) for i in range(10, len(up) + 1)]
        self.assertIn("buy", [s.action for s in sigs])
        down = series([100 + i for i in range(30)] + [130 - 2 * i for i in range(1, 20)])
        sigs = [make_strategy("ma_crossover", {"short": 3, "long": 8}).signal(down[:i]) for i in range(10, len(down) + 1)]
        self.assertIn("sell", [s.action for s in sigs])

    def test_rsi_oversold_recovery_and_downtrend_filter(self):
        base = [100 - i * 0.6 for i in range(40)] + [76 + i * 1.5 for i in range(1, 6)]
        acts = lambda st: [st.signal(series(base)[:i]).action for i in range(20, len(base) + 1)]
        self.assertIn("buy", acts(make_strategy("rsi", {"trend_filter": False})))
        strong_down = [200 - i * 1.5 for i in range(110)] + [35 + i * 0.6 for i in range(1, 8)]
        a = [make_strategy("rsi", {"max_below_trend_pct": 5}).signal(series(strong_down)[:i]) for i in range(60, len(strong_down) + 1)]
        self.assertNotIn("buy", [x.action for x in a])

    def test_all_strategies_return_valid_signals_and_warmup_holds(self):
        c = generate_candles(600, 3)
        for name in REGISTRY:
            st = make_strategy(name)
            self.assertEqual(st.signal(c[:5]).action, "hold")
            for i in range(120, 600, 37):
                s = st.signal(c[:i]); self.assertIn(s.action, ("buy", "sell", "hold")); self.assertTrue(0 <= s.score <= 1)

    def test_no_lookahead_in_strategy_signals(self):
        c = generate_candles(400, 4)
        tampered = c[:300] + [Candle(x.ts, x.o * 3, x.h * 3, x.l * 3, x.c * 3, x.v) for x in c[300:]]
        for name in REGISTRY:
            st = make_strategy(name)
            for i in range(120, 300, 11):
                self.assertEqual(st.signal(c[:i]), st.signal(tampered[:i]), name)

    def test_param_validation(self):
        with self.assertRaises(ValueError): make_strategy("ma_crossover", {"short": 30, "long": 10})
        with self.assertRaises(ValueError): make_strategy("rsi", {"bogus": 1})
        with self.assertRaises(ValueError): make_strategy("nope")


class RiskTests(unittest.TestCase):
    def setup(self, **kw):
        L = PaperLedger(config=ExecConfig(maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0, impact_bps=0))
        return L, RiskManager(RiskConfig(**kw))

    def q(self, sym="X", bid="100", ask="100.1", ts=NOW): return Quote.of(sym, bid, ask, ts)

    def verdict(self, L, R, q, score=0.9, now=NOW):
        return R.evaluate_entry(L, q, score, L.summary({s["symbol"]: q.bid for s in L.positions()} | {q.symbol: q.bid}), now)

    def test_sizing_obeys_position_and_cash_caps(self):
        L, R = self.setup(max_position_pct=10, max_cash_per_trade=500)
        v = self.verdict(L, R, self.q()); self.assertTrue(v.allowed); self.assertLessEqual(v.notional, 500)
        L, R = self.setup(max_position_pct=2, max_cash_per_trade=5000)
        self.assertEqual(self.verdict(L, R, self.q()).notional, D(200))

    def test_blocks(self):
        L, R = self.setup()
        self.assertFalse(self.verdict(L, R, self.q(), score=0.3).allowed)
        self.assertIn("spread", " ".join(self.verdict(L, R, self.q(bid="100", ask="101")).reasons))
        self.assertIn("stale", " ".join(self.verdict(L, R, self.q(ts=NOW - 100)).reasons))
        v = R.evaluate_entry(L, self.q(), .9, L.summary({}) | {"complete": False}, NOW)
        self.assertFalse(v.allowed)

    def test_max_positions_and_no_pyramiding(self):
        L, R = self.setup(max_open_positions=1)
        L.submit_order("a", "A", "buy", "1", self.q("A"), now=NOW)
        self.assertIn("already holding", " ".join(self.verdict(L, R, self.q("A")).reasons))
        self.assertIn("max open", " ".join(self.verdict(L, R, self.q("B")).reasons))

    def test_exposure_cap(self):
        L, R = self.setup(max_exposure_pct=10, max_position_pct=50, max_cash_per_trade=1e9)
        L.submit_order("a", "A", "buy", "9", self.q("A", "100", "100"), now=NOW)       # $900 = 9% of equity
        v = self.verdict(L, R, self.q("B"))
        self.assertTrue(v.allowed); self.assertEqual(v.notional, D(100))               # capped to the $100 of room left
        L.submit_order("c", "A", "buy", "0.95", self.q("A", "100", "100"), now=NOW)    # now ~$995 invested: $5 room < $10 minimum
        self.assertFalse(self.verdict(L, R, self.q("B")).allowed)

    def test_drawdown_halt_is_sticky_and_daily_loss(self):
        L, R = self.setup(max_drawdown_pct=5)
        R.update(D(10000), NOW); R.update(D(9400), NOW + 1)
        self.assertIsNotNone(R.halt)
        R.update(D(10500), NOW + 2); self.assertIsNotNone(R.halt)                     # recovery does not clear it
        self.assertIn("drawdown", " ".join(self.verdict(L, R, self.q()).reasons))
        R.reset_halt(10500); self.assertIsNone(R.halt)
        L, R = self.setup(max_daily_loss_pct=2, max_drawdown_pct=99)
        R.update(D(10000), NOW)
        L.submit_order("a", "X", "buy", "50", self.q(bid="100", ask="100"), now=NOW)
        v = R.evaluate_entry(L, self.q(bid="95", ask="95.05"), .9, L.summary({"X": D(95)}), NOW + 5)
        self.assertIn("daily loss", " ".join(v.reasons))

    def test_cooldown_after_consecutive_losses(self):
        L, R = self.setup(cooldown_after_losses=2, cooldown_s=1000)
        for i in range(2):
            L.submit_order(f"b{i}", "X", "buy", "1", self.q(bid="100", ask="100"), now=NOW + i)
            L.submit_order(f"s{i}", "X", "sell", "1", self.q(bid="90", ask="90.1", ts=NOW + i), now=NOW + i)
        self.assertIn("cooldown", " ".join(self.verdict(L, R, self.q(ts=NOW + 10), now=NOW + 10).reasons))
        self.assertTrue(self.verdict(L, R, self.q(ts=NOW + 2000), now=NOW + 2000).allowed)

    def test_exit_rules(self):
        L, R = self.setup(stop_loss_pct=3, take_profit_pct=5, trailing_stop_pct=2)
        L.submit_order("a", "X", "buy", "1", self.q(bid="100", ask="100"), now=NOW)
        pos = L.position("X")
        self.assertEqual(R.exit_reason(pos, self.q(bid="96.9", ask="97")), "stop_loss")
        self.assertEqual(R.exit_reason(pos, self.q(bid="105.1", ask="105.2")), "take_profit")
        L.on_quote(self.q(bid="104", ask="104.1"), NOW); pos = L.position("X")
        self.assertEqual(R.exit_reason(pos, self.q(bid="101.9", ask="102")), "trailing_stop")


class BotTests(unittest.TestCase):
    def mk(self, state="running", mode="auto"):
        L = PaperLedger(); st = make_strategy("ma_crossover", {"short": 3, "long": 8})
        return L, PaperBot(L, st, RiskManager(), mode, state)

    def drive(self, bot, candles, upto):
        q = Quote.of("X", candles[upto].o, candles[upto].o * 1.0005, candles[upto].ts)
        return bot.step("X", candles[:upto], q, candles[upto].ts, candles[upto].ts)

    def rising(self): return series([100] * 30 + [100 + i * .5 for i in range(1, 25)])

    def test_paused_and_stopped_and_manual_block_entries_and_log_why(self):
        c = self.rising()
        for state, mode in (("paused", "auto"), ("stopped", "auto"), ("running", "manual")):
            L, bot = self.mk(state, mode)
            for i in range(10, len(c)): self.drive(bot, c, i)
            self.assertEqual(L.positions(), [], (state, mode))
        L, bot = self.mk("paused")
        for i in range(10, len(c)): self.drive(bot, c, i)
        self.assertTrue(any(d["action"] == "buy_blocked" for d in L.decisions()))

    def test_running_bot_enters_and_logs_explainable_decision(self):
        L, bot = self.mk(); c = self.rising()
        for i in range(10, len(c)): self.drive(bot, c, i)
        self.assertTrue(L.positions() or L.trades())
        buy = [d for d in L.decisions() if d["action"] == "buy" and d["executed"]]
        self.assertTrue(buy); self.assertIn("crossed above", buy[0]["detail"])     # the log explains WHY it entered

    def test_replaying_same_bar_after_restart_does_not_duplicate(self):
        L, bot = self.mk(); c = self.rising()
        for i in range(10, len(c)): self.drive(bot, c, i)
        n_orders = len(L.db.execute("SELECT * FROM orders").fetchall()); n_fills = len(L.db.execute("SELECT * FROM fills").fetchall())
        bot2 = PaperBot(L, make_strategy("ma_crossover", {"short": 3, "long": 8}), RiskManager(), "auto", "running")
        for i in range(10, len(c)): self.drive(bot2, c, i)
        self.assertEqual(n_orders, len(L.db.execute("SELECT * FROM orders").fetchall()))
        self.assertEqual(n_fills, len(L.db.execute("SELECT * FROM fills").fetchall()))
        marks = {p["symbol"]: D(100) for p in L.positions()}
        self.assertEqual(L.check_invariants(marks), [])

    def test_stale_feed_blocks_entry(self):
        L, bot = self.mk(); c = self.rising()
        for i in range(10, 33): self.drive(bot, c, i)
        stale_q = Quote.of("X", 130, 130.1, c[40].ts - 500)
        bot.state = "running"
        out = bot.step("X", c[:40], stale_q, c[40].ts, c[40].ts)
        self.assertFalse(out["executed"])


class BacktestTests(unittest.TestCase):
    def data(self, n=900): return {"S1": generate_candles(n, 11), "S2": generate_candles(n, 12, 40)}

    def test_runs_all_strategies_with_clean_accounting(self):
        for name in REGISTRY:
            r = run_backtest(self.data(), name)
            self.assertEqual(r["invariant_problems"], [], name)
            self.assertEqual(len(r["equity_curve"]), r["bars"][1] - r["bars"][0])
            self.assertIn("limitations", r)
            for t in r["trades"]: self.assertGreater(t["exit_ts"], t["entry_ts"] - 1)

    def test_deterministic(self):
        a, b = run_backtest(self.data(), "trend"), run_backtest(self.data(), "trend")
        self.assertEqual(a["equity_curve"], b["equity_curve"])

    def test_no_lookahead_future_candles_cannot_change_past_trades(self):
        d1 = self.data(900); d2 = {k: v[:700] + [Candle(c.ts, c.o * 5, c.h * 5, c.l * 5, c.c * 5, c.v) for c in v[700:]] for k, v in d1.items()}
        for name in ("combined", "ma_crossover"):
            a = run_backtest(d1, name, start=100, end=700, liquidate_at_end=False)
            b = run_backtest(d2, name, start=100, end=700, liquidate_at_end=False)
            self.assertEqual(a["equity_curve"], b["equity_curve"]); self.assertEqual(a["trades"], b["trades"])

    def test_costs_matter(self):
        free = ExecConfig(maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0, impact_bps=0)
        costly = ExecConfig(taker_fee_bps=100, slippage_bps=20)   # stays under the 25bps risk guard
        a = run_backtest(self.data(), "macd", execution=free, spread_bps=0)["metrics"]
        b = run_backtest(self.data(), "macd", execution=costly, spread_bps=20)["metrics"]
        self.assertGreater(a["total_return_pct"], b["total_return_pct"]); self.assertGreater(b["fees_paid"], 0)

    def test_stop_loss_exits_gap_aware(self):
        px = [100] * 40 + [100 + i for i in range(1, 8)] + [60] * 5     # crash after uptrend
        c = [Candle(NOW + i * 3600, p, p * 1.001, p * .999, p, 1) for i, p in enumerate(px)]
        r = run_backtest({"X": c}, "ma_crossover", {"short": 3, "long": 8}, RiskConfig(stop_loss_pct=5, take_profit_pct=None), allow_gaps=False)
        self.assertTrue(any(t["reason"] == "stop_loss" for t in r["trades"]))
        self.assertEqual(r["invariant_problems"], [])

    def test_bad_data_detected(self):
        c = generate_candles(300, 5)
        with self.assertRaises(DataError): validate_candles(c[:50] + c[40:60], 3600)                         # non-monotonic
        with self.assertRaises(DataError): validate_candles([c[0]._replace(h=c[0].l * 0.5)], 3600)           # inconsistent OHLC
        with self.assertRaises(DataError): validate_candles([c[0]._replace(c=float("nan"))], 3600)
        with self.assertRaises(DataError): validate_candles([c[0]._replace(o=-1)], 3600)
        gap = c[:100] + c[105:]
        self.assertEqual(len(validate_candles(gap, 3600)), 1)
        with self.assertRaises(DataError): run_backtest({"X": gap}, "macd")
        self.assertTrue(run_backtest({"X": gap}, "macd", allow_gaps=True)["data_gaps"])

    def test_misaligned_symbols_use_common_timestamps(self):
        a = generate_candles(500, 1); b = generate_candles(500, 2)[20:]
        r = run_backtest({"A": a, "B": b}, "macd", allow_gaps=True)
        self.assertEqual(r["invariant_problems"], [])

    def test_split_is_chronological_and_nonoverlapping(self):
        sp = chronological_split(1000, 100)
        self.assertEqual(sp["train"][1], sp["validation"][0]); self.assertEqual(sp["validation"][1], sp["test"][0]); self.assertEqual(sp["test"][1], 1000)
        with self.assertRaises(DataError): chronological_split(110, 100)

    def test_benchmark_present(self):
        r = run_backtest(self.data(), "trend"); self.assertEqual(len(r["benchmark"]["curve"]), len(r["equity_curve"]))


class ResearchTests(unittest.TestCase):
    def test_pipeline_selects_on_train_and_never_auto_promotes(self):
        data = {"S1": generate_candles(1800, 21)}
        rep = evaluate(data, "ma_crossover", {"short": 9, "long": 21}, {"short": [5, 9, 12], "long": [21, 34, 55]}, min_trades=5)
        self.assertGreater(rep["candidates_tried"], 0); self.assertIn("multiple_testing_note", rep)
        for f in rep["finalists"]:
            self.assertEqual(f["passed"], not f["failures"])
        if rep["recommendation"]:
            self.assertIn("manual approval", rep["recommendation"]["next_step"])

    def test_losing_candidates_are_rejected(self):
        data = {"S1": generate_candles(1800, 22)}
        rep = evaluate(data, "macd", {}, {"fast": [5, 8], "slow": [20, 30]}, min_trades=5,
                       execution=ExecConfig(taker_fee_bps=300, slippage_bps=50))
        self.assertTrue(all(not f["passed"] for f in rep["finalists"]))
        self.assertIsNone(rep["recommendation"])


if __name__ == "__main__":
    unittest.main()
