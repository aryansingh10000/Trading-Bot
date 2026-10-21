# papertrader (stage 1: verified paper-trading core)

**Simulated money only.** There is no exchange-credential setting and no code path that submits real orders.
Simulated results are not earnings and do not predict real performance.

## Status (honest)
Built and tested (49 automated tests pass; run in the build sandbox with Python 3.12, stdlib only):
- Decimal-safe SQLite ledger: market/limit orders, maker/taker fees, spread, slippage, depth impact, partial fills,
  cash/position reservations, expiry, cancel, idempotent client order IDs, audit trail, invariant checker, confirmed reset.
- Strategies: MA crossover, RSI (with trend filter), MACD, trend-following, combined score. Indicators: SMA/EMA/RSI/MACD/Bollinger/ATR/ROC/volatility.
- Risk layer: position/exposure/cash caps, max positions, daily-loss and sticky drawdown halts, min score, spread, data freshness,
  loss cooldown, stop-loss / take-profit / trailing stop. Fails safe when marks are missing or stale.
- Bot engine (start/pause/resume/stop, auto/manual) with a recorded decision log, shared by live trading and backtests.
- Backtester (no look-ahead, next-bar-open execution, gap-aware stops, buy-and-hold benchmark, data validation) and a
  train/validation/test research pipeline that can only *recommend* forward-testing, never activate anything.

**NOT built or NOT verified yet** (the sandbox had no network, FastAPI, Docker or npm registry):
- Live market-data feed (WebSocket + REST, reconnect, provider status), and the 24/7 runner process
- FastAPI server, WebSocket push, authentication, rate limiting, CSV export endpoints
- React/TypeScript dashboard and charts
- AI analyst (LLM + rule-based fallback), notifications (dashboard/Telegram/email)
- Docker/Compose, deployment guide, backup scripts, full user guide
- Performance-over-time (24h/7d/30d) views, per-strategy/pair analytics endpoints

## Run it (any OS with Python 3.10+; no installs needed)
```
cd papertrader
python -m unittest discover -s tests -t .          # run the test suite (about 15 s)
python -m papertrader.cli backtest --strategy macd # backtest on SYNTHETIC data
python -m papertrader.cli research --strategy ma_crossover
python -m papertrader.cli sample-data --out sample_data
python -m papertrader.cli backtest --strategy trend --csv sample_data/SYN1-USD_1h.csv
```
CSV columns: `ts,open,high,low,close,volume` (UNIX seconds, ascending, fixed interval).

## Verifying the money is simulated
`grep -rniE "api[_-]?key|secret|withdraw|private" papertrader/` finds no exchange credential or order-submission code.
All orders go through `PaperLedger.submit_order`, which only edits the local SQLite file.


## Open the live dashboard

This package now includes `run_dashboard.py`, a local browser dashboard that uses Binance's
public BTC/USDT and ETH/USDT bid/ask endpoint. It does not require API keys and does not send
real orders. All fills and balances are simulated and saved locally in `data/papertrader.db`.

### Run in PyCharm
1. Extract the ZIP file to a folder.
2. In PyCharm, choose **Open** and select the extracted `papertrader` folder (the folder containing this README).
3. Open `run_dashboard.py` and click the green Run button.
4. Open `http://127.0.0.1:8080` in your browser.

Use Python 3.10 or newer. The dashboard uses only Python's standard library, so no pip installs
should be needed. If port 8080 is busy, set the `PAPERTRADER_PORT` environment variable to another
port before running.

On macOS, you can also double-click `start_dashboard.command`. On Windows, double-click
`start_dashboard.bat`.

The strategy receives quote samples every two seconds rather than official completed exchange
candles. Treat this as an experimental demonstration, not a validated trading strategy. Binance
network access is required for live prices; if it is unavailable, the dashboard will show a warning.
