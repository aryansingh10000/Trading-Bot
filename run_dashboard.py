"""
Local crypto paper-trading dashboard.
Uses public Binance bid/ask quotes and the project's simulated ledger only.
No exchange keys and no real orders.
"""
import http.server
import json
import os
import socketserver
import threading
import time
import urllib.request
from decimal import Decimal

from papertrader.engine import PaperBot
from papertrader.ledger import ExecConfig, PaperLedger, Quote
from papertrader.risk import RiskConfig, RiskManager
from papertrader.strategies import Candle, MACDStrategy

STATE = {
    "running": True, "prices": {},
    "status": "Connecting to Binance public market data...",
    "ledger": None, "bot": None,
    "candles": {"BTCUSDT": [], "ETHUSDT": []},
}
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def fetch_binance_prices():
    url = "https://api.binance.com/api/v3/ticker/bookTicker"
    req = urllib.request.Request(url, headers={"User-Agent": "PaperTrader/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = {}
    for item in data:
        if item.get("symbol") in SYMBOLS:
            result[item["symbol"]] = {
                "bid": Decimal(item["bidPrice"]),
                "ask": Decimal(item["askPrice"]),
            }
    if len(result) != len(SYMBOLS):
        raise RuntimeError("Binance response did not include both BTCUSDT and ETHUSDT")
    return result


def market_loop():
    ledger, bot = STATE["ledger"], STATE["bot"]
    while True:
        try:
            prices = fetch_binance_prices()
            now = time.time()
            STATE["prices"] = {
                k: {"bid": float(v["bid"]), "ask": float(v["ask"])}
                for k, v in prices.items()
            }
            STATE["status"] = "Connected to Binance public API"
            for sym, p in prices.items():
                quote = Quote.of(sym, p["bid"], p["ask"], now)
                candles = STATE["candles"][sym]
                # This runner samples quotes every two seconds. These are sample bars,
                # not official exchange candles.
                candles.append(Candle(now, float(p["bid"]), float(p["ask"]),
                                      float(p["bid"]), float(p["bid"]), 0.0))
                if len(candles) > 400:
                    del candles[:-400]
                bot.step(sym, candles, quote, now, now)
                ledger.snapshot({k: Decimal(str(v["bid"])) for k, v in STATE["prices"].items()}, now)
        except Exception as exc:
            STATE["status"] = f"Market data warning: {type(exc).__name__}: {exc}"
        time.sleep(2)


def json_response(handler, obj, status=200):
    body = json.dumps(obj, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/api/state":
            json_response(self, {"error": "Not found"}, 404)
            return
        ledger = STATE["ledger"]
        prices = STATE["prices"]
        marks = {k: Decimal(str(v["bid"])) for k, v in prices.items() if "bid" in v}
        summary = ledger.summary(marks)
        fills = ledger.db.execute(
            "SELECT f.symbol, f.side, f.qty, f.price, f.fee, f.ts, o.status "
            "FROM fills f JOIN orders o ON o.id=f.order_id ORDER BY f.id DESC LIMIT 10"
        ).fetchall()
        positions = []
        for p in ledger.positions():
            if p["qty"] > 0:
                positions.append({
                    "symbol": p["symbol"], "qty": str(p["qty"]),
                    "cost_basis": str(p["gross_cost"] / p["qty"]) if p["qty"] else "0",
                    "current_price": str(marks.get(p["symbol"], Decimal(0))),
                })
        json_response(self, {
            "running": STATE["running"], "status": STATE["status"], "prices": prices,
            "cash": str(summary["cash"]), "equity": str(summary["equity"] if summary["equity"] is not None else summary["cash"]),
            "realized_pnl": str(summary["realized"]), "unrealized_pnl": str(summary["unrealized"]),
            "positions": positions,
            "trades": [{"symbol": r["symbol"], "side": r["side"], "qty": r["qty"],
                        "price": r["price"], "pnl": "0", "status": r["status"]} for r in fills],
        })

    def do_POST(self):
        if self.path != "/api/toggle":
            json_response(self, {"error": "Not found"}, 404)
            return
        STATE["running"] = not STATE["running"]
        if STATE["running"]:
            STATE["bot"].start()
        else:
            STATE["bot"].pause()
        json_response(self, {"running": STATE["running"]})

    def log_message(self, fmt, *args):
        # Keep the terminal readable while the dashboard polls every two seconds.
        if args and str(args[1] if len(args) > 1 else "") not in ("200",):
            super().log_message(fmt, *args)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Paper Trader</title>
<style>
:root{color-scheme:dark}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;color:#f8fafc;margin:0;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:22px}
h1{font-size:1.5rem;margin:0}.muted{color:#94a3b8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:18px}
.card{background:#1e293b;border-radius:12px;padding:18px;min-width:0;box-shadow:0 4px 14px #0002}
.label{color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.06em}.value{font-size:1.55rem;font-weight:700;margin-top:8px;overflow-wrap:anywhere}
.status{display:inline-block;padding:8px 11px;border-radius:6px;background:#334155;color:#cbd5e1;font-size:.85rem}
button{background:#3b82f6;color:white;border:0;padding:10px 16px;border-radius:7px;font-size:.95rem;cursor:pointer;margin-left:8px}button.stop{background:#ef4444}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin-top:12px;white-space:nowrap}th,td{text-align:left;padding:10px;border-bottom:1px solid #334155;font-size:.88rem}th{color:#94a3b8;font-weight:500}
.note{font-size:.9rem;color:#94a3b8;line-height:1.5}.good{color:#86efac}.bad{color:#fca5a5}
@media(max-width:600px){body{padding:14px}.value{font-size:1.3rem}}
</style></head><body>
<header><div><h1>Crypto Paper Trader</h1><div class="note">Simulated trading only. No real orders are sent.</div></div>
<div><span id="status" class="status">Connecting...</span><button id="toggleBtn" onclick="toggleBot()">Pause Bot</button></div></header>
<div class="grid">
<div class="card"><div class="label">BTC / USDT · bid</div><div id="btcPrice" class="value">--</div><div id="btcSpread" class="note">Waiting for quote</div></div>
<div class="card"><div class="label">ETH / USDT · bid</div><div id="ethPrice" class="value">--</div><div id="ethSpread" class="note">Waiting for quote</div></div>
<div class="card"><div class="label">Total equity</div><div id="equity" class="value">$10,000.00</div><div id="cash" class="note">Cash: --</div></div>
<div class="card"><div class="label">Realized P&amp;L</div><div id="pnl" class="value">$0.00</div><div id="unrealized" class="note">Unrealized: $0.00</div></div>
</div>
<div class="grid">
<div class="card"><h3>Open positions</h3><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Quantity</th><th>Avg entry</th><th>Current bid</th></tr></thead><tbody id="posTable"><tr><td colspan="4">No open positions</td></tr></tbody></table></div></div>
<div class="card"><h3>Recent fills</h3><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Side</th><th>Quantity</th><th>Price</th><th>Status</th></tr></thead><tbody id="tradeTable"><tr><td colspan="5">No fills yet</td></tr></tbody></table></div></div>
</div>
<div class="card"><h3>About this dashboard</h3><p class="note">Market prices come from Binance's public market-data endpoint. The strategy uses sampled quotes, so its signals are experimental and are not equivalent to trading on completed exchange candles. Paper results do not predict real trading performance.</p></div>
<script>
const money = n => Number(n).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const qty = n => Number(n).toFixed(6);
async function update(){
 try{
  const res=await fetch('/api/state',{cache:'no-store'}); if(!res.ok) throw new Error('Dashboard API unavailable');
  const d=await res.json();
  document.getElementById('status').innerText=d.status;
  for(const [sym,id] of [['BTCUSDT','btc'],['ETHUSDT','eth']]){
   const p=d.prices[sym];
   document.getElementById(id+'Price').innerText=p?'$'+money(p.bid):'--';
   document.getElementById(id+'Spread').innerText=p?'Ask $'+money(p.ask):'Waiting for quote';
  }
  document.getElementById('equity').innerText='$'+money(d.equity);
  document.getElementById('cash').innerText='Available cash: $'+money(d.cash);
  document.getElementById('pnl').innerText='$'+money(d.realized_pnl);
  document.getElementById('unrealized').innerText='Unrealized: $'+money(d.unrealized_pnl);
  document.getElementById('pnl').className='value '+(Number(d.realized_pnl)>=0?'good':'bad');
  const b=document.getElementById('toggleBtn');b.innerText=d.running?'Pause Bot':'Start Bot';b.className=d.running?'stop':'';
  document.getElementById('posTable').innerHTML=d.positions.length?d.positions.map(p=>`<tr><td>${p.symbol}</td><td>${qty(p.qty)}</td><td>$${money(p.cost_basis)}</td><td>$${money(p.current_price)}</td></tr>`).join(''):'<tr><td colspan="4">No open positions</td></tr>';
  document.getElementById('tradeTable').innerHTML=d.trades.length?d.trades.map(t=>`<tr><td>${t.symbol}</td><td>${t.side.toUpperCase()}</td><td>${qty(t.qty)}</td><td>$${money(t.price)}</td><td>${t.status}</td></tr>`).join(''):'<tr><td colspan="5">No fills yet</td></tr>';
 }catch(e){document.getElementById('status').innerText='Dashboard connection error';}
}
async function toggleBot(){await fetch('/api/toggle',{method:'POST'});update()}
setInterval(update,2000);update();
</script></body></html>"""


def main():
    os.makedirs("data", exist_ok=True)
    ledger = PaperLedger("data/papertrader.db", initial_balance="10000", config=ExecConfig())
    strategy = MACDStrategy(fast=8, slow=21, signal=9)
    risk = RiskManager(RiskConfig())
    bot = PaperBot(ledger=ledger, strategy=strategy, risk=risk, mode="auto", state="running")
    STATE["ledger"], STATE["bot"] = ledger, bot
    threading.Thread(target=market_loop, daemon=True).start()
    port = int(os.environ.get("PAPERTRADER_PORT", "8080"))
    class ReusableThreadingServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with ReusableThreadingServer(("127.0.0.1", port), DashboardHandler) as server:
        print("\\nPaperTrader dashboard is running.")
        print(f"Open http://127.0.0.1:{port} in your browser.")
        print("This is paper trading only. Press Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nShutting down paper trader.")


if __name__ == "__main__":
    main()
