#!/usr/bin/env python3
"""
Fetch BTC and ETH daily price data via Yahoo Finance.
Output: data/economic/crypto/
"""

import json, os, sys
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance")
    sys.exit(1)

OUT_DIR = Path(__file__).parent.parent / "data" / "economic" / "crypto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "btc_usd": {"ticker": "BTC-USD", "name": "Bitcoin", "exchange": "CoinMarketCap/Yahoo"},
    "eth_usd": {"ticker": "ETH-USD", "name": "Ethereum", "exchange": "CoinMarketCap/Yahoo"},
}

def main():
    print("Fetching crypto data from Yahoo Finance...")
    results = {}

    for name, info in TICKERS.items():
        ticker = info["ticker"]
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="max")
            if len(df) > 0:
                data_points = []
                for date, row in df.iterrows():
                    data_points.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"]) if row["Volume"] > 0 else 0,
                    })
                results[name] = {
                    "source": "Yahoo Finance",
                    "ticker": ticker,
                    "name": info["name"],
                    "exchange": info["exchange"],
                    "frequency": "daily",
                    "unit": "USD",
                    "count": len(data_points),
                    "start": data_points[0]["date"],
                    "end": data_points[-1]["date"],
                    "data": data_points,
                }
                print(f"  OK  {name:15s} ({ticker:10s}): {len(data_points):5d} points, {data_points[0]['date']} ~ {data_points[-1]['date']}")
            else:
                print(f"  FAIL {name}: no data")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    if results:
        out_file = OUT_DIR / "crypto.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_file} ({len(results)} series)")

if __name__ == "__main__":
    main()
