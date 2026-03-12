#!/usr/bin/env python3
"""Fetch US economic data from FRED and output macro.json + financial.json."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd

OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/us"
START_DATE = "1971-01-01"
END_DATE = "2026-03-12"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, monthly_agg: bool = False) -> pd.DataFrame | None:
    """Fetch a FRED series via curl. Use fq=Monthly for daily/weekly series."""
    params = f"id={series_id}&cosd={START_DATE}&coed={END_DATE}"
    if monthly_agg:
        params += "&fq=Monthly&fas=avg"  # Monthly frequency, average aggregation
    url = f"{FRED_CSV_URL}?{params}"
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-s", "-f", "--max-time", "30", url],
                capture_output=True, text=True, timeout=45
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl exit {result.returncode}")
            csv_text = result.stdout
            if not csv_text.strip() or csv_text.strip().startswith("<!"):
                raise RuntimeError("empty or HTML response")
            df = pd.read_csv(StringIO(csv_text), na_values=".")
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            return df
        except Exception as e:
            print(f"  RETRY {attempt+1}/3 {series_id}: {e}", file=sys.stderr, flush=True)
            time.sleep(2 * (attempt + 1))
    print(f"  ERROR {series_id}: all retries failed", file=sys.stderr, flush=True)
    return None


def to_monthly_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily/weekly data to monthly averages."""
    df = df.set_index("date")
    monthly = df.resample("MS").mean().dropna()
    monthly = monthly.reset_index()
    monthly.columns = ["date", "value"]
    return monthly


def compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year percent change from an index series."""
    df = df.set_index("date").sort_index()
    # Ensure monthly frequency
    df = df.resample("MS").last().dropna()
    yoy = df["value"].pct_change(periods=12) * 100
    yoy = yoy.dropna().reset_index()
    yoy.columns = ["date", "value"]
    return yoy


def compute_mom_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute month-over-month change (absolute difference)."""
    df = df.set_index("date").sort_index()
    df = df.resample("MS").last().dropna()
    mom = df["value"].diff()
    mom = mom.dropna().reset_index()
    mom.columns = ["date", "value"]
    return mom


def quarterly_to_series(df: pd.DataFrame) -> list[dict]:
    """Convert quarterly data to series entries with YYYY-MM format aligned to quarter start."""
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        # Align to quarter start month
        q_month = ((d.month - 1) // 3) * 3 + 1
        date_str = f"{d.year}-{q_month:02d}"
        entries.append({
            "date": date_str,
            "value": round(float(row["value"]), 4),
            "confidence": "official"
        })
    return entries


def monthly_to_series(df: pd.DataFrame) -> list[dict]:
    """Convert monthly data to series entries with YYYY-MM format."""
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        date_str = f"{d.year}-{d.month:02d}"
        entries.append({
            "date": date_str,
            "value": round(float(row["value"]), 4),
            "confidence": "official"
        })
    return entries


def print_status(name: str, series_id: str, series: list[dict] | None):
    if series and len(series) > 0:
        print(f"  OK  {name:30s} ({series_id:15s}): {len(series):5d} points, {series[0]['date']} ~ {series[-1]['date']}", flush=True)
    else:
        print(f"  FAIL {name:30s} ({series_id:15s}): no data", flush=True)


def build_indicator(description: str, unit: str, frequency: str, fred_id: str, series: list[dict]) -> dict:
    return {
        "description": description,
        "unit": unit,
        "frequency": frequency,
        "fred_series_id": fred_id,
        "series": series,
    }


# ── MACRO indicators ─────────────────────────────────────────────────────────

MACRO_SPECS = [
    # (key, fred_id, description, unit, frequency, transform)
    # transform: "raw", "yoy", "mom", "raw_quarterly"
    ("gdp_growth",      "A191RL1Q225SBEA", "Real GDP Growth Rate (quarterly, annualized)",     "percent",   "quarterly", "raw_quarterly"),
    ("cpi_yoy",         "CPIAUCSL",        "CPI-U Year-over-Year Change",                       "percent",   "monthly",   "yoy"),
    ("core_cpi_yoy",    "CPILFESL",        "Core CPI (Less Food & Energy) Year-over-Year",      "percent",   "monthly",   "yoy"),
    ("fed_funds_rate",  "FEDFUNDS",        "Effective Federal Funds Rate",                       "percent",   "monthly",   "raw"),
    ("unemployment",    "UNRATE",          "Unemployment Rate (U-3)",                            "percent",   "monthly",   "raw"),
    ("nonfarm_payrolls_change", "PAYEMS",  "Total Nonfarm Payrolls Month-over-Month Change",    "thousands", "monthly",   "mom"),
    ("pce_yoy",         "PCEPI",           "PCE Price Index Year-over-Year Change",              "percent",   "monthly",   "yoy"),
    ("core_pce_yoy",    "PCEPILFE",        "Core PCE Price Index Year-over-Year Change",         "percent",   "monthly",   "yoy"),
    ("consumer_sentiment", "UMCSENT",      "University of Michigan Consumer Sentiment",          "index",     "monthly",   "raw"),
    ("manufacturing_employment", "MANEMP", "Manufacturing Employment (PMI proxy)",               "thousands", "monthly",   "raw"),
    ("industrial_production", "INDPRO",    "Industrial Production Index",                        "index",     "monthly",   "raw"),
    ("retail_sales",    "RSAFS",           "Advance Retail Sales",                               "millions_usd", "monthly", "raw"),
    ("home_price_index", "CSUSHPISA",      "S&P/Case-Shiller U.S. National Home Price Index",   "index",     "monthly",   "raw"),
]

# ── FINANCIAL indicators ──────────────────────────────────────────────────────

FINANCIAL_SPECS = [
    ("sp500",             "SP500",       "S&P 500 Index (monthly average)",                  "index",     "monthly", "daily_to_monthly"),
    ("treasury_10y",      "GS10",        "10-Year Treasury Constant Maturity Rate",          "percent",   "monthly", "raw"),
    ("treasury_2y",       "GS2",         "2-Year Treasury Constant Maturity Rate",           "percent",   "monthly", "raw"),
    ("yield_curve_10y2y", "T10Y2Y",      "10-Year minus 2-Year Treasury Spread",             "percent",   "monthly", "raw"),
    ("vix",               "VIXCLS",      "CBOE Volatility Index (monthly average)",          "index",     "monthly", "daily_to_monthly"),
    ("usd_index",         "DTWEXBGS",    "Trade Weighted USD Index Broad (monthly average)", "index",     "monthly", "daily_to_monthly"),
    ("credit_spread",     "BAMLC0A0CM",  "ICE BofA Corporate Bond Spread (monthly average)","percent",   "monthly", "daily_to_monthly"),
    ("m2_money_supply",   "M2SL",        "M2 Money Stock",                                  "billions_usd", "monthly", "raw"),
    ("fed_total_assets",  "WALCL",       "Federal Reserve Total Assets (monthly average)",   "millions_usd", "monthly", "weekly_to_monthly"),
]


def process_specs(specs, label):
    indicators = {}
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for i, (key, fred_id, desc, unit, freq, transform) in enumerate(specs):
        if i > 0:
            time.sleep(1)
        monthly_agg = transform in ("daily_to_monthly", "weekly_to_monthly")
        raw = fetch_fred_series(fred_id, monthly_agg=monthly_agg)
        if raw is None or raw.empty:
            print_status(key, fred_id, None)
            continue

        if transform == "raw":
            series = monthly_to_series(raw)
        elif transform == "raw_quarterly":
            series = quarterly_to_series(raw)
        elif transform == "yoy":
            processed = compute_yoy(raw)
            series = monthly_to_series(processed)
        elif transform == "mom":
            processed = compute_mom_change(raw)
            series = monthly_to_series(processed)
        elif transform in ("daily_to_monthly", "weekly_to_monthly"):
            # FRED already aggregated to monthly via fq=Monthly&fas=avg
            series = monthly_to_series(raw)
        else:
            series = monthly_to_series(raw)

        print_status(key, fred_id, series)

        if series:
            indicators[key] = build_indicator(desc, unit, freq, fred_id, series)

    return indicators


def main():
    print("Fetching US economic data from FRED...")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Macro
    macro_indicators = process_specs(MACRO_SPECS, "MACRO INDICATORS (macro.json)")

    macro_output = {
        "entity": "US",
        "source": "FRED",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "indicators": macro_indicators,
    }

    macro_path = os.path.join(OUTPUT_DIR, "macro.json")
    with open(macro_path, "w", encoding="utf-8") as f:
        json.dump(macro_output, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {macro_path} ({len(macro_indicators)} indicators)")

    # Financial
    financial_indicators = process_specs(FINANCIAL_SPECS, "FINANCIAL INDICATORS (financial.json)")

    financial_output = {
        "entity": "US",
        "source": "FRED",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "indicators": financial_indicators,
    }

    financial_path = os.path.join(OUTPUT_DIR, "financial.json")
    with open(financial_path, "w", encoding="utf-8") as f:
        json.dump(financial_output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {financial_path} ({len(financial_indicators)} indicators)")

    # Summary
    total = len(macro_indicators) + len(financial_indicators)
    expected = len(MACRO_SPECS) + len(FINANCIAL_SPECS)
    print(f"\nTotal: {total}/{expected} indicators fetched successfully")


if __name__ == "__main__":
    main()
