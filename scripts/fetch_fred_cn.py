#!/usr/bin/env python3
"""Fetch China economic data from FRED and output macro.json + financial.json."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd

OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/cn"
START_DATE = "1971-01-01"
END_DATE = "2026-03-12"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, monthly_agg: bool = False) -> pd.DataFrame | None:
    """Fetch a FRED series via curl. Use fq=Monthly for daily/weekly series."""
    params = f"id={series_id}&cosd={START_DATE}&coed={END_DATE}"
    if monthly_agg:
        params += "&fq=Monthly&fas=avg"
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


def compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year percent change from an index series."""
    df = df.set_index("date").sort_index()
    df = df.resample("MS").last().dropna()
    yoy = df["value"].pct_change(periods=12) * 100
    yoy = yoy.dropna().reset_index()
    yoy.columns = ["date", "value"]
    return yoy


def quarterly_to_series(df: pd.DataFrame, confidence: str = "cross_validated") -> list[dict]:
    """Convert quarterly data to series entries with YYYY-MM format aligned to quarter start."""
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        q_month = ((d.month - 1) // 3) * 3 + 1
        date_str = f"{d.year}-{q_month:02d}"
        entries.append({
            "date": date_str,
            "value": round(float(row["value"]), 4),
            "confidence": confidence
        })
    return entries


def monthly_to_series(df: pd.DataFrame, confidence: str = "cross_validated") -> list[dict]:
    """Convert monthly data to series entries with YYYY-MM format."""
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        date_str = f"{d.year}-{d.month:02d}"
        entries.append({
            "date": date_str,
            "value": round(float(row["value"]), 4),
            "confidence": confidence
        })
    return entries


def annual_to_series(df: pd.DataFrame, confidence: str = "cross_validated") -> list[dict]:
    """Convert annual data to series entries with YYYY-01 format."""
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        date_str = f"{d.year}-01"
        entries.append({
            "date": date_str,
            "value": round(float(row["value"]), 4),
            "confidence": confidence
        })
    return entries


def print_status(name: str, series_id: str, series: list[dict] | None):
    if series and len(series) > 0:
        print(f"  OK  {name:30s} ({series_id:20s}): {len(series):5d} points, {series[0]['date']} ~ {series[-1]['date']}", flush=True)
    else:
        print(f"  FAIL {name:30s} ({series_id:20s}): no data", flush=True)


def build_indicator(description: str, unit: str, frequency: str, fred_id: str, series: list[dict]) -> dict:
    return {
        "description": description,
        "unit": unit,
        "frequency": frequency,
        "fred_series_id": fred_id,
        "series": series,
    }


# ── MACRO indicators ─────────────────────────────────────────────────────────
# China data on FRED: mix of World Bank (annual), OECD (monthly/quarterly), IMF
# confidence: "official_unreliable" for Chinese official sources,
#             "cross_validated" for OECD/World Bank cross-validated data

MACRO_SPECS = [
    # (key, fred_id, description, unit, frequency, transform, confidence)
    # GDP - World Bank annual
    ("gdp_growth",        "MKTGDPCNA646NWDB",  "GDP (current US$, World Bank)",                      "current_usd",  "annual",    "raw_annual",    "cross_validated"),
    ("gdp_growth_alt",    "NYGDPPCAPKDCHN",     "GDP per capita (constant 2015 US$, World Bank)",     "constant_usd",  "annual",    "raw_annual",    "cross_validated"),

    # CPI
    ("cpi_yoy",           "CHNCPIALLMINMEI",    "CPI All Items (OECD MEI)",                           "index_2015",   "monthly",   "yoy",           "cross_validated"),
    ("cpi_index",         "CHNCPIALLMINMEI",    "CPI All Items Index (OECD MEI, 2015=100)",           "index_2015",   "monthly",   "raw",           "cross_validated"),

    # Interest rates
    ("discount_rate",     "INTDSRCNM193N",      "Central Bank Discount Rate (IMF IFS)",               "percent",      "monthly",   "raw",           "official_unreliable"),
    ("lending_rate",      "LTIRACNM156N",       "Lending Rate (IMF IFS)",                             "percent",      "monthly",   "raw",           "official_unreliable"),

    # Unemployment
    ("unemployment",      "LMUNRRTTCNM156S",    "Harmonized Unemployment Rate (OECD)",                "percent",      "monthly",   "raw",           "cross_validated"),
    ("unemployment_alt",  "LMUNRRTTCNQ156S",    "Harmonized Unemployment Rate Quarterly (OECD)",      "percent",      "quarterly", "raw_quarterly", "cross_validated"),

    # Industrial production
    ("industrial_prod",   "CHNPIEAMPGDPQ",      "Industrial Production incl. Construction (OECD)",    "index_2015",   "quarterly", "raw_quarterly", "cross_validated"),
    ("industrial_prod_m", "CHNPROINDMISMEI",     "Production of Total Industry (OECD MEI)",            "index_2015",   "monthly",   "raw",           "cross_validated"),

    # Money supply
    ("m2",                "MABMM201CNM189S",    "M2 Money Supply (OECD MEI)",                         "national_currency", "monthly", "raw",        "official_unreliable"),
    ("m1",                "MANMM101CNM189S",    "M1 Money Supply (OECD MEI)",                         "national_currency", "monthly", "raw",        "official_unreliable"),

    # Trade
    ("exports",           "XTEXVA01CNM667S",    "Exports of Goods and Services (OECD)",                "usd",         "monthly",   "raw",           "cross_validated"),
    ("imports",           "XTIMVA01CNM667S",    "Imports of Goods and Services (OECD)",                "usd",         "monthly",   "raw",           "cross_validated"),

    # PPI
    ("ppi",               "CHNPIEAMPGDPQ",      "PPI (try OECD quarterly)",                           "index",       "quarterly", "raw_quarterly", "cross_validated"),

    # GDP growth rate (World Bank)
    ("gdp_growth_rate",   "NYGDPMKTPKDZCHN",    "GDP Growth Rate (annual %, World Bank)",              "percent",     "annual",    "raw_annual",    "cross_validated"),

    # Population (for per-capita calculations)
    ("population",        "POPTOTCNA647NWDB",   "Population Total (World Bank)",                       "persons",     "annual",    "raw_annual",    "cross_validated"),

    # Current account
    ("current_account_gdp", "BNCABFUNDCN_GDP",  "Current Account Balance (% of GDP, World Bank)",     "percent",     "annual",    "raw_annual",    "cross_validated"),
]


# ── FINANCIAL indicators ──────────────────────────────────────────────────────

FINANCIAL_SPECS = [
    # Exchange rate
    ("usd_cny",           "DEXCHUS",            "USD/CNY Exchange Rate (daily avg to monthly)",         "cny_per_usd", "monthly",  "daily_to_monthly", "official_unreliable"),

    # Foreign reserves
    ("fx_reserves",       "TRESEGCNM052N",      "Foreign Exchange Reserves (IMF IFS)",                 "usd",         "monthly",  "raw",              "cross_validated"),

    # 10-year government bond yield (if available)
    ("govt_bond_10y",     "IRLTLT01CNM156N",    "Long-Term Government Bond Yield (OECD)",              "percent",     "monthly",  "raw",              "cross_validated"),

    # Short-term interest rate
    ("short_rate",        "IRSTCI01CNM156N",    "Short-Term Interest Rate (OECD)",                      "percent",     "monthly",  "raw",              "cross_validated"),

    # Share prices
    ("share_prices",      "SPASTT01CNM661N",    "Share Prices Total All Shares (OECD)",                 "index_2015",  "monthly",  "raw",              "cross_validated"),

    # Real effective exchange rate
    ("reer",              "CCRETT01CNM661N",     "Real Effective Exchange Rate (BIS, OECD)",             "index",       "monthly",  "raw",              "cross_validated"),
]


def process_specs(specs, label):
    indicators = {}
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for i, (key, fred_id, desc, unit, freq, transform, confidence) in enumerate(specs):
        if i > 0:
            time.sleep(1)
        monthly_agg = transform in ("daily_to_monthly", "weekly_to_monthly")
        raw = fetch_fred_series(fred_id, monthly_agg=monthly_agg)
        if raw is None or raw.empty:
            print_status(key, fred_id, None)
            continue

        if transform == "raw":
            series = monthly_to_series(raw, confidence)
        elif transform == "raw_quarterly":
            series = quarterly_to_series(raw, confidence)
        elif transform == "raw_annual":
            series = annual_to_series(raw, confidence)
        elif transform == "yoy":
            processed = compute_yoy(raw)
            series = monthly_to_series(processed, confidence)
        elif transform in ("daily_to_monthly", "weekly_to_monthly"):
            series = monthly_to_series(raw, confidence)
        else:
            series = monthly_to_series(raw, confidence)

        print_status(key, fred_id, series)

        if series:
            indicators[key] = build_indicator(desc, unit, freq, fred_id, series)

    return indicators


def main():
    print("Fetching China economic data from FRED...")
    print(f"Date range: {START_DATE} to {END_DATE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Macro
    macro_indicators = process_specs(MACRO_SPECS, "CN MACRO INDICATORS (macro.json)")

    macro_output = {
        "entity": "CN",
        "source": "FRED",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "indicators": macro_indicators,
    }

    macro_path = os.path.join(OUTPUT_DIR, "macro.json")
    with open(macro_path, "w", encoding="utf-8") as f:
        json.dump(macro_output, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {macro_path} ({len(macro_indicators)} indicators)")

    # Financial
    financial_indicators = process_specs(FINANCIAL_SPECS, "CN FINANCIAL INDICATORS (financial.json)")

    financial_output = {
        "entity": "CN",
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
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"Macro:     {len(macro_indicators)}/{len(MACRO_SPECS)} indicators")
    print(f"Financial: {len(financial_indicators)}/{len(FINANCIAL_SPECS)} indicators")
    print(f"Total:     {total}/{expected} indicators fetched successfully")

    if total < expected:
        failed = expected - total
        print(f"\n{failed} series failed — these may not exist on FRED for China.")


if __name__ == "__main__":
    main()
