#!/usr/bin/env python3
"""Fetch EU (Eurozone) and UK economic data from FRED and output macro.json + financial.json."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd

EU_OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/eu"
UK_OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/uk"
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


def quarterly_to_series(df: pd.DataFrame) -> list[dict]:
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        q_month = ((d.month - 1) // 3) * 3 + 1
        date_str = f"{d.year}-{q_month:02d}"
        entries.append({
            "date": date_str,
            "value": round(float(row["value"]), 4),
            "confidence": "official"
        })
    return entries


def monthly_to_series(df: pd.DataFrame) -> list[dict]:
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


def compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("date").sort_index()
    df = df.resample("MS").last().dropna()
    yoy = df["value"].pct_change(periods=12) * 100
    yoy = yoy.dropna().reset_index()
    yoy.columns = ["date", "value"]
    return yoy


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


def process_specs(specs, label):
    indicators = {}
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for i, (key, fred_id, desc, unit, freq, transform) in enumerate(specs):
        if i > 0:
            time.sleep(1)
        monthly_agg = transform in ("daily_to_monthly",)
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
        elif transform == "daily_to_monthly":
            series = monthly_to_series(raw)
        else:
            series = monthly_to_series(raw)

        print_status(key, fred_id, series)

        if series:
            indicators[key] = build_indicator(desc, unit, freq, fred_id, series)

    return indicators


# ── EU (Eurozone) MACRO ──────────────────────────────────────────────────────

EU_MACRO_SPECS = [
    ("gdp_growth",          "CLVMNACSCAB1GQEA19", "Real GDP EA19 (quarterly, chain-linked volumes)", "millions_eur", "quarterly", "raw_quarterly"),
    ("hicp_all_items",      "CP0000EZ19M086NEST",  "HICP All Items EA19",                             "index_2015",   "monthly",   "raw"),
    ("interest_rate",       "IR3TIB01EZM156N",     "3-Month Interbank Rate EA",                       "percent",      "monthly",   "raw"),
    ("unemployment",        "LRHUTTTTEZM156S",     "Harmonized Unemployment Rate EA",                 "percent",      "monthly",   "raw"),
    ("m3_money_supply",     "MABMM301EZM189S",     "M3 Money Supply EA (growth rate)",                "percent",      "monthly",   "raw"),
    ("industrial_production", "EA19PRINTO01IXOBM",  "Industrial Production Index EA19",                "index",        "monthly",   "raw"),
    ("eur_usd",             "DEXUSEU",             "Exchange Rate USD per EUR (monthly average)",      "usd_per_eur",  "monthly",   "daily_to_monthly"),
]

# ── EU FINANCIAL ─────────────────────────────────────────────────────────────

EU_FINANCIAL_SPECS = [
    ("germany_10y_bond",    "IRLTLT01DEM156N",     "Germany 10-Year Government Bond Yield",           "percent",      "monthly",   "raw"),
]

# ── UK MACRO ─────────────────────────────────────────────────────────────────

UK_MACRO_SPECS = [
    ("gdp_growth",          "CLVMNACSCAB1GQUK",    "Real GDP UK (quarterly, chain-linked volumes)",   "millions_gbp", "quarterly", "raw_quarterly"),
    ("cpi",                 "GBRCPIALLMINMEI",     "CPI All Items UK",                                "index_2015",   "monthly",   "raw"),
    ("interest_rate",       "INTDSRGBM193N",       "Bank of England Bank Rate",                       "percent",      "monthly",   "raw"),
    ("unemployment",        "LMUNRRTTGBM156S",     "Unemployment Rate UK",                            "percent",      "monthly",   "raw"),
    ("industrial_production", "GBRPROINDMISMEI",    "Industrial Production Index UK",                  "index",        "monthly",   "raw"),
    ("gbp_usd",             "DEXUSUK",             "Exchange Rate USD per GBP (monthly average)",      "usd_per_gbp",  "monthly",   "daily_to_monthly"),
    ("m2_money_supply",     "MABMM201GBM189S",    "M2 Money Supply UK (growth rate)",                 "percent",      "monthly",   "raw"),
]

# ── UK FINANCIAL ─────────────────────────────────────────────────────────────

UK_FINANCIAL_SPECS = [
    ("uk_10y_bond",         "IRLTLT01GBM156N",     "UK 10-Year Government Bond Yield",                "percent",      "monthly",   "raw"),
]


def write_json(output_dir: str, filename: str, entity: str, indicators: dict):
    output = {
        "entity": entity,
        "source": "FRED",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "indicators": indicators,
    }
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {path} ({len(indicators)} indicators)")
    return path


def main():
    print("Fetching EU + UK economic data from FRED...")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # ── EU ────────────────────────────────────────────────────────────────────
    eu_macro = process_specs(EU_MACRO_SPECS, "EU (Eurozone) MACRO INDICATORS")
    eu_financial = process_specs(EU_FINANCIAL_SPECS, "EU (Eurozone) FINANCIAL INDICATORS")
    write_json(EU_OUTPUT_DIR, "macro.json", "EU", eu_macro)
    write_json(EU_OUTPUT_DIR, "financial.json", "EU", eu_financial)

    # ── UK ────────────────────────────────────────────────────────────────────
    uk_macro = process_specs(UK_MACRO_SPECS, "UK MACRO INDICATORS")
    uk_financial = process_specs(UK_FINANCIAL_SPECS, "UK FINANCIAL INDICATORS")
    write_json(UK_OUTPUT_DIR, "macro.json", "UK", uk_macro)
    write_json(UK_OUTPUT_DIR, "financial.json", "UK", uk_financial)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for label, macro, financial in [("EU", eu_macro, eu_financial), ("UK", uk_macro, uk_financial)]:
        total = len(macro) + len(financial)
        print(f"\n  {label}:")
        print(f"    Macro:     {len(macro)} indicators")
        for k, v in macro.items():
            s = v["series"]
            print(f"      {k:30s}: {len(s):5d} points, {s[0]['date']} ~ {s[-1]['date']}")
        print(f"    Financial: {len(financial)} indicators")
        for k, v in financial.items():
            s = v["series"]
            print(f"      {k:30s}: {len(s):5d} points, {s[0]['date']} ~ {s[-1]['date']}")
        print(f"    Total:     {total} indicators")


if __name__ == "__main__":
    main()
