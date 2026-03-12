#!/usr/bin/env python3
"""
Fetch Japan (JP) and Russia (RU) economic data from FRED.
Outputs macro.json and financial.json for each country.
"""

import csv
import io
import json
import subprocess
import time
import sys
from datetime import date

BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
START_DATE = "1971-01-01"
END_DATE = "2026-03-12"
TODAY = "2026-03-12"

# --- Series definitions ---

JP_MACRO = {
    "gdp_growth": {
        "series_id": "JPNRGDPEXP",
        "description": "Real GDP Japan (Expenditure Approach)",
        "unit": "index_2015=100",
        "frequency": "quarterly",
    },
    "cpi": {
        "series_id": "JPNCPIALLMINMEI",
        "description": "CPI All Items Japan",
        "unit": "index_2015=100",
        "frequency": "monthly",
    },
    "interest_rate": {
        "series_id": "IRSTCI01JPM156N",
        "description": "Immediate Rate (Call Rate) Japan",
        "unit": "percent",
        "frequency": "monthly",
    },
    "unemployment": {
        "series_id": "LMUNRRTTJPM156S",
        "description": "Unemployment Rate Japan",
        "unit": "percent",
        "frequency": "monthly",
    },
    "industrial_production": {
        "series_id": "JPNPROINDMISMEI",
        "description": "Industrial Production Index Japan",
        "unit": "index_2015=100",
        "frequency": "monthly",
    },
    "m2": {
        "series_id": "MABMM201JPM189S",
        "description": "M2 Money Supply Japan",
        "unit": "national_currency",
        "frequency": "monthly",
    },
    "exchange_rate_usd_jpy": {
        "series_id": "DEXJPUS",
        "description": "USD/JPY Exchange Rate",
        "unit": "jpy_per_usd",
        "frequency": "daily",
        "aggregate_monthly": True,
    },
}

JP_FINANCIAL = {
    "govt_bond_10y": {
        "series_id": "IRLTLT01JPM156N",
        "description": "10-Year Government Bond Yield Japan",
        "unit": "percent",
        "frequency": "monthly",
    },
    "nikkei225": {
        "series_id": "NIKKEI225",
        "description": "Nikkei 225 Stock Index",
        "unit": "index",
        "frequency": "daily",
        "aggregate_monthly": True,
    },
}

RU_MACRO = {
    "gdp": {
        "series_id": "MKTGDPRUA646NWDB",
        "description": "GDP Russia (World Bank, current USD)",
        "unit": "current_usd",
        "frequency": "annual",
    },
    "cpi": {
        "series_id": "RUSCPIALLMINMEI",
        "description": "CPI All Items Russia",
        "unit": "index_2015=100",
        "frequency": "monthly",
    },
    "interest_rate": {
        "series_id": "INTDSRRUM193N",
        "description": "Discount Rate Russia",
        "unit": "percent",
        "frequency": "monthly",
    },
    "unemployment": {
        "series_id": "LMUNRRTTRUM156S",
        "description": "Unemployment Rate Russia",
        "unit": "percent",
        "frequency": "monthly",
    },
    "industrial_production": {
        "series_id": "RUSPROINDMISMEI",
        "description": "Industrial Production Index Russia",
        "unit": "index_2015=100",
        "frequency": "monthly",
    },
    "exchange_rate_usd_rub": {
        "series_id": "DEXRUS",
        "description": "USD/RUB Exchange Rate",
        "unit": "rub_per_usd",
        "frequency": "daily",
        "aggregate_monthly": True,
    },
    "m2": {
        "series_id": "MABMM201RUM189S",
        "description": "M2 Money Supply Russia",
        "unit": "national_currency",
        "frequency": "monthly",
    },
}

RU_FINANCIAL = {}  # No reliable FRED financial series for Russia


def fetch_fred_csv(series_id, aggregate_monthly=False):
    """Fetch a FRED series as CSV via curl. Returns list of (date_str, value)."""
    url = f"{BASE_URL}?id={series_id}&cosd={START_DATE}&coed={END_DATE}"
    if aggregate_monthly:
        url += "&fq=Monthly&fas=avg"

    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-sS", "--fail", "-L", "--max-time", "60", url],
                capture_output=True, text=True, timeout=90
            )
            if result.returncode == 0:
                break
            print(f"  WARN: curl returned {result.returncode} for {series_id} (attempt {attempt+1})")
        except subprocess.TimeoutExpired:
            print(f"  WARN: timeout for {series_id} (attempt {attempt+1})")
        if attempt < 2:
            time.sleep(3)
    else:
        print(f"  ERROR: failed after 3 attempts for {series_id}")
        return []

    if result.returncode != 0:
        print(f"  WARN: curl failed for {series_id}: {result.stderr.strip()}")
        return []

    rows = []
    reader = csv.reader(io.StringIO(result.stdout))
    header = next(reader, None)
    if not header:
        print(f"  WARN: empty response for {series_id}")
        return []

    for row in reader:
        if len(row) < 2:
            continue
        date_str, val_str = row[0], row[1]
        if val_str == "." or val_str == "" or val_str == "#N/A":
            continue
        try:
            value = float(val_str)
        except ValueError:
            continue
        # Convert YYYY-MM-DD to YYYY-MM
        date_short = date_str[:7]
        rows.append((date_short, value))

    return rows


def build_indicator(defn, confidence, rows):
    """Build an indicator dict from definition and fetched rows."""
    series = [
        {"date": d, "value": v, "confidence": confidence}
        for d, v in rows
    ]
    ind = {
        "description": defn["description"],
        "unit": defn["unit"],
        "frequency": "monthly" if defn.get("aggregate_monthly") else defn["frequency"],
        "fred_series_id": defn["series_id"],
        "series": series,
    }
    return ind


def fetch_and_build(indicators_def, entity, confidence):
    """Fetch all series for a country and build the indicators dict."""
    indicators = {}
    for key, defn in indicators_def.items():
        sid = defn["series_id"]
        agg = defn.get("aggregate_monthly", False)
        print(f"  Fetching {sid} ({key})...", end=" ", flush=True)
        rows = fetch_fred_csv(sid, aggregate_monthly=agg)
        print(f"{len(rows)} data points")
        if rows:
            indicators[key] = build_indicator(defn, confidence, rows)
        time.sleep(1)  # Be polite to FRED
    return indicators


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


def main():
    summary = {}

    # --- Japan ---
    print("\n=== JAPAN (JP) macro.json ===")
    jp_macro_indicators = fetch_and_build(JP_MACRO, "JP", "official")
    jp_macro = {
        "entity": "JP",
        "source": "FRED",
        "last_updated": TODAY,
        "indicators": jp_macro_indicators,
    }
    write_json("/home/user/global-financial-sim/data/economic/jp/macro.json", jp_macro)
    summary["JP macro"] = {k: len(v["series"]) for k, v in jp_macro_indicators.items()}

    print("\n=== JAPAN (JP) financial.json ===")
    jp_fin_indicators = fetch_and_build(JP_FINANCIAL, "JP", "official")
    jp_fin = {
        "entity": "JP",
        "source": "FRED",
        "last_updated": TODAY,
        "indicators": jp_fin_indicators,
    }
    write_json("/home/user/global-financial-sim/data/economic/jp/financial.json", jp_fin)
    summary["JP financial"] = {k: len(v["series"]) for k, v in jp_fin_indicators.items()}

    # --- Russia ---
    print("\n=== RUSSIA (RU) macro.json ===")
    ru_macro_indicators = fetch_and_build(RU_MACRO, "RU", "official_unreliable")
    ru_macro = {
        "entity": "RU",
        "source": "FRED",
        "last_updated": TODAY,
        "indicators": ru_macro_indicators,
    }
    write_json("/home/user/global-financial-sim/data/economic/ru/macro.json", ru_macro)
    summary["RU macro"] = {k: len(v["series"]) for k, v in ru_macro_indicators.items()}

    # Russia financial - minimal
    if RU_FINANCIAL:
        print("\n=== RUSSIA (RU) financial.json ===")
        ru_fin_indicators = fetch_and_build(RU_FINANCIAL, "RU", "official_unreliable")
    else:
        ru_fin_indicators = {}
        print("\n=== RUSSIA (RU) financial.json (no FRED series, creating empty) ===")
    ru_fin = {
        "entity": "RU",
        "source": "FRED",
        "last_updated": TODAY,
        "indicators": ru_fin_indicators,
    }
    write_json("/home/user/global-financial-sim/data/economic/ru/financial.json", ru_fin)
    summary["RU financial"] = {k: len(v["series"]) for k, v in ru_fin_indicators.items()}

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for section, counts in summary.items():
        print(f"\n{section}:")
        if not counts:
            print("  (no indicators)")
        for k, n in counts.items():
            print(f"  {k}: {n} data points")

    # Date range info
    print("\n--- Date Ranges ---")
    for label, data in [("JP macro", jp_macro), ("JP financial", jp_fin),
                         ("RU macro", ru_macro), ("RU financial", ru_fin)]:
        print(f"\n{label}:")
        for k, ind in data["indicators"].items():
            s = ind["series"]
            if s:
                print(f"  {k}: {s[0]['date']} to {s[-1]['date']}")
            else:
                print(f"  {k}: (empty)")

    print("\nDone.")


if __name__ == "__main__":
    main()
