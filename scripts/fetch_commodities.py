#!/usr/bin/env python3
"""
Fetch commodity prices + implied volatility data from multiple free sources.
Sources:
  1. World Bank Pink Sheet (monthly, 1960+, ~50 commodities)
  2. FRED (implied volatility indices + some commodity series)
  3. Yahoo Finance (daily futures, 2000+)

Output: data/economic/commodities/
"""

import json, os, sys, time, io
from datetime import datetime
from pathlib import Path
import urllib.request

# Try imports
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. pip install pandas")
    sys.exit(1)

OUT_DIR = Path(__file__).parent.parent / "data" / "economic" / "commodities"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1971-01-01"
TODAY = datetime.now().strftime("%Y-%m-%d")

def fetch_url(url, timeout=60):
    """Fetch URL with retry."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            print(f"  RETRY {attempt+1}/3: {e}")
            time.sleep(2)
    return None

# ============================================================
#  SOURCE 1: World Bank Pink Sheet (Monthly, 1960+)
# ============================================================
def fetch_world_bank():
    print("\n" + "="*70)
    print("  WORLD BANK COMMODITY PINK SHEET")
    print("="*70)

    url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
    print(f"  Downloading from World Bank...")

    data = fetch_url(url, timeout=120)
    if data is None:
        print("  ERROR: Failed to download World Bank data")
        return {}

    print(f"  Downloaded {len(data)/1024:.0f} KB")

    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name="Monthly Prices", skiprows=4)
    except Exception as e:
        # Try different sheet names
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            print(f"  Available sheets: {xls.sheet_names}")
            # Try first sheet
            df = pd.read_excel(io.BytesIO(data), sheet_name=0, skiprows=4)
        except Exception as e2:
            print(f"  ERROR parsing Excel: {e2}")
            return {}

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns (first 10): {list(df.columns[:10])}")

    # First column should be date
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})

    # Try to parse dates - World Bank uses "YYYYMDD" format like "1960M01"
    def parse_wb_date(d):
        try:
            if isinstance(d, str) and "M" in d:
                return pd.to_datetime(d.replace("M", "-"), format="%Y-%m")
            return pd.to_datetime(d)
        except:
            return pd.NaT

    df["date"] = df["date"].apply(parse_wb_date)
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= START_DATE]
    df = df.sort_values("date").reset_index(drop=True)

    # Map World Bank column names to our standard names
    # WB uses various naming conventions, try to match
    commodity_map = {
        # Energy
        "CRUDE_BRENT": "crude_oil_brent",
        "CRUDE_WTI": "crude_oil_wti",
        "CRUDE_PETRO": "crude_oil_avg",
        "NGAS_US": "natural_gas_us",
        "NGAS_EUR": "natural_gas_eu",
        "NGAS_JP": "natural_gas_jp",
        "COAL_AUS": "coal_australia",
        # Metals
        "GOLD": "gold",
        "SILVER": "silver",
        "PLATINUM": "platinum",
        "COPPER": "copper",
        "ALUMINUM": "aluminum",
        "NICKEL": "nickel",
        "IRON_ORE": "iron_ore",
        "TIN": "tin",
        "ZINC": "zinc",
        "LEAD": "lead",
        # Agriculture
        "SOYBEANS": "soybeans",
        "WHEAT_US_HRW": "wheat",
        "MAIZE": "corn",
        "COTTON_A_INDX": "cotton",
        "SUGAR_WLD": "sugar_world",
        "COFFEE_ARABIC": "coffee_arabica",
        "COFFEE_ROBUS": "coffee_robusta",
        "COCOA": "cocoa",
        "PALM_OIL": "palm_oil",
        "RUBBER1_MYSG": "rubber",
        "TEA_AVG": "tea",
        "RICE_05": "rice",
        "ORANGE": "orange_juice",
        "BANANA_US": "banana",
        # Fertilizers (geopolitically important - Russia/Belarus)
        "DAP": "fertilizer_dap",
        "UREA_EE_BULK": "fertilizer_urea",
        "POTASH": "fertilizer_potash",
        "PHITE": "phosphate_rock",
    }

    results = {}
    found = 0

    # Try exact match first, then fuzzy
    available_cols = {c.upper().strip(): c for c in df.columns if c != "date"}

    for wb_name, our_name in commodity_map.items():
        # Try exact match
        matched_col = None
        for avail_upper, avail_orig in available_cols.items():
            if wb_name.upper() in avail_upper or avail_upper in wb_name.upper():
                matched_col = avail_orig
                break

        if matched_col is None:
            # Try partial match
            for avail_upper, avail_orig in available_cols.items():
                key_parts = wb_name.split("_")
                if any(part in avail_upper for part in key_parts if len(part) > 3):
                    matched_col = avail_orig
                    break

        if matched_col and matched_col in df.columns:
            series = df[["date", matched_col]].dropna()
            series = series[pd.to_numeric(series[matched_col], errors="coerce").notna()]
            if len(series) > 0:
                series[matched_col] = pd.to_numeric(series[matched_col])
                data_points = []
                for _, row in series.iterrows():
                    data_points.append({
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "value": round(float(row[matched_col]), 4)
                    })
                results[our_name] = {
                    "source": "World Bank Pink Sheet",
                    "wb_column": matched_col,
                    "frequency": "monthly",
                    "unit": "USD",
                    "count": len(data_points),
                    "start": data_points[0]["date"],
                    "end": data_points[-1]["date"],
                    "data": data_points
                }
                found += 1
                print(f"  OK  {our_name:35s} ({matched_col:20s}): {len(data_points):5d} points, {data_points[0]['date']} ~ {data_points[-1]['date']}")

    # Also grab any columns we didn't map
    unmapped = [c for c in df.columns if c != "date" and c not in [r.get("wb_column") for r in results.values() if "wb_column" in r]]
    if unmapped:
        print(f"\n  Unmapped columns ({len(unmapped)}): {unmapped[:20]}...")

    print(f"\n  World Bank: {found} commodities fetched")
    return results


# ============================================================
#  SOURCE 2: FRED (Volatility indices + supplementary)
# ============================================================
def fetch_fred_series(series_id, timeout=30):
    """Fetch a single FRED series as CSV."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START_DATE}&coed={TODAY}"
    data = fetch_url(url, timeout=timeout)
    if data is None:
        return None
    try:
        df = pd.read_csv(io.BytesIO(data))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  ERROR parsing {series_id}: {e}")
        return None

def fetch_fred_commodities():
    print("\n" + "="*70)
    print("  FRED: VOLATILITY INDICES + SUPPLEMENTARY COMMODITIES")
    print("="*70)

    fred_series = {
        # Implied Volatility
        "ovx_crude_oil_volatility": "OVXCLS",
        "gvz_gold_volatility": "GVZCLS",
        "vix": "VIXCLS",
        "emv_commodity_vol": "EMVCOMMMKT",
        # Long-history commodities (pre-2000)
        "wti_crude_monthly_long": "WTISPLC",
        "wti_crude_daily": "DCOILWTICO",
        "brent_crude_daily": "DCOILBRENTEU",
        "henry_hub_ng_daily": "DHHNGSP",
        # IMF commodity indices on FRED
        "copper_imf": "PCOPPUSDM",
        "aluminum_imf": "PALUMUSDM",
        "nickel_imf": "PNICKUSDM",
        "iron_ore_imf": "PIORECRUSDM",
        "uranium_imf": "PURANUSDM",
        "coal_imf": "PCOALAUUSDM",
        "soybeans_imf": "PSOYBUSDM",
        "wheat_imf": "PWHEAMTUSDM",
        "corn_imf": "PMAIZMTUSDM",
        "cotton_imf": "PCOTTINDUSDM",
        "sugar_imf": "PSUGAISAUSDM",
        "coffee_arabica_imf": "PCOFFOTMUSDM",
        "coffee_robusta_imf": "PCOFFROBUSDM",
        # Oil indices from FRED (IMF)
        "brent_imf": "POILBREUSDM",
        "wti_imf": "POILWTIUSDM",
        # Natural gas
        "ng_us_imf": "PNGASUSUSDM",
        "ng_eu_imf": "PNGASEUUSDM",
        "lng_japan_imf": "PNGASJPUSDM",
    }

    results = {}
    ok = 0
    fail = 0

    for name, series_id in fred_series.items():
        df = fetch_fred_series(series_id)
        if df is not None and len(df) > 0:
            # Convert to monthly if daily (for consistency)
            is_daily = "daily" in name or series_id.startswith("D") or series_id in ["OVXCLS", "GVZCLS", "VIXCLS"]

            data_points = []
            for _, row in df.iterrows():
                data_points.append({
                    "date": row["date"],
                    "value": round(float(row["value"]), 4)
                })

            freq = "daily" if is_daily else "monthly"
            results[name] = {
                "source": "FRED",
                "series_id": series_id,
                "frequency": freq,
                "count": len(data_points),
                "start": data_points[0]["date"],
                "end": data_points[-1]["date"],
                "data": data_points
            }
            ok += 1
            print(f"  OK  {name:35s} ({series_id:15s}): {len(data_points):5d} points, {data_points[0]['date']} ~ {data_points[-1]['date']}")
        else:
            fail += 1
            print(f"  FAIL {name:35s} ({series_id:15s}): no data")

        time.sleep(0.5)  # Rate limit

    print(f"\n  FRED: {ok} OK, {fail} failed")
    return results


# ============================================================
#  SOURCE 3: Yahoo Finance (Daily futures, 2000+)
# ============================================================
def fetch_yahoo_futures():
    print("\n" + "="*70)
    print("  YAHOO FINANCE: DAILY FUTURES + VOLATILITY")
    print("="*70)

    try:
        import yfinance as yf
    except ImportError:
        print("  SKIP: yfinance not installed (pip install yfinance)")
        return {}

    tickers = {
        # Commodities
        "crude_wti_daily": "CL=F",
        "crude_brent_daily": "BZ=F",
        "gold_daily": "GC=F",
        "silver_daily": "SI=F",
        "copper_daily": "HG=F",
        "natural_gas_daily": "NG=F",
        "soybeans_daily": "ZS=F",
        "wheat_daily": "ZW=F",
        "corn_daily": "ZC=F",
        "palladium_daily": "PA=F",
        "platinum_daily": "PL=F",
        "cotton_daily": "CT=F",
        "sugar_daily": "SB=F",
        "coffee_daily": "KC=F",
        "cocoa_daily": "CC=F",
        # Volatility indices
        "move_bond_vol": "^MOVE",
    }

    results = {}
    ok = 0
    fail = 0

    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=START_DATE, end=TODAY)
            if len(df) > 0:
                data_points = []
                for date, row in df.iterrows():
                    data_points.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "value": round(float(row["Close"]), 4)
                    })
                results[name] = {
                    "source": "Yahoo Finance",
                    "ticker": ticker,
                    "frequency": "daily",
                    "count": len(data_points),
                    "start": data_points[0]["date"],
                    "end": data_points[-1]["date"],
                    "data": data_points
                }
                ok += 1
                print(f"  OK  {name:35s} ({ticker:10s}): {len(data_points):5d} points, {data_points[0]['date']} ~ {data_points[-1]['date']}")
            else:
                fail += 1
                print(f"  FAIL {name:35s} ({ticker:10s}): no data")
        except Exception as e:
            fail += 1
            print(f"  FAIL {name:35s} ({ticker:10s}): {e}")

        time.sleep(0.3)

    print(f"\n  Yahoo Finance: {ok} OK, {fail} failed")
    return results


# ============================================================
#  MAIN
# ============================================================
def main():
    print(f"Fetching commodity data from multiple sources...")
    print(f"Date range: {START_DATE} to {TODAY}")

    all_results = {}

    # Source 1: World Bank
    wb = fetch_world_bank()
    if wb:
        with open(OUT_DIR / "world_bank_commodities.json", "w") as f:
            json.dump(wb, f, indent=2)
        print(f"\nWrote {OUT_DIR / 'world_bank_commodities.json'} ({len(wb)} series)")
        all_results.update({f"wb_{k}": v for k, v in wb.items()})

    # Source 2: FRED
    fred = fetch_fred_commodities()
    if fred:
        with open(OUT_DIR / "fred_commodities.json", "w") as f:
            json.dump(fred, f, indent=2)
        print(f"\nWrote {OUT_DIR / 'fred_commodities.json'} ({len(fred)} series)")
        all_results.update({f"fred_{k}": v for k, v in fred.items()})

    # Source 3: Yahoo Finance
    yf_data = fetch_yahoo_futures()
    if yf_data:
        with open(OUT_DIR / "yahoo_futures.json", "w") as f:
            json.dump(yf_data, f, indent=2)
        print(f"\nWrote {OUT_DIR / 'yahoo_futures.json'} ({len(yf_data)} series)")
        all_results.update({f"yf_{k}": v for k, v in yf_data.items()})

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    total = len(all_results)
    sources = {}
    for k, v in all_results.items():
        src = v.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    for src, count in sorted(sources.items()):
        print(f"  {src:30s}: {count} series")
    print(f"  {'TOTAL':30s}: {total} series")

    # Write manifest
    manifest = {}
    for k, v in all_results.items():
        manifest[k] = {
            "source": v.get("source"),
            "frequency": v.get("frequency"),
            "count": v.get("count"),
            "start": v.get("start"),
            "end": v.get("end"),
        }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'manifest.json'}")

if __name__ == "__main__":
    main()
