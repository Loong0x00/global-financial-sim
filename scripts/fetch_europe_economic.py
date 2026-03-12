#!/usr/bin/env python3
"""Fetch individual European country economic data from FRED.

Target countries: DE, FR, IT, ES, NL, PL, HU (7 countries)
Indicators: GDP, CPI/inflation, unemployment, debt/GDP, fiscal deficit,
            current account, 10Y bond yield, industrial production,
            plus supplementary (defense spending, energy dependency, etc.)

Data saved to: data/economic/europe/individual_countries.json
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd

OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/europe"
START_DATE = "1990-01-01"
END_DATE = "2026-03-13"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, monthly_agg: bool = False) -> pd.DataFrame | None:
    """Fetch a FRED series via curl CSV endpoint (no API key needed)."""
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


def annual_to_series(df: pd.DataFrame) -> list[dict]:
    entries = []
    for _, row in df.iterrows():
        d = row["date"]
        date_str = f"{d.year}"
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


def build_indicator(description: str, unit: str, frequency: str, fred_id: str, series: list[dict]) -> dict:
    return {
        "description": description,
        "unit": unit,
        "frequency": frequency,
        "fred_series_id": fred_id,
        "series": series,
    }


def print_status(country: str, name: str, series_id: str, series: list[dict] | None):
    if series and len(series) > 0:
        print(f"  OK  [{country}] {name:35s} ({series_id:25s}): {len(series):5d} pts, {series[0]['date']} ~ {series[-1]['date']}", flush=True)
    else:
        print(f"  FAIL [{country}] {name:35s} ({series_id:25s}): no data", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FRED Series IDs for European Countries
# ══════════════════════════════════════════════════════════════════════════════
# Format: (indicator_key, fred_series_id, description, unit, frequency, transform)
#
# FRED naming patterns for European data:
#   GDP: CLVMNACSCAB1GQ{CC} (chain-linked volumes, quarterly)
#   GDP growth rate: CLVMNACSCAB1GQDE → compute YoY from levels
#   CPI: CP0000{CC}M086NEST (HICP all items) or CPALTT01{CC}M657N
#   Unemployment: LRHUTTTT{CC}M156S (harmonized) or LMUNRRTT{CC}M156S
#   Industrial prod: {CC}PRINTO01IXOBM or {CC}PROINDMISMEI
#   10Y bond: IRLTLT01{CC}M156N
#   Current account: BN6BLTT02{CC}Q628S or BPBLTT01{CC}Q637S
#   Govt debt/GDP: GGGDTA{CC}GDP or DEBTTL{CC}188A (annual)
#   Fiscal balance: GGNLBA{CC}GDP (annual)

COUNTRIES = {
    "DE": {
        "name": "Germany",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQDE",  "Real GDP Germany (quarterly, chain-linked volumes)",   "millions_eur", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01DEQ657S",   "GDP Growth Rate Germany (YoY %)",                      "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000DEM086NEST",  "HICP All Items Germany",                                "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "DEUCPIALLMINMEI",   "CPI All Items Germany (index)",                         "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTDEM156S",   "Harmonized Unemployment Rate Germany",                  "percent",      "monthly",   "raw"),
            ("industrial_production", "DEUPROINDMISMEI",   "Industrial Production Index Germany",                   "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01DEQ637S",   "Current Account Balance Germany (quarterly)",           "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01DEM156N",   "Germany 10-Year Government Bond Yield",                 "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTADEA188N",     "Government Debt/GDP Germany",                           "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBADEA188N",     "Government Net Lending/Borrowing (% GDP) Germany",      "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINADEA646NWDB",  "Military Expenditure (% GDP) Germany",                  "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
    "FR": {
        "name": "France",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQFR",  "Real GDP France (quarterly)",                          "millions_eur", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01FRQ657S",   "GDP Growth Rate France (YoY %)",                       "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000FRM086NEST",  "HICP All Items France",                                 "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "FRACPIALLMINMEI",   "CPI All Items France (index)",                          "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTFRM156S",   "Harmonized Unemployment Rate France",                   "percent",      "monthly",   "raw"),
            ("industrial_production", "FRAPROINDMISMEI",   "Industrial Production Index France",                    "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01FRQ637S",   "Current Account Balance France (quarterly)",            "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01FRM156N",   "France 10-Year Government Bond Yield",                  "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTAFRA188N",     "Government Debt/GDP France",                            "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBAFRA188N",     "Government Net Lending/Borrowing (% GDP) France",       "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINAFRA646NWDB",  "Military Expenditure (% GDP) France",                   "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
    "IT": {
        "name": "Italy",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQIT",  "Real GDP Italy (quarterly)",                           "millions_eur", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01ITQ657S",   "GDP Growth Rate Italy (YoY %)",                        "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000ITM086NEST",  "HICP All Items Italy",                                  "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "ITACPIALLMINMEI",   "CPI All Items Italy (index)",                           "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTITM156S",   "Harmonized Unemployment Rate Italy",                    "percent",      "monthly",   "raw"),
            ("industrial_production", "ITAPROINDMISMEI",   "Industrial Production Index Italy",                     "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01ITQ637S",   "Current Account Balance Italy (quarterly)",             "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01ITM156N",   "Italy 10-Year Government Bond Yield",                   "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTAITA188N",     "Government Debt/GDP Italy",                             "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBAITA188N",     "Government Net Lending/Borrowing (% GDP) Italy",        "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINAITA646NWDB",  "Military Expenditure (% GDP) Italy",                    "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
    "ES": {
        "name": "Spain",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQES",  "Real GDP Spain (quarterly)",                           "millions_eur", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01ESQ657S",   "GDP Growth Rate Spain (YoY %)",                        "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000ESM086NEST",  "HICP All Items Spain",                                  "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "ESPCPIALLMINMEI",   "CPI All Items Spain (index)",                           "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTESM156S",   "Harmonized Unemployment Rate Spain",                    "percent",      "monthly",   "raw"),
            ("industrial_production", "ESPPROINDMISMEI",   "Industrial Production Index Spain",                     "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01ESQ637S",   "Current Account Balance Spain (quarterly)",             "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01ESM156N",   "Spain 10-Year Government Bond Yield",                   "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTAESA188N",     "Government Debt/GDP Spain",                             "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBAESA188N",     "Government Net Lending/Borrowing (% GDP) Spain",        "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINAESA646NWDB",  "Military Expenditure (% GDP) Spain",                    "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
    "NL": {
        "name": "Netherlands",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQNL",  "Real GDP Netherlands (quarterly)",                     "millions_eur", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01NLQ657S",   "GDP Growth Rate Netherlands (YoY %)",                  "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000NLM086NEST",  "HICP All Items Netherlands",                            "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "NLDCPIALLMINMEI",   "CPI All Items Netherlands (index)",                     "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTNLM156S",   "Harmonized Unemployment Rate Netherlands",              "percent",      "monthly",   "raw"),
            ("industrial_production", "NLDPROINDMISMEI",   "Industrial Production Index Netherlands",               "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01NLQ637S",   "Current Account Balance Netherlands (quarterly)",       "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01NLM156N",   "Netherlands 10-Year Government Bond Yield",             "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTANLA188N",     "Government Debt/GDP Netherlands",                       "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBANLA188N",     "Government Net Lending/Borrowing (% GDP) Netherlands",  "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINANLA646NWDB",  "Military Expenditure (% GDP) Netherlands",              "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
    "PL": {
        "name": "Poland",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQPL",  "Real GDP Poland (quarterly)",                          "millions_pln", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01PLQ657S",   "GDP Growth Rate Poland (YoY %)",                       "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000PLM086NEST",  "HICP All Items Poland",                                 "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "POLCPIALLMINMEI",   "CPI All Items Poland (index)",                          "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTPLM156S",   "Harmonized Unemployment Rate Poland",                   "percent",      "monthly",   "raw"),
            ("industrial_production", "POLPROINDMISMEI",   "Industrial Production Index Poland",                    "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01PLQ637S",   "Current Account Balance Poland (quarterly)",            "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01PLM156N",   "Poland 10-Year Government Bond Yield",                  "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTAPLA188N",     "Government Debt/GDP Poland",                            "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBAPLA188N",     "Government Net Lending/Borrowing (% GDP) Poland",       "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINAPLA646NWDB",  "Military Expenditure (% GDP) Poland",                   "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
    "HU": {
        "name": "Hungary",
        "macro": [
            ("gdp_real",              "CLVMNACSCAB1GQHU",  "Real GDP Hungary (quarterly)",                         "millions_huf", "quarterly", "raw_quarterly"),
            ("gdp_growth_yoy",        "NAEXKP01HUQ657S",   "GDP Growth Rate Hungary (YoY %)",                      "percent",      "quarterly", "raw_quarterly"),
            ("hicp",                  "CP0000HUM086NEST",  "HICP All Items Hungary",                                "index_2015",   "monthly",   "raw"),
            ("cpi_yoy",               "HUNCPIALLMINMEI",   "CPI All Items Hungary (index)",                         "index",        "monthly",   "yoy"),
            ("unemployment",          "LRHUTTTTHUM156S",   "Harmonized Unemployment Rate Hungary",                  "percent",      "monthly",   "raw"),
            ("industrial_production", "HUNPROINDMISMEI",   "Industrial Production Index Hungary",                   "index",        "monthly",   "raw"),
            ("current_account",       "BPBLTT01HUQ637S",   "Current Account Balance Hungary (quarterly)",           "millions_usd", "quarterly", "raw_quarterly"),
        ],
        "financial": [
            ("bond_10y",              "IRLTLT01HUM156N",   "Hungary 10-Year Government Bond Yield",                 "percent",      "monthly",   "raw"),
        ],
        "supplementary": [
            ("govt_debt_gdp",         "GGGDTAHUA188N",     "Government Debt/GDP Hungary",                           "percent_gdp",  "annual",    "raw_annual"),
            ("fiscal_balance_gdp",    "GGNLBAHUA188N",     "Government Net Lending/Borrowing (% GDP) Hungary",      "percent_gdp",  "annual",    "raw_annual"),
            ("defense_spending_gdp",  "MIKINAHUA646NWDB",  "Military Expenditure (% GDP) Hungary",                  "percent_gdp",  "annual",    "raw_annual"),
        ],
    },
}

# ── Spread calculations (IT-DE, ES-DE bond spreads) ──────────────────────────
SPREAD_PAIRS = [
    ("it_de_spread", "IT", "DE", "Italy-Germany 10Y Bond Spread"),
    ("es_de_spread", "ES", "DE", "Spain-Germany 10Y Bond Spread"),
    ("pl_de_spread", "PL", "DE", "Poland-Germany 10Y Bond Spread"),
    ("hu_de_spread", "HU", "DE", "Hungary-Germany 10Y Bond Spread"),
    ("fr_de_spread", "FR", "DE", "France-Germany 10Y Bond Spread"),
]

# ── Alternative FRED series IDs to try if primary ones fail ──────────────────
# Some FRED IDs use different country code conventions
FALLBACK_SERIES = {
    # Unemployment alternatives
    "LRHUTTTTDEM156S": ["LMUNRRTTDEM156S"],
    "LRHUTTTTFRM156S": ["LMUNRRTTFRM156S"],
    "LRHUTTTTITM156S": ["LMUNRRTTITM156S"],
    "LRHUTTTTESM156S": ["LMUNRRTTESM156S"],
    "LRHUTTTTNLM156S": ["LMUNRRTTNLM156S"],
    "LRHUTTTTPLM156S": ["LMUNRRTTPLM156S"],
    "LRHUTTTTHU M156S": ["LRHUTTTTHUM156S", "LMUNRRTTHUM156S"],
    # Industrial production alternatives
    "DEUPROINDMISMEI": ["DEUPRODMISMEI", "EA19PRINTO01IXOBDE"],
    "FRAPROINDMISMEI": ["FRAPRODMISMEI"],
    "ITAPROINDMISMEI": ["ITAPRODMISMEI"],
    "ESPPROINDMISMEI": ["ESPPRODMISMEI"],
    "NLDPROINDMISMEI": ["NLDPRODMISMEI"],
    "POLPROINDMISMEI": ["POLPRODMISMEI"],
    "HUNPROINDMISMEI": ["HUNPRODMISMEI"],
    # Current account alternatives
    "BPBLTT01DEQ637S": ["BN6BLTT02DEQ628S"],
    "BPBLTT01FRQ637S": ["BN6BLTT02FRQ628S"],
    "BPBLTT01ITQ637S": ["BN6BLTT02ITQ628S"],
    "BPBLTT01ESQ637S": ["BN6BLTT02ESQ628S"],
    "BPBLTT01NLQ637S": ["BN6BLTT02NLQ628S"],
    "BPBLTT01PLQ637S": ["BN6BLTT02PLQ628S"],
    "BPBLTT01HUQ637S": ["BN6BLTT02HUQ628S"],
    # GDP growth alternatives
    "NAEXKP01DEQ657S": ["NAEXKP01DEQ661S"],
    "NAEXKP01FRQ657S": ["NAEXKP01FRQ661S"],
    "NAEXKP01ITQ657S": ["NAEXKP01ITQ661S"],
    "NAEXKP01ESQ657S": ["NAEXKP01ESQ661S"],
    "NAEXKP01NLQ657S": ["NAEXKP01NLQ661S"],
    "NAEXKP01PLQ657S": ["NAEXKP01PLQ661S"],
    "NAEXKP01HUQ657S": ["NAEXKP01HUQ661S"],
    # Govt debt alternatives
    "GGGDTADEA188N": ["DEBTTLDEA188A"],
    "GGGDTAFRA188N": ["DEBTTLFRA188A"],
    "GGGDTAITA188N": ["DEBTTLITA188A"],
    "GGGDTAESA188N": ["DEBTTLESA188A"],
    "GGGDTANLA188N": ["DEBTTLNLA188A"],
    "GGGDTAPLA188N": ["DEBTTLPLA188A"],
    "GGGDTAHUA188N": ["DEBTTLHUA188A"],
}


def fetch_with_fallback(series_id: str, monthly_agg: bool = False) -> tuple[pd.DataFrame | None, str]:
    """Try primary series ID, then fallbacks."""
    # Fix any whitespace in series IDs
    series_id = series_id.replace(" ", "")

    df = fetch_fred_series(series_id, monthly_agg=monthly_agg)
    if df is not None and not df.empty:
        return df, series_id

    # Try fallbacks
    fallbacks = FALLBACK_SERIES.get(series_id, [])
    for fb_id in fallbacks:
        fb_id = fb_id.replace(" ", "")
        time.sleep(0.5)
        df = fetch_fred_series(fb_id, monthly_agg=monthly_agg)
        if df is not None and not df.empty:
            print(f"    -> Used fallback: {fb_id}", flush=True)
            return df, fb_id

    return None, series_id


def process_country_specs(country_code: str, specs: list[tuple], label: str) -> dict:
    """Process a list of (key, fred_id, desc, unit, freq, transform) specs."""
    indicators = {}
    country_name = COUNTRIES[country_code]["name"]

    print(f"\n{'─'*70}")
    print(f"  [{country_code}] {country_name} - {label}")
    print(f"{'─'*70}")

    for i, (key, fred_id, desc, unit, freq, transform) in enumerate(specs):
        if i > 0:
            time.sleep(0.8)  # Rate limiting

        monthly_agg = transform in ("daily_to_monthly",)
        raw, actual_id = fetch_with_fallback(fred_id, monthly_agg=monthly_agg)

        if raw is None or raw.empty:
            print_status(country_code, key, fred_id, None)
            continue

        if transform == "raw":
            series = monthly_to_series(raw)
        elif transform == "raw_quarterly":
            series = quarterly_to_series(raw)
        elif transform == "raw_annual":
            series = annual_to_series(raw)
        elif transform == "yoy":
            processed = compute_yoy(raw)
            series = monthly_to_series(processed)
        elif transform == "daily_to_monthly":
            series = monthly_to_series(raw)
        else:
            series = monthly_to_series(raw)

        print_status(country_code, key, actual_id, series)

        if series:
            indicators[key] = build_indicator(desc, unit, freq, actual_id, series)

    return indicators


def compute_spreads(all_data: dict) -> dict:
    """Compute bond yield spreads vs Germany."""
    spreads = {}

    print(f"\n{'═'*70}")
    print(f"  BOND YIELD SPREADS (vs Germany)")
    print(f"{'═'*70}")

    # Get DE 10Y series
    de_financial = all_data.get("DE", {}).get("financial", {})
    de_bond = de_financial.get("bond_10y")
    if not de_bond:
        print("  SKIP: No German bond data for spread calculation", flush=True)
        return spreads

    de_series = {e["date"]: e["value"] for e in de_bond["series"]}

    for spread_key, country_a, country_b, desc in SPREAD_PAIRS:
        a_financial = all_data.get(country_a, {}).get("financial", {})
        a_bond = a_financial.get("bond_10y")
        if not a_bond:
            print(f"  SKIP {spread_key}: No {country_a} bond data", flush=True)
            continue

        spread_series = []
        for entry in a_bond["series"]:
            date = entry["date"]
            if date in de_series:
                spread_val = round(entry["value"] - de_series[date], 4)
                spread_series.append({
                    "date": date,
                    "value": spread_val,
                    "confidence": "derived"
                })

        if spread_series:
            spreads[spread_key] = build_indicator(
                desc, "percent", "monthly", f"derived:{country_a}-DE", spread_series
            )
            print(f"  OK  {spread_key:25s}: {len(spread_series):5d} pts, {spread_series[0]['date']} ~ {spread_series[-1]['date']}", flush=True)
        else:
            print(f"  FAIL {spread_key}: no overlapping dates", flush=True)

    return spreads


def main():
    print("=" * 70)
    print("  EUROPEAN INDIVIDUAL COUNTRY DATA COLLECTION (FRED)")
    print("=" * 70)
    country_list = ', '.join(f'{cc} ({COUNTRIES[cc]["name"]})' for cc in COUNTRIES)
    print(f"  Countries: {country_list}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Output: {OUTPUT_DIR}/individual_countries.json")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = {}
    total_indicators = 0
    total_failed = 0

    for cc, config in COUNTRIES.items():
        country_data = {}

        # Macro indicators
        macro = process_country_specs(cc, config["macro"], "MACRO")
        country_data["macro"] = macro

        # Financial indicators
        financial = process_country_specs(cc, config["financial"], "FINANCIAL")
        country_data["financial"] = financial

        # Supplementary indicators
        supplementary = process_country_specs(cc, config["supplementary"], "SUPPLEMENTARY")
        country_data["supplementary"] = supplementary

        all_data[cc] = country_data

        n_ok = len(macro) + len(financial) + len(supplementary)
        n_total = len(config["macro"]) + len(config["financial"]) + len(config["supplementary"])
        n_fail = n_total - n_ok
        total_indicators += n_ok
        total_failed += n_fail

        print(f"\n  [{cc}] {config['name']}: {n_ok}/{n_total} indicators OK", flush=True)

    # Compute spreads
    spreads = compute_spreads(all_data)

    # ── Build output JSON ────────────────────────────────────────────────────
    output = {
        "entity": "Europe Individual Countries",
        "source": "FRED",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "countries": {},
        "cross_country": {
            "bond_spreads": spreads,
        },
    }

    for cc, config in COUNTRIES.items():
        country_output = {
            "name": config["name"],
            "iso_code": cc,
            "indicators": {},
        }
        for category in ["macro", "financial", "supplementary"]:
            for key, indicator in all_data[cc].get(category, {}).items():
                # Prefix with category for clarity
                full_key = f"{key}" if category == "macro" else f"{category}_{key}" if category == "supplementary" else key
                country_output["indicators"][full_key] = indicator

        output["countries"][cc] = country_output

    # Write main output
    output_path = os.path.join(OUTPUT_DIR, "individual_countries.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Also write per-country files for easy access
    for cc in COUNTRIES:
        country_dir = os.path.join(OUTPUT_DIR, cc.lower())
        os.makedirs(country_dir, exist_ok=True)
        country_file = os.path.join(country_dir, "macro.json")
        country_out = {
            "entity": f"{COUNTRIES[cc]['name']} ({cc})",
            "source": "FRED",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "indicators": output["countries"][cc]["indicators"],
        }
        with open(country_file, "w", encoding="utf-8") as f:
            json.dump(country_out, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {country_file}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"{'═'*70}")
    print(f"\n  Total indicators fetched: {total_indicators}")
    print(f"  Total failed: {total_failed}")
    print(f"  Bond spreads computed: {len(spreads)}")
    print()

    for cc, config in COUNTRIES.items():
        data = all_data[cc]
        n = sum(len(data[cat]) for cat in ["macro", "financial", "supplementary"])
        print(f"  [{cc}] {config['name']:15s}: {n:2d} indicators")
        for cat in ["macro", "financial", "supplementary"]:
            for key, ind in data[cat].items():
                s = ind["series"]
                print(f"    {cat:15s} {key:35s}: {len(s):5d} pts, {s[0]['date']} ~ {s[-1]['date']}")

    if spreads:
        print(f"\n  Cross-country spreads:")
        for key, ind in spreads.items():
            s = ind["series"]
            print(f"    {key:25s}: {len(s):5d} pts, {s[0]['date']} ~ {s[-1]['date']}")

    print(f"\n  Output: {output_path}")
    print(f"  Per-country dirs: {OUTPUT_DIR}/{{de,fr,it,es,nl,pl,hu}}/macro.json")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
