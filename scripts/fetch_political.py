#!/usr/bin/env python3
"""Fetch World Bank Governance Indicators for key countries.

Indicators:
  VA.EST  Voice and Accountability
  PV.EST  Political Stability and Absence of Violence/Terrorism
  GE.EST  Government Effectiveness
  RQ.EST  Regulatory Quality
  RL.EST  Rule of Law
  CC.EST  Control of Corruption

Countries: USA, CHN, JPN, DEU, GBR, FRA, RUS, IRN, ISR
Date range: 1996-2024 (WGI data starts 1996)

Output: data/political/governance_indicators.json

Note: Uses allorigins.win CORS proxy to work around Cloudflare TLS
incompatibility with OpenSSL 3.6+ on api.worldbank.org.
"""

import json
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime

# World Bank API base
WB_API_BASE = "https://api.worldbank.org/v2/country"

# CORS proxy (needed because api.worldbank.org Cloudflare blocks OpenSSL 3.6 TLS)
PROXY_BASE = "https://api.allorigins.win/get?url="

COUNTRIES = ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "RUS", "IRN", "ISR"]
DATE_RANGE = "1996:2024"
OUTPUT_DIR = "/home/user/global-financial-sim/data/political"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "governance_indicators.json")

INDICATORS = {
    "VA.EST": "Voice and Accountability",
    "PV.EST": "Political Stability and Absence of Violence/Terrorism",
    "GE.EST": "Government Effectiveness",
    "RQ.EST": "Regulatory Quality",
    "RL.EST": "Rule of Law",
    "CC.EST": "Control of Corruption",
}

COUNTRY_NAMES = {
    "USA": "United States", "CHN": "China", "JPN": "Japan",
    "DEU": "Germany", "GBR": "United Kingdom", "FRA": "France",
    "RUS": "Russian Federation", "IRN": "Iran, Islamic Rep.", "ISR": "Israel",
}


def fetch_via_proxy(target_url: str, timeout: int = 30) -> str | None:
    """Fetch a URL through the allorigins CORS proxy. Returns raw text or None."""
    proxy_url = PROXY_BASE + urllib.parse.quote(target_url, safe="")
    for attempt in range(3):
        try:
            req = urllib.request.Request(proxy_url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                wrapper = json.loads(resp.read().decode("utf-8"))
                return wrapper.get("contents", "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            print(f"    Proxy attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None


def fetch_indicator_for_country(indicator_code: str, country_iso3: str) -> list[dict]:
    """Fetch a single WGI indicator for a single country."""
    target_url = (
        f"{WB_API_BASE}/{country_iso3}/indicator/{indicator_code}"
        f"?date={DATE_RANGE}&format=json&per_page=500"
    )

    raw = fetch_via_proxy(target_url)
    if raw is None:
        print(f"    FAILED: {country_iso3}/{indicator_code}")
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"    Invalid JSON for {country_iso3}/{indicator_code}")
        return []

    if not isinstance(data, list) or len(data) < 2:
        print(f"    No data for {country_iso3}/{indicator_code}")
        return []

    records = data[1]
    if records is None:
        return []

    results = []
    non_null = 0
    for rec in records:
        value = rec.get("value")
        if value is not None:
            non_null += 1
        results.append({
            "indicator_code": indicator_code,
            "indicator_name": INDICATORS[indicator_code],
            "country_iso2": rec.get("country", {}).get("id", ""),
            "country_iso3": rec.get("countryiso3code", country_iso3),
            "country_name": rec.get("country", {}).get("value", ""),
            "year": int(rec.get("date", 0)) if rec.get("date") else None,
            "value": round(value, 4) if value is not None else None,
        })

    print(f"    {country_iso3}: {non_null}/{len(results)} non-null")
    return results


def fetch_all_indicators() -> list[dict]:
    """Fetch all indicators for all countries."""
    all_records = []
    for code, name in INDICATORS.items():
        print(f"\n{'='*60}")
        print(f"Indicator: {code} ({name})")
        print(f"{'='*60}")
        for country in COUNTRIES:
            records = fetch_indicator_for_country(code, country)
            all_records.extend(records)
            time.sleep(0.5)  # Be polite
    return all_records


def try_freedom_house() -> dict | None:
    """Try to fetch Freedom House aggregate scores via CORS proxy.

    Freedom House publishes an Excel file with country ratings 1973-2024.
    """
    print(f"\n{'='*60}")
    print("Attempting Freedom House data fetch...")

    # Try to get the Excel file info
    fh_url = "https://freedomhouse.org/sites/default/files/2024-02/Country_and_Territory_Ratings_and_Statuses_FIW_1973-2024.xlsx"
    try:
        raw = fetch_via_proxy(fh_url, timeout=20)
        if raw and len(raw) > 1000:
            print(f"  Got Freedom House data ({len(raw)} chars)")
            print(f"  Note: Excel file - would need openpyxl to parse")
            return {
                "source": fh_url,
                "note": "Excel file available via proxy, needs openpyxl to parse",
                "size_chars": len(raw),
            }
    except Exception as e:
        print(f"  Freedom House fetch failed: {e}")

    print("  Freedom House data not available via simple HTTP fetch")
    return None


def main():
    print("=" * 60)
    print("World Bank Governance Indicators Fetcher")
    print(f"Countries: {', '.join(COUNTRIES)}")
    print(f"Date range: {DATE_RANGE}")
    print(f"Indicators: {len(INDICATORS)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Method: allorigins.win CORS proxy (Cloudflare TLS workaround)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch all governance indicators
    all_records = fetch_all_indicators()

    # Build indicator summaries
    indicator_summaries = {}
    for code, name in INDICATORS.items():
        recs = [r for r in all_records if r["indicator_code"] == code]
        non_null = sum(1 for r in recs if r["value"] is not None)
        indicator_summaries[code] = {
            "name": name,
            "total_records": len(recs),
            "non_null_records": non_null,
        }

    # Reorganize data by country for easier downstream use
    by_country = {}
    for rec in all_records:
        iso2 = rec["country_iso2"]
        if iso2 not in by_country:
            by_country[iso2] = {
                "country_name": rec["country_name"],
                "country_iso2": iso2,
                "country_iso3": rec.get("country_iso3", ""),
                "indicators": {},
            }
        code = rec["indicator_code"]
        if code not in by_country[iso2]["indicators"]:
            by_country[iso2]["indicators"][code] = {
                "name": rec["indicator_name"],
                "values": {},
            }
        if rec["value"] is not None:
            by_country[iso2]["indicators"][code]["values"][str(rec["year"])] = rec["value"]

    # Try Freedom House
    freedom_house = try_freedom_house()

    # Build output
    output = {
        "metadata": {
            "source": "World Bank Worldwide Governance Indicators (WGI)",
            "url": "https://info.worldbank.org/governance/wgi/",
            "fetch_date": datetime.now().isoformat(),
            "countries": COUNTRIES,
            "date_range": DATE_RANGE,
            "indicators": indicator_summaries,
            "total_records": len(all_records),
            "non_null_records": sum(1 for r in all_records if r["value"] is not None),
            "note": "Values range from approximately -2.5 (weak) to 2.5 (strong) governance performance. "
                    "Fetched via allorigins.win proxy due to Cloudflare/OpenSSL 3.6 TLS incompatibility.",
        },
        "by_country": by_country,
        "raw_records": all_records,
    }

    if freedom_house:
        output["freedom_house"] = freedom_house

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(all_records)}")
    print(f"Non-null values: {sum(1 for r in all_records if r['value'] is not None)}")
    print(f"\nPer-indicator:")
    for code, summary in indicator_summaries.items():
        print(f"  {code} ({summary['name']}): {summary['non_null_records']}/{summary['total_records']}")
    print(f"\nPer-country:")
    for iso2, cdata in sorted(by_country.items()):
        total_values = sum(
            len(ind["values"]) for ind in cdata["indicators"].values()
        )
        print(f"  {iso2} ({cdata['country_name']}): {total_values} data points across {len(cdata['indicators'])} indicators")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE):,} bytes")


if __name__ == "__main__":
    main()
