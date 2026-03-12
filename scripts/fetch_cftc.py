#!/usr/bin/env python3
"""Fetch CFTC Commitments of Traders (COT) data from public reporting API and historical zips.

Sources:
  - Legacy futures (Socrata API): https://publicreporting.cftc.gov/resource/jun7-fc8e.csv
  - Historical zip files: https://www.cftc.gov/files/dea/history/deahistfo_{year}.zip

Output: data/economic/cftc/cot_data.json
"""

import csv
import io
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime

OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/cftc"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cot_data.json")

# CFTC contract codes to track
CONTRACTS = {
    "13874A": "E-MINI S&P 500",
    "209742": "NASDAQ-100 (E-MINI)",
    "043602": "10-YEAR T-NOTE",
    "042601": "2-YEAR T-NOTE",
    "067651": "CRUDE OIL",
    "088691": "GOLD",
    "084691": "SILVER",
    "099741": "EURO FX",
    "097741": "JAPANESE YEN",
    "133741": "BITCOIN",
}

# Socrata API endpoint for legacy futures COT
SOCRATA_URL = "https://publicreporting.cftc.gov/resource/jun7-fc8e.csv"
SOCRATA_LIMIT = 50000  # max rows per request

# Historical zip base URL (futures only, legacy format)
HIST_ZIP_URL = "https://www.cftc.gov/files/dea/history/deahistfo{year}.zip"

# Column name mappings — the legacy CSV uses these column headers
# The Socrata API uses slightly different names; we handle both.
COLUMN_MAPS = {
    # Socrata API field names (lowercase, underscores)
    "socrata": {
        "date": "report_date_as_yyyy_mm_dd",
        "contract_code": "cftc_contract_market_code",
        "contract_name": "market_and_exchange_names",
        "noncomm_long": "noncommercial_positions_long_all",
        "noncomm_short": "noncommercial_positions_short_all",
        "noncomm_spread": "noncommercial_positions_spreading_all",  # may not exist in legacy
        "comm_long": "commercial_positions_long_all",
        "comm_short": "commercial_positions_short_all",
        "oi": "open_interest_all",
    },
    # Historical zip CSV column headers (mixed case, spaces)
    "zip": {
        "date": "As of Date in Form YYYY-MM-DD",
        "contract_code": "CFTC Contract Market Code",
        "contract_name": "Market and Exchange Names",
        "noncomm_long": "Noncommercial Positions-Long (All)",
        "noncomm_short": "Noncommercial Positions-Short (All)",
        "noncomm_spread": "Noncommercial Positions-Spreading (All)",
        "comm_long": "Commercial Positions-Long (All)",
        "comm_short": "Commercial Positions-Short (All)",
        "oi": "Open Interest (All)",
    },
}

# Fallback column names (some years use slightly different headers)
ZIP_FALLBACK_COLS = {
    "date": ["As of Date in Form YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD"],
    "contract_code": [
        "CFTC Contract Market Code",
        "CFTC_Contract_Market_Code",
    ],
    "contract_name": [
        "Market and Exchange Names",
        "Market_and_Exchange_Names",
    ],
    "noncomm_long": [
        "Noncommercial Positions-Long (All)",
        "NonComm_Positions_Long_All",
        "Noncommercial Positions-Long (All) ",  # trailing space variant
    ],
    "noncomm_short": [
        "Noncommercial Positions-Short (All)",
        "NonComm_Positions_Short_All",
    ],
    "noncomm_spread": [
        "Noncommercial Positions-Spreading (All)",
        "NonComm_Positions_Spread_All",
    ],
    "comm_long": [
        "Commercial Positions-Long (All)",
        "Comm_Positions_Long_All",
    ],
    "comm_short": [
        "Commercial Positions-Short (All)",
        "Comm_Positions_Short_All",
    ],
    "oi": [
        "Open Interest (All)",
        "Open_Interest_All",
    ],
}


def safe_int(val):
    """Convert value to int, returning None if not possible."""
    if val is None:
        return None
    try:
        return int(str(val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def find_column(headers, field_name, source_type="zip"):
    """Find the actual column header matching a logical field name."""
    # Try direct map first
    col_map = COLUMN_MAPS.get(source_type, {})
    direct = col_map.get(field_name, "")
    if direct in headers:
        return direct

    # Try fallback list
    fallbacks = ZIP_FALLBACK_COLS.get(field_name, [])
    for candidate in fallbacks:
        if candidate in headers:
            return candidate

    # Try case-insensitive + stripped match
    lower_headers = {h.strip().lower().replace(" ", "_").replace("-", "_"): h for h in headers}
    for candidate in fallbacks:
        normalized = candidate.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in lower_headers:
            return lower_headers[normalized]

    return None


def extract_records_from_csv(reader, headers, source_type="zip"):
    """Extract COT records for target contracts from CSV data."""
    # Resolve column names
    col_date = find_column(headers, "date", source_type)
    col_code = find_column(headers, "contract_code", source_type)
    col_name = find_column(headers, "contract_name", source_type)
    col_nl = find_column(headers, "noncomm_long", source_type)
    col_ns = find_column(headers, "noncomm_short", source_type)
    col_nsp = find_column(headers, "noncomm_spread", source_type)
    col_cl = find_column(headers, "comm_long", source_type)
    col_cs = find_column(headers, "comm_short", source_type)
    col_oi = find_column(headers, "oi", source_type)

    if not col_date or not col_code:
        print(f"  WARNING: Could not find date or contract_code columns.")
        print(f"  Available headers (first 10): {headers[:10]}")
        return []

    records = []
    target_codes = set(CONTRACTS.keys())
    # Also match without leading zeros — CFTC sometimes strips them
    target_codes_stripped = {c.lstrip("0"): c for c in CONTRACTS.keys()}

    for row in reader:
        raw_code = row.get(col_code, "").strip()
        # Try exact match first, then stripped match
        code = raw_code if raw_code in target_codes else target_codes_stripped.get(raw_code.lstrip("0"))
        if not code:
            continue

        date_str = row.get(col_date, "").strip()
        if not date_str:
            continue
        # Normalize date to YYYY-MM-DD
        try:
            if "T" in date_str:
                date_str = date_str.split("T")[0]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_out = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        nl = safe_int(row.get(col_nl)) if col_nl else None
        ns = safe_int(row.get(col_ns)) if col_ns else None
        nsp = safe_int(row.get(col_nsp)) if col_nsp else None
        cl = safe_int(row.get(col_cl)) if col_cl else None
        cs = safe_int(row.get(col_cs)) if col_cs else None
        oi_val = safe_int(row.get(col_oi)) if col_oi else None

        net_spec = None
        if nl is not None and ns is not None:
            net_spec = nl - ns

        records.append({
            "date": date_out,
            "contract_code": code,
            "contract_name": CONTRACTS.get(code, row.get(col_name, "").strip()),
            "noncomm_long": nl,
            "noncomm_short": ns,
            "noncomm_spread": nsp,
            "comm_long": cl,
            "comm_short": cs,
            "open_interest": oi_val,
            "net_speculative": net_spec,
        })

    return records


def fetch_url(url, max_retries=3, timeout=60):
    """Fetch URL content with retries."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Python/global-financial-sim)",
                "Accept": "text/csv,application/csv,*/*",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def fetch_from_historical_zips(start_year=2006, end_year=None):
    """Download and parse historical zip files from CFTC."""
    if end_year is None:
        end_year = datetime.now().year

    all_records = []

    for year in range(start_year, end_year + 1):
        url = HIST_ZIP_URL.format(year=year)
        print(f"[ZIP] Fetching {year}... {url}")
        data = fetch_url(url, timeout=90)
        if data is None:
            print(f"  FAILED to download {year}, skipping.")
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_files = [f for f in zf.namelist() if f.lower().endswith((".csv", ".txt"))]
                if not csv_files:
                    print(f"  No CSV found in zip for {year}.")
                    continue

                for csv_name in csv_files:
                    with zf.open(csv_name) as cf:
                        text = cf.read().decode("utf-8", errors="replace")
                        reader = csv.DictReader(io.StringIO(text))
                        headers = reader.fieldnames or []
                        records = extract_records_from_csv(reader, headers, "zip")
                        all_records.extend(records)
                        print(f"  {year}: {len(records)} matching records from {csv_name}")
        except zipfile.BadZipFile:
            print(f"  Bad zip file for {year}, skipping.")
        except Exception as e:
            print(f"  Error processing {year}: {e}")

    return all_records


def fetch_from_socrata():
    """Fetch data from CFTC Socrata API (legacy futures)."""
    all_records = []
    offset = 0

    print(f"\n[SOCRATA] Fetching from {SOCRATA_URL}")

    # Build filter for target contract codes
    codes_filter = " OR ".join(
        f"cftc_contract_market_code='{code}'" for code in CONTRACTS.keys()
    )
    where_clause = f"({codes_filter})"

    while True:
        params = (
            f"$where={urllib.request.quote(where_clause)}"
            f"&$limit={SOCRATA_LIMIT}"
            f"&$offset={offset}"
            f"&$order={urllib.request.quote('report_date_as_yyyy_mm_dd DESC')}"
        )
        url = f"{SOCRATA_URL}?{params}"
        print(f"  Fetching offset={offset}...")

        data = fetch_url(url, timeout=120)
        if data is None:
            print(f"  Failed at offset={offset}, stopping Socrata fetch.")
            break

        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        records = extract_records_from_csv(reader, headers, "socrata")

        if not records:
            # Check if we got any rows at all
            text_lines = text.strip().split("\n")
            if len(text_lines) <= 1:
                print(f"  No more data at offset={offset}.")
                break
            else:
                print(f"  Got {len(text_lines) - 1} rows but 0 matching contracts at offset={offset}.")
                # Still might have data for other contracts, continue
                offset += SOCRATA_LIMIT
                if len(text_lines) - 1 < SOCRATA_LIMIT:
                    break
                continue

        all_records.extend(records)
        print(f"  Got {len(records)} matching records (total: {len(all_records)})")

        offset += SOCRATA_LIMIT
        # If we got fewer than limit, we're done
        text_lines = text.strip().split("\n")
        if len(text_lines) - 1 < SOCRATA_LIMIT:
            break

        time.sleep(0.5)  # Be polite to the API

    return all_records


def merge_and_deduplicate(records):
    """Merge records, deduplicate by (date, contract_code), prefer most complete."""
    seen = {}
    for rec in records:
        key = (rec["date"], rec["contract_code"])
        if key not in seen:
            seen[key] = rec
        else:
            # Keep the record with more non-None fields
            existing = seen[key]
            existing_count = sum(1 for v in existing.values() if v is not None)
            new_count = sum(1 for v in rec.values() if v is not None)
            if new_count > existing_count:
                seen[key] = rec

    return list(seen.values())


def organize_output(records):
    """Organize records by contract into the final output structure."""
    by_contract = {}
    for rec in records:
        code = rec["contract_code"]
        if code not in by_contract:
            by_contract[code] = {
                "contract_code": code,
                "contract_name": CONTRACTS.get(code, rec.get("contract_name", "UNKNOWN")),
                "data": [],
            }
        by_contract[code]["data"].append({
            "date": rec["date"],
            "noncomm_long": rec["noncomm_long"],
            "noncomm_short": rec["noncomm_short"],
            "noncomm_spread": rec["noncomm_spread"],
            "comm_long": rec["comm_long"],
            "comm_short": rec["comm_short"],
            "open_interest": rec["open_interest"],
            "net_speculative": rec["net_speculative"],
        })

    # Sort each contract's data by date
    for code in by_contract:
        by_contract[code]["data"].sort(key=lambda x: x["date"])

    return by_contract


def print_summary(output):
    """Print a summary of fetched data."""
    print("\n" + "=" * 70)
    print("CFTC COT Data Summary")
    print("=" * 70)
    total = 0
    for code in sorted(output.keys()):
        contract = output[code]
        data = contract["data"]
        count = len(data)
        total += count
        if count > 0:
            date_range = f"{data[0]['date']} to {data[-1]['date']}"
        else:
            date_range = "NO DATA"
        print(f"  {contract['contract_name']:25s} ({code}): {count:5d} weeks  [{date_range}]")
    print(f"\n  Total records: {total}")
    print("=" * 70)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []

    # Strategy 1: Historical zip files (2006+, most reliable for bulk data)
    print("=" * 70)
    print("Phase 1: Fetching historical zip files (2006-present)")
    print("=" * 70)
    zip_records = fetch_from_historical_zips(start_year=2006)
    print(f"\nZip files yielded {len(zip_records)} records.")
    all_records.extend(zip_records)

    # Strategy 2: Socrata API (may have additional/newer data)
    print("\n" + "=" * 70)
    print("Phase 2: Fetching from Socrata API (supplementary)")
    print("=" * 70)
    socrata_records = fetch_from_socrata()
    print(f"\nSocrata API yielded {len(socrata_records)} records.")
    all_records.extend(socrata_records)

    if not all_records:
        print("\nERROR: No data fetched from any source. Check network connectivity.")
        sys.exit(1)

    # Merge and deduplicate
    print(f"\nTotal raw records: {len(all_records)}")
    merged = merge_and_deduplicate(all_records)
    print(f"After deduplication: {len(merged)}")

    # Organize by contract
    output = organize_output(merged)

    # Build final JSON
    result = {
        "metadata": {
            "source": "CFTC Commitments of Traders (COT) - Legacy Futures",
            "description": "Weekly speculative and commercial positioning in key futures contracts",
            "fetch_date": datetime.now().strftime("%Y-%m-%d"),
            "contracts_tracked": len(output),
            "total_records": len(merged),
            "fields": {
                "noncomm_long": "Non-commercial (speculative) long positions",
                "noncomm_short": "Non-commercial (speculative) short positions",
                "noncomm_spread": "Non-commercial spread positions",
                "comm_long": "Commercial (hedger) long positions",
                "comm_short": "Commercial (hedger) short positions",
                "open_interest": "Total open interest",
                "net_speculative": "Net speculative position (noncomm_long - noncomm_short)",
            },
        },
        "contracts": output,
    }

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nOutput written to: {OUTPUT_FILE}")
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    print_summary(output)


if __name__ == "__main__":
    main()
