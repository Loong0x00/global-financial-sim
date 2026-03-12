#!/usr/bin/env python3
"""Fetch BIS (Bank for International Settlements) statistical data.

Downloads bulk CSV ZIP files from BIS, parses them, and outputs JSON time series
per country for use in the global financial simulation engine.

BIS CSV flat file format:
- Headers are composite: "REF_AREA:Reference area", "TIME_PERIOD:Time period"
- Cell values include code + label: "US: United States", "Q: Quarterly"
- We extract the code prefix before the colon for matching.

Datasets (priority order):
1. Locational banking statistics (cross-border positions / capital flows)
2. Total credit to non-financial sector (leverage levels)
3. Selected property prices (housing bubble indicator)
4. Effective exchange rates (trade-weighted FX)
5. Debt securities statistics
6. Credit-to-GDP gaps (early warning indicator)
7. Debt service ratios
8. Central bank policy rates
9. Global liquidity indicators
10. Consolidated banking statistics

Target countries: US, CN, JP, DE, GB, FR, RU, KR, BR, IN
Output: /home/user/global-financial-sim/data/economic/bis/
"""

import csv
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
import zipfile
from collections import defaultdict
import datetime as _dt
from datetime import datetime, timezone

OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/bis"
CACHE_DIR = os.path.join(OUTPUT_DIR, "_cache")

# BIS uses ISO 2-letter country codes in cells like "US: United States"
TARGET_CODES = {"US", "CN", "JP", "DE", "GB", "FR", "RU", "KR", "BR", "IN"}

DATASETS = [
    {
        "name": "locational_banking",
        "label": "Locational Banking Statistics (cross-border positions)",
        "url": "https://data.bis.org/static/bulk/WS_LBS_D_PUB_csv_flat.zip",
        "description": "Cross-border credit flows between countries - shows capital flows",
    },
    {
        "name": "total_credit",
        "label": "Credit to the Non-Financial Sector",
        "url": "https://data.bis.org/static/bulk/WS_TC_csv_flat.zip",
        "description": "Credit/GDP ratios and total credit - shows leverage levels",
    },
    {
        "name": "property_prices",
        "label": "Selected Residential Property Prices",
        "url": "https://data.bis.org/static/bulk/WS_SPP_csv_flat.zip",
        "description": "Residential property price indices - housing bubble indicator",
    },
    {
        "name": "effective_exchange_rates",
        "label": "Effective Exchange Rates",
        "url": "https://data.bis.org/static/bulk/WS_EER_csv_flat.zip",
        "description": "Real and nominal effective exchange rates - trade-weighted FX",
    },
    {
        "name": "debt_securities",
        "label": "International Debt Securities",
        "url": "https://data.bis.org/static/bulk/WS_DEBT_SEC2_PUB_csv_flat.zip",
        "description": "International debt securities outstanding by country",
    },
    {
        "name": "credit_gap",
        "label": "Credit-to-GDP Gaps",
        "url": "https://data.bis.org/static/bulk/WS_CREDIT_GAP_csv_flat.zip",
        "description": "Credit-to-GDP gap - BIS early warning indicator for banking crises",
    },
    {
        "name": "debt_service_ratios",
        "label": "Debt Service Ratios",
        "url": "https://data.bis.org/static/bulk/WS_DSR_csv_flat.zip",
        "description": "Debt service ratios for the private non-financial sector",
    },
    {
        "name": "policy_rates",
        "label": "Central Bank Policy Rates",
        "url": "https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip",
        "description": "Central bank policy interest rates",
    },
    {
        "name": "global_liquidity",
        "label": "Global Liquidity Indicators",
        "url": "https://data.bis.org/static/bulk/WS_GLI_csv_flat.zip",
        "description": "Global liquidity indicators - USD credit outside US etc.",
    },
    {
        "name": "consolidated_banking",
        "label": "Consolidated Banking Statistics",
        "url": "https://data.bis.org/static/bulk/WS_CBS_PUB_csv_flat.zip",
        "description": "Consolidated cross-border banking claims",
    },
]


def download_zip(url: str, cache_path: str) -> bytes | None:
    """Download a ZIP file from BIS. Uses cache if available and fresh (<24h)."""
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 24:
            print(f"  Using cached file ({age_hours:.1f}h old)", flush=True)
            with open(cache_path, "rb") as f:
                return f.read()
        else:
            print(f"  Cache expired ({age_hours:.1f}h old), re-downloading", flush=True)

    for attempt in range(3):
        try:
            print(f"  Downloading from {url} ...", flush=True)
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (global-financial-sim research project)",
                "Accept": "application/zip, */*",
            })
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            size_mb = len(data) / (1024 * 1024)
            print(f"  Downloaded {size_mb:.1f} MB", flush=True)
            with open(cache_path, "wb") as f:
                f.write(data)
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"  RETRY {attempt + 1}/3: {e}", file=sys.stderr, flush=True)
            time.sleep(3 * (attempt + 1))

    print(f"  ERROR: all download retries failed for {url}", file=sys.stderr, flush=True)
    return None


def extract_code(cell: str) -> str:
    """Extract the code prefix from a BIS cell value.

    BIS cells look like 'US: United States' or 'Q: Quarterly'.
    Returns the code part ('US', 'Q') or the full stripped value if no colon.
    """
    cell = cell.strip()
    if ": " in cell:
        return cell.split(": ", 1)[0].strip()
    return cell


def extract_header_key(header: str) -> str:
    """Extract the key name from a BIS composite header.

    Headers look like 'REF_AREA:Reference area' or 'TIME_PERIOD:Time period or range'.
    Returns 'REF_AREA' or 'TIME_PERIOD'.
    """
    header = header.strip()
    if ":" in header:
        return header.split(":", 1)[0].strip()
    return header


def parse_headers(raw_headers: list[str]) -> dict:
    """Parse BIS CSV headers and identify column roles.

    Returns dict with:
      'country_cols': {header_key: col_index} for country-identifying columns
      'time_idx': column index of TIME_PERIOD
      'value_idx': column index of OBS_VALUE
      'desc_cols': {header_key: col_index} for descriptor columns
      'raw': original headers
      'keys': extracted header keys
    """
    keys = [extract_header_key(h) for h in raw_headers]

    # Find time and value columns
    time_idx = -1
    value_idx = -1
    for i, k in enumerate(keys):
        ku = k.upper()
        if ku == "TIME_PERIOD":
            time_idx = i
        elif ku == "OBS_VALUE":
            value_idx = i

    # Find country columns - BIS uses various naming conventions:
    # REF_AREA (most datasets), BORROWERS_CTY (credit/liquidity),
    # L_REP_CTY / L_CP_COUNTRY / L_PARENT_CTY (locational/consolidated banking),
    # COUNTERPART_AREA (debt securities DSS), ISSUER_RES / ISSUER_NAT (debt sec2)
    country_patterns = {
        "REF_AREA", "BORROWERS_CTY", "L_REF_AREA",
        "COUNTERPARTY_AREA", "COUNTERPART_AREA",
        "L_REP_CTY", "L_CP_COUNTRY", "L_PARENT_CTY",
        "ISSUER_RES", "ISSUER_NAT",
    }
    country_cols = {}
    for i, k in enumerate(keys):
        if k.upper() in country_patterns:
            country_cols[k] = i

    # Find descriptor columns (for building series keys)
    skip_keys = {"STRUCTURE", "STRUCTURE_ID", "ACTION", "TIME_PERIOD", "OBS_VALUE",
                 "OBS_STATUS", "OBS_CONF", "DECIMALS", "UNIT_MULT", "TIME_FORMAT",
                 "COMPILATION", "SOURCE_REF", "SUPP_INFO_BREAKS", "TITLE_TS",
                 "BREAKS", "COVERAGE", "COLLECTION", "ORG_VISIBILITY", "AVAILABILITY",
                 "TITLE_GRP", "TITLE", "TITLE_COMPL", "COMMENT_DSET", "COMMENT_TS",
                 "COMMENT_OBS", "LAST_UPDATE", "COMPILING_ORG", "COLL_PERIOD",
                 "CONF_STATUS", "EMBARGO_DATE", "OBS_PRE_BREAK", "OBS_EDP_WBB",
                 "TABLE_IDENTIFIER", "DISS_ORG", "GFS_ECOFUNC", "GFS_TAXCAT",
                 "DATA_COMP", "CURRENCY", "REF_PERIOD_DETAIL", "REPYEARSTART",
                 "REPYEAREND", "REF_YEAR_PRICE", "CUST_BREAKDOWN_LB", "TIME_PER_COLLECT",
                 "FREQ"}
    # Also skip all country columns
    country_keys_upper = {k.upper() for k in country_cols}
    desc_cols = {}
    for i, k in enumerate(keys):
        ku = k.upper()
        if ku not in skip_keys and ku not in country_keys_upper and i != time_idx and i != value_idx:
            desc_cols[k] = i

    return {
        "country_cols": country_cols,
        "time_idx": time_idx,
        "value_idx": value_idx,
        "desc_cols": desc_cols,
        "raw": raw_headers,
        "keys": keys,
    }


def process_csv_streaming(zip_data: bytes, dataset_name: str) -> dict:
    """Parse a BIS CSV ZIP file and extract time series per country.

    Returns: {country_code: {series_key: {time_period: value}}}
    """
    result = defaultdict(lambda: defaultdict(dict))
    row_count = 0
    matched_rows = 0

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data))
    except zipfile.BadZipFile:
        print(f"  ERROR: not a valid ZIP file", file=sys.stderr, flush=True)
        return {}

    csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_files:
        print(f"  ERROR: no CSV files found in ZIP", file=sys.stderr, flush=True)
        return {}

    print(f"  ZIP contains: {csv_files}", flush=True)

    for csv_name in csv_files:
        print(f"  Parsing {csv_name} ...", flush=True)

        with zf.open(csv_name) as f:
            text_wrapper = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace")
            reader = csv.reader(text_wrapper)

            try:
                raw_headers = next(reader)
            except StopIteration:
                print(f"  WARNING: empty CSV file {csv_name}", flush=True)
                continue

            info = parse_headers(raw_headers)
            time_idx = info["time_idx"]
            value_idx = info["value_idx"]
            country_cols = info["country_cols"]
            desc_cols = info["desc_cols"]

            if time_idx < 0 or value_idx < 0:
                print(f"  WARNING: could not find TIME_PERIOD/OBS_VALUE columns", flush=True)
                print(f"  Header keys: {info['keys'][:20]}", flush=True)
                continue

            # Determine primary country column (prefer REF_AREA > reporting > borrowers > counterparty)
            primary_col = None
            for pref in ["REF_AREA", "L_REP_CTY", "BORROWERS_CTY", "ISSUER_RES",
                          "L_REF_AREA", "L_CP_COUNTRY", "COUNTERPART_AREA",
                          "COUNTERPARTY_AREA", "ISSUER_NAT"]:
                if pref in country_cols:
                    primary_col = pref
                    break
            if primary_col is None and country_cols:
                primary_col = next(iter(country_cols))

            print(f"  Time col: [{time_idx}] {info['keys'][time_idx]}", flush=True)
            print(f"  Value col: [{value_idx}] {info['keys'][value_idx]}", flush=True)
            print(f"  Country cols: {list(country_cols.keys())} (primary: {primary_col})", flush=True)
            print(f"  Descriptor cols ({len(desc_cols)}): {list(desc_cols.keys())[:10]}", flush=True)

            max_col = max(time_idx, value_idx)

            for row in reader:
                row_count += 1
                if row_count % 1000000 == 0:
                    print(f"    ... {row_count:,} rows, {matched_rows:,} matched", flush=True)

                if len(row) <= max_col:
                    continue

                # Extract country code from primary column
                country = None
                if primary_col is not None:
                    cidx = country_cols[primary_col]
                    if cidx < len(row):
                        code = extract_code(row[cidx])
                        if code in TARGET_CODES:
                            country = code

                # Fallback: try other country columns
                if country is None:
                    for col_name, cidx in country_cols.items():
                        if cidx < len(row):
                            code = extract_code(row[cidx])
                            if code in TARGET_CODES:
                                country = code
                                break

                if country is None:
                    continue

                # Parse time period (extract code before label)
                tp_raw = row[time_idx].strip()
                # Some cells might not have the "code: label" format for time
                # Time periods are typically just "2008-Q1" or "2020-01"
                tp = tp_raw
                if tp_raw and tp_raw[0].isdigit():
                    # Already a plain date/period
                    pass
                else:
                    # Might be empty or have other format
                    if not tp_raw:
                        continue

                # Parse value
                val_str = row[value_idx].strip()
                if not val_str or val_str.upper() in ("", "NA", "N/A", "NAN", ".", ".."):
                    continue
                try:
                    value = float(val_str)
                except ValueError:
                    continue

                # Build series key from descriptor columns
                parts = []
                for col_name, cidx in desc_cols.items():
                    if cidx < len(row):
                        cell = row[cidx].strip()
                        if cell and len(cell) < 100:
                            parts.append(cell)
                        elif cell:
                            parts.append(extract_code(cell))
                series_key = " | ".join(parts) if parts else "value"

                result[country][series_key][tp] = value
                matched_rows += 1

    print(f"  Total rows: {row_count:,}, matched: {matched_rows:,}", flush=True)
    return dict(result)


def deduplicate_series(country_data: dict, max_series: int = 300) -> dict:
    """Keep the most informative series per country."""
    if len(country_data) <= max_series:
        return country_data

    scored = []
    for key, ts in country_data.items():
        n = len(ts)
        if n == 0:
            continue
        scored.append((n, key, ts))

    scored.sort(key=lambda x: -x[0])

    kept = {}
    for n_points, key, ts in scored:
        if len(kept) >= max_series and n_points < 10:
            break
        kept[key] = ts

    return kept


def format_output(raw_data: dict, dataset_name: str, ds_info: dict) -> dict:
    """Format parsed data into the output JSON structure."""
    output = {
        "source": "BIS",
        "dataset": dataset_name,
        "label": ds_info["label"],
        "description": ds_info["description"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "countries": {},
    }

    for cc in sorted(raw_data.keys()):
        series = deduplicate_series(raw_data[cc])
        formatted = {}
        for skey, ts in series.items():
            sorted_ts = dict(sorted(ts.items()))
            formatted[skey] = {
                "n_observations": len(sorted_ts),
                "time_range": [min(sorted_ts.keys()), max(sorted_ts.keys())] if sorted_ts else [],
                "data": sorted_ts,
            }

        output["countries"][cc] = {
            "n_series": len(formatted),
            "series": formatted,
        }

    return output


def print_summary(output: dict, dataset_name: str):
    """Print a summary of fetched data."""
    print(f"\n  === {dataset_name} Summary ===", flush=True)
    countries = output.get("countries", {})
    if not countries:
        print("  No data for target countries.", flush=True)
        return

    total_series = 0
    total_obs = 0
    for cc in sorted(countries.keys()):
        cdata = countries[cc]
        n_ser = cdata["n_series"]
        n_obs = sum(s["n_observations"] for s in cdata["series"].values())
        total_series += n_ser
        total_obs += n_obs

        all_starts = [s["time_range"][0] for s in cdata["series"].values() if s["time_range"]]
        all_ends = [s["time_range"][1] for s in cdata["series"].values() if s["time_range"]]
        time_str = f"  {min(all_starts)} to {max(all_ends)}" if all_starts else ""
        print(f"  {cc}: {n_ser:>4} series, {n_obs:>8,} obs{time_str}", flush=True)

    print(f"  TOTAL: {total_series} series, {total_obs:,} obs, {len(countries)} countries", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("BIS Statistical Data Fetcher", flush=True)
    print(f"Target countries: {', '.join(sorted(TARGET_CODES))}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(f"Datasets to fetch: {len(DATASETS)}", flush=True)
    print("=" * 70, flush=True)

    results_summary = {}

    for i, ds in enumerate(DATASETS):
        name = ds["name"]
        print(f"\n[{i + 1}/{len(DATASETS)}] {ds['label']}", flush=True)
        print(f"  {ds['description']}", flush=True)

        cache_path = os.path.join(CACHE_DIR, f"{name}.zip")
        zip_data = download_zip(ds["url"], cache_path)
        if zip_data is None:
            print(f"  SKIPPED: download failed", flush=True)
            results_summary[name] = "FAILED (download)"
            continue

        try:
            raw_data = process_csv_streaming(zip_data, name)
        except Exception as e:
            print(f"  ERROR parsing: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            results_summary[name] = f"FAILED (parse: {e})"
            continue

        if not raw_data:
            print(f"  WARNING: no data matched target countries", flush=True)
            results_summary[name] = "NO DATA"
            continue

        output = format_output(raw_data, name, ds)
        print_summary(output, name)

        out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=1)
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Written: {out_path} ({file_size_mb:.1f} MB)", flush=True)

        n_countries = len(output.get("countries", {}))
        results_summary[name] = f"OK ({n_countries} countries)"

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for name, status in results_summary.items():
        label = next((d["label"] for d in DATASETS if d["name"] == name), name)
        print(f"  {label}: {status}", flush=True)

    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "target_countries": sorted(TARGET_CODES),
        "datasets": results_summary,
    }
    meta_path = os.path.join(OUTPUT_DIR, "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
