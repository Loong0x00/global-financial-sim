#!/usr/bin/env python3
"""Fetch supplementary economic data to fill known gaps in the FRED dataset.

Targets:
  1. UK Bank Rate (2013+) from Bank of England
  2. Russia key series from CBR / IMF WEO
  3. China supplementary series from FRED
  4. Additional US stress/sentiment indicators from FRED

Output: /home/user/global-financial-sim/data/economic/supplementary/
"""

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "/home/user/global-financial-sim/data/economic/supplementary"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
START_DATE = "1971-01-01"
END_DATE = "2026-03-12"

# Relaxed SSL context for sites with certificate issues (e.g., CBR)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


def _url_get(url: str, timeout: int = 45, allow_insecure: bool = False) -> str:
    """Fetch a URL and return the response body as text."""
    ctx = _ssl_ctx if allow_insecure else None
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) GlobalFinSim/1.0",
        "Accept": "text/csv,text/plain,application/json,*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
        # Try utf-8 first, fall back to latin-1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_fred_csv(csv_text: str) -> list[dict]:
    """Parse a FRED CSV (DATE,VALUE) into [{"date": "YYYY-MM-DD", "value": float}]."""
    entries = []
    for line in csv_text.strip().splitlines()[1:]:  # skip header
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        val_str = parts[1].strip()
        if val_str == "." or val_str == "":
            continue
        try:
            value = float(val_str)
        except ValueError:
            continue
        entries.append({"date": date_str, "value": value})
    return entries


def fetch_fred_series(series_id: str, start: str = START_DATE, end: str = END_DATE,
                      monthly_agg: bool = False) -> list[dict] | None:
    """Fetch a single FRED series via CSV endpoint. Returns list of {date, value}."""
    params = f"id={series_id}&cosd={start}&coed={end}"
    if monthly_agg:
        params += "&fq=Monthly&fas=avg"
    url = f"{FRED_CSV_URL}?{params}"

    for attempt in range(3):
        try:
            csv_text = _url_get(url)
            if not csv_text.strip() or csv_text.strip().startswith("<!"):
                raise RuntimeError("empty or HTML response")
            entries = parse_fred_csv(csv_text)
            if not entries:
                raise RuntimeError("parsed 0 rows")
            return entries
        except Exception as e:
            print(f"    retry {attempt+1}/3 FRED {series_id}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def to_monthly(entries: list[dict]) -> list[dict]:
    """Aggregate daily/weekly entries to monthly averages keyed by YYYY-MM-DD (1st of month)."""
    buckets: dict[str, list[float]] = {}
    for e in entries:
        key = e["date"][:7]  # YYYY-MM
        buckets.setdefault(key, []).append(e["value"])
    result = []
    for ym in sorted(buckets):
        vals = buckets[ym]
        avg = round(sum(vals) / len(vals), 4)
        result.append({"date": f"{ym}-01", "value": avg})
    return result


def print_progress(label: str, data: list[dict] | None):
    if data and len(data) > 0:
        first = data[0]["date"]
        last = data[-1]["date"]
        print(f"  OK   {label:50s}  {len(data):5d} pts  {first} .. {last}", flush=True)
    else:
        print(f"  FAIL {label:50s}  no data", flush=True)


def build_series_block(source: str, series_id: str, data: list[dict]) -> dict:
    return {
        "source": source,
        "series_id": series_id,
        "data": data,
    }


def save_json(filename: str, payload: dict):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  -> Wrote {path}", flush=True)


# ── 1. UK Bank Rate from Bank of England ──────────────────────────────────────

def fetch_boe_bank_rate() -> dict:
    """Fetch UK Bank Rate from Bank of England CSV endpoint (IUDBEDR series)."""
    print("\n=== 1. UK Bank Rate (Bank of England) ===", flush=True)

    # BoE CSV endpoint for Official Bank Rate (IUDBEDR)
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
        "?csv.x=yes"
        "&Datefrom=01/Jan/2013"
        "&Dateto=01/Mar/2026"
        "&SeriesCodes=IUDBEDR"
        "&CSVF=TN"
        "&UsingCodes=Y"
    )

    result = {}
    data = None
    try:
        csv_text = _url_get(url, timeout=30)
        # BoE CSV format: first line header (DATE, IUDBEDR), then data
        entries = []
        for line in csv_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            date_str = parts[0].strip().strip('"')
            val_str = parts[1].strip().strip('"')
            # Skip header
            if "DATE" in date_str.upper() or "IUDBEDR" in val_str.upper():
                continue
            try:
                value = float(val_str)
            except ValueError:
                continue
            # BoE dates are DD/Mon/YYYY or DD Mon YYYY — normalize to YYYY-MM-DD
            normalized = _parse_boe_date(date_str)
            if normalized:
                entries.append({"date": normalized, "value": value})

        if entries:
            entries.sort(key=lambda e: e["date"])
            data = entries
    except Exception as e:
        print(f"    BoE CSV failed: {e}", flush=True)

    print_progress("UK Bank Rate (IUDBEDR) - BoE direct", data)

    if data:
        result["uk_bank_rate"] = build_series_block("Bank of England", "IUDBEDR", data)

    # Also try FRED IUDSOIA (BoE alternate) as fallback
    print("  Trying FRED fallback series...", flush=True)
    for sid, label in [
        ("BOGZ1FL072052006Q", "UK Bank Rate proxy (FRED)"),
        ("IR3TIB01GBM156N", "UK 3-Month Interbank Rate"),
        ("IRSTCI01GBM156N", "UK Short-Term Interest Rate"),
    ]:
        time.sleep(1)
        fallback = fetch_fred_series(sid, start="2013-01-01")
        print_progress(f"{label} ({sid})", fallback)
        if fallback:
            result[f"uk_rate_fallback_{sid}"] = build_series_block("FRED", sid, fallback)

    return result


def _parse_boe_date(s: str) -> str | None:
    """Parse BoE date formats to YYYY-MM-DD."""
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        "january": "01", "february": "02", "march": "03", "april": "04",
        "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    s = s.strip()
    # Try DD Mon YYYY or DD/Mon/YYYY
    for sep in [" ", "/"]:
        parts = s.split(sep)
        if len(parts) == 3:
            day, mon, year = parts[0].strip(), parts[1].strip().lower().rstrip("."), parts[2].strip()
            if mon in months and year.isdigit() and day.isdigit():
                return f"{year}-{months[mon]}-{int(day):02d}"
    # Try ISO
    if len(s) == 10 and s[4] == "-":
        return s
    return None


# ── 2. Russia data from CBR / FRED / IMF ─────────────────────────────────────

def fetch_russia_supplementary() -> dict:
    """Fetch Russia economic data from CBR XML API and FRED alternatives."""
    print("\n=== 2. Russia Supplementary ===", flush=True)
    result = {}

    # --- CBR Key Rate (from CBR XML API) ---
    print("  Fetching CBR Key Rate...", flush=True)
    cbr_data = _fetch_cbr_key_rate()
    print_progress("RU CBR Key Rate (cbr.ru)", cbr_data)
    if cbr_data:
        result["ru_cbr_key_rate"] = build_series_block("CBR", "key_rate", cbr_data)

    # --- FRED series that may still have some Russia data ---
    ru_fred = [
        ("INTDSRRUM193N", "RU Discount Rate (FRED)", False),
        ("CCUSMA02RUM618N", "RU CPI Monthly (FRED)", False),
        ("LRUN64TTRUM156S", "RU Unemployment Rate (FRED)", False),
        ("MANMM101RUM189S", "RU M1 Money Supply (FRED)", False),
        ("MABMM301RUM189S", "RU M3 Money Supply (FRED)", False),
        ("IRSTCB01RUM156N", "RU Short-Term Rate (FRED)", False),
        ("RUSPOPDPA", "RU Population (FRED annual proxy)", False),
        ("NGDPD_RUS", "RU Nominal GDP (IMF-style via FRED)", False),
    ]

    for sid, label, monthly in ru_fred:
        time.sleep(1)
        data = fetch_fred_series(sid, monthly_agg=monthly)
        print_progress(f"{label} ({sid})", data)
        if data:
            result[f"ru_{sid}"] = build_series_block("FRED", sid, data)

    return result


def _fetch_cbr_key_rate() -> list[dict] | None:
    """Fetch CBR key rate from cbr.ru XML API."""
    # CBR provides key rate history at this endpoint
    url = "https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/132988?Posted=True&FromDate=01%2F01%2F2013&ToDate=01%2F03%2F2026"
    # Try the simpler CSV-like endpoint
    url2 = "https://www.cbr.ru/eng/key-indicators/"

    # Attempt: use the CBR XML daily rates API
    try:
        # CBR has an XML endpoint for key rate
        xml_url = "https://www.cbr.ru/scripts/xml_keyrate.asp?d1=01/01/2013&d2=01/03/2026"
        text = _url_get(xml_url, timeout=30, allow_insecure=True)
        entries = _parse_cbr_xml_keyrate(text)
        if entries:
            return entries
    except Exception as e:
        print(f"    CBR XML keyrate failed: {e}", flush=True)

    # Fallback: CBR key rate from a different endpoint
    try:
        url3 = "https://www.cbr.ru/scripts/xml_key_rate.asp?d1=01/01/2013&d2=01/03/2026"
        text = _url_get(url3, timeout=30, allow_insecure=True)
        entries = _parse_cbr_xml_keyrate(text)
        if entries:
            return entries
    except Exception as e:
        print(f"    CBR XML key_rate fallback failed: {e}", flush=True)

    return None


def _parse_cbr_xml_keyrate(xml_text: str) -> list[dict] | None:
    """Parse CBR XML key rate response. Format:
    <KeyRate>
      <KR>
        <DT>2013-09-16T00:00:00</DT>
        <Rate>5.50</Rate>
      </KR>
      ...
    </KeyRate>
    """
    import xml.etree.ElementTree as ET
    entries = []
    try:
        root = ET.fromstring(xml_text)
        for kr in root.iter():
            dt_el = kr.find("DT")
            rate_el = kr.find("Rate")
            if dt_el is not None and rate_el is not None:
                dt_str = dt_el.text.strip()
                rate_str = rate_el.text.strip().replace(",", ".")
                date = dt_str[:10]  # YYYY-MM-DD
                try:
                    value = float(rate_str)
                    entries.append({"date": date, "value": value})
                except ValueError:
                    continue
    except ET.ParseError as e:
        print(f"    XML parse error: {e}", flush=True)
        # Try a simple regex fallback
        import re
        for m in re.finditer(r"<DT>(\d{4}-\d{2}-\d{2})[^<]*</DT>\s*<Rate>([\d.,]+)</Rate>", xml_text):
            date = m.group(1)
            try:
                value = float(m.group(2).replace(",", "."))
                entries.append({"date": date, "value": value})
            except ValueError:
                continue

    if entries:
        entries.sort(key=lambda e: e["date"])
        return entries
    return None


# ── 3. China Supplementary from FRED ─────────────────────────────────────────

def fetch_china_supplementary() -> dict:
    """Fetch additional China series from FRED."""
    print("\n=== 3. China Supplementary (FRED) ===", flush=True)
    result = {}

    cn_series = [
        ("MYAGM2CNM052N",    "CN M2 Money Supply",               False),
        ("CHNPROINDMISMEI",  "CN Industrial Production",          False),
        ("LRUN64TTCNM156S",  "CN Unemployment Rate",              False),
        ("CHNCPIALLMINMEI",  "CN CPI All Items",                  False),
        ("XTEXVA01CNM667S",  "CN Exports Value",                  False),
        ("XTIMVA01CNM667S",  "CN Imports Value",                  False),
        ("QCNR628BIS",       "CN Real Effective Exchange Rate",   False),
        ("CCUSMA02CNM618N",  "CN CPI Monthly Change",             False),
        ("MANMM101CNM189S",  "CN M1 Money Supply (Narrow)",       False),
        ("INTDSRCNM193N",    "CN Discount Rate",                  False),
        ("IRSTCI01CNM156N",  "CN Short-Term Interest Rate",       False),
        ("CHNPROINDMISMEI",  "CN Production of Total Industry",   False),
    ]

    seen = set()
    for sid, label, monthly in cn_series:
        if sid in seen:
            continue
        seen.add(sid)
        time.sleep(1)
        data = fetch_fred_series(sid, monthly_agg=monthly)
        print_progress(f"{label} ({sid})", data)
        if data:
            result[f"cn_{sid}"] = build_series_block("FRED", sid, data)

    return result


# ── 4. Additional US Stress/Sentiment Indicators from FRED ────────────────────

def fetch_us_additional() -> dict:
    """Fetch additional US financial stress and sentiment indicators."""
    print("\n=== 4. US Additional Indicators (FRED) ===", flush=True)
    result = {}

    us_series = [
        # Credit stress
        ("BAMLH0A0HYM2",  "US High Yield Spread (BofA)",      True),
        # Yield curve
        ("T10Y2Y",        "US 10Y-2Y Treasury Spread",        True),
        # Banking stress
        ("TEDRATE",       "TED Spread",                        True),
        # Dollar index
        ("DTWEXBGS",      "Trade Weighted Dollar Index",       True),
        # Home prices
        ("USSTHPI",       "US Home Price Index (FHFA)",        False),
        ("CSUSHPISA",     "Case-Shiller National HPI",         False),
        # Consumer sentiment
        ("UMCSENT",       "U Michigan Consumer Sentiment",     False),
        # Treasury yields (daily -> monthly)
        ("DGS10",         "10Y Treasury Yield (daily)",        True),
        ("DGS2",          "2Y Treasury Yield (daily)",         True),
        # Additional stress indicators
        ("STLFSI2",       "STL Financial Stress Index",        False),
        ("NFCI",          "Chicago Fed Financial Conditions",  False),
        ("ANFCI",         "Adjusted NFCI",                     False),
        # Breakeven inflation
        ("T10YIE",        "10Y Breakeven Inflation Rate",      True),
        ("T5YIE",         "5Y Breakeven Inflation Rate",       True),
        # Real rates
        ("DFII10",        "10Y TIPS Yield (Real Rate)",        True),
        # Credit
        ("BAMLC0A4CBBB",  "BBB Corporate Bond Spread",        True),
        ("BAMLC0A0CM",    "Investment Grade Spread",           True),
    ]

    seen = set()
    for sid, label, daily_to_monthly in us_series:
        if sid in seen:
            continue
        seen.add(sid)
        time.sleep(1)
        data = fetch_fred_series(sid, monthly_agg=daily_to_monthly)
        if data and daily_to_monthly:
            # Data already aggregated monthly by FRED via fq=Monthly&fas=avg
            pass
        print_progress(f"{label} ({sid})", data)
        if data:
            result[f"us_{sid}"] = build_series_block("FRED", sid, data)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Supplementary Data Fetcher — Filling Known Gaps")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track overall stats
    total_series = 0
    total_ok = 0

    # 1. UK Bank Rate
    uk_data = fetch_boe_bank_rate()
    total_series += 4  # 1 BoE + 3 FRED fallbacks attempted
    total_ok += len(uk_data)
    if uk_data:
        save_json("uk_bank_rate.json", uk_data)

    # 2. Russia
    ru_data = fetch_russia_supplementary()
    total_series += 9  # 1 CBR + 8 FRED
    total_ok += len(ru_data)
    if ru_data:
        save_json("russia_supplementary.json", ru_data)

    # 3. China
    cn_data = fetch_china_supplementary()
    total_series += 11  # deduplicated count
    total_ok += len(cn_data)
    if cn_data:
        save_json("china_supplementary.json", cn_data)

    # 4. US additional
    us_data = fetch_us_additional()
    total_series += 17  # deduplicated count
    total_ok += len(us_data)
    if us_data:
        save_json("us_additional.json", us_data)

    # Summary
    print("\n" + "=" * 70)
    print(f"  DONE — {total_ok} series fetched successfully out of ~{total_series} attempted")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print("=" * 70)

    # Write a manifest
    manifest = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Fill known gaps in FRED economic dataset",
        "files": {},
    }
    for fname, data in [
        ("uk_bank_rate.json", uk_data),
        ("russia_supplementary.json", ru_data),
        ("china_supplementary.json", cn_data),
        ("us_additional.json", us_data),
    ]:
        if data:
            series_summary = {}
            for key, block in data.items():
                d = block.get("data", [])
                series_summary[key] = {
                    "source": block.get("source", "unknown"),
                    "series_id": block.get("series_id", "unknown"),
                    "points": len(d),
                    "range": f"{d[0]['date']} .. {d[-1]['date']}" if d else "empty",
                }
            manifest["files"][fname] = series_summary

    save_json("manifest.json", manifest)


if __name__ == "__main__":
    main()
