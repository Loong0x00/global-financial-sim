#!/usr/bin/env python3
"""
Fetch bilateral trade data for quantifying trade decoupling between major economies.

Target pairs: US-CN, US-EU, US-JP, US-KR, CN-EU, CN-JP, CN-KR, US-RU, EU-RU, CN-RU

Data sources:
1. UN Comtrade public preview API (working, no key needed)
2. Compiled reference data from US Census/USTR/IMF DOTS as fallback

Output: /home/user/global-financial-sim/data/economic/trade/bilateral_trade.json
"""

import json
import time
import urllib.request
import urllib.error
import ssl
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("/home/user/global-financial-sim/data/economic/trade")
OUTPUT_FILE = OUTPUT_DIR / "bilateral_trade.json"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

COMTRADE_CODES = {
    "us": 842, "cn": 156, "de": 276, "jp": 392,
    "gb": 826, "kr": 410, "ru": 643, "fr": 251,
    "it": 381, "nl": 528,
}

EU_MEMBERS = ["de", "fr", "it", "nl", "gb"]

TRADE_PAIRS = [
    ("us", "cn"), ("us", "eu"), ("us", "jp"), ("us", "kr"),
    ("cn", "eu"), ("cn", "jp"), ("cn", "kr"),
    ("us", "ru"), ("eu", "ru"), ("cn", "ru"),
]

YEARS = list(range(2000, 2025))


def fetch_json(url, timeout=20):
    """Fetch and parse JSON from URL."""
    headers = {"User-Agent": "GlobalFinancialSim/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX)
        return json.loads(resp.read())
    except Exception as e:
        return None


def comtrade_fetch(reporter_code, partner_code, years_batch):
    """Fetch bilateral trade from UN Comtrade for a batch of years."""
    periods = ",".join(str(y) for y in years_batch)
    url = (
        f"https://comtradeapi.un.org/public/v1/preview/C/A/HS"
        f"?reporterCode={reporter_code}&partnerCode={partner_code}"
        f"&period={periods}&cmdCode=TOTAL&flowCode=X,M"
    )
    data = fetch_json(url)
    results = {}
    if data and "data" in data and data["data"]:
        for rec in data["data"]:
            yr = int(rec.get("period", 0))
            flow = rec.get("flowCode", "")
            val = rec.get("primaryValue")
            if val is None:
                val = rec.get("fobvalue") or rec.get("cifvalue")
            if yr and val is not None:
                if yr not in results:
                    results[yr] = {}
                if flow == "X":
                    results[yr]["exports"] = float(val)
                elif flow == "M":
                    results[yr]["imports"] = float(val)
    return results


def fetch_comtrade_bilateral(reporter, partner):
    """Fetch full bilateral trade series from Comtrade."""
    rep_code = COMTRADE_CODES.get(reporter)
    par_code = COMTRADE_CODES.get(partner)
    if not rep_code or not par_code:
        return None

    print(f"  [Comtrade] {reporter.upper()}({rep_code}) <-> {partner.upper()}({par_code})")
    results = {}

    # Batch years in groups of 5
    for i in range(0, len(YEARS), 5):
        batch = YEARS[i:i + 5]
        batch_data = comtrade_fetch(rep_code, par_code, batch)
        results.update(batch_data)
        # Quick bail if first batch returns nothing
        if i == 0 and not batch_data:
            print(f"    No data from first batch, trying single year...")
            # Try just one year as sanity check
            single = comtrade_fetch(rep_code, par_code, [2022])
            if not single:
                print(f"    No data available")
                return None
            results.update(single)
        time.sleep(1.2)  # Rate limit

    if results:
        print(f"    Got {len(results)} years")
    return results if results else None


# ---------------------------------------------------------------------------
# Reference data compiled from US Census Bureau, USTR, IMF DOTS, UNCTAD
# Values in USD billions
# ---------------------------------------------------------------------------
KNOWN_TRADE = {
    "us_cn": {
        "source": "US Census Bureau / USTR",
        "data": {
            2000: (16.3, 100.1), 2001: (19.2, 102.3), 2002: (22.1, 125.2),
            2003: (28.4, 152.4), 2004: (34.7, 196.7), 2005: (41.8, 243.5),
            2006: (55.2, 287.8), 2007: (65.2, 321.5), 2008: (71.5, 337.8),
            2009: (69.6, 296.4), 2010: (91.9, 364.9), 2011: (104.1, 399.3),
            2012: (110.6, 425.6), 2013: (121.7, 440.4), 2014: (123.7, 468.5),
            2015: (115.9, 483.2), 2016: (115.8, 462.6), 2017: (129.9, 505.5),
            2018: (120.3, 539.5), 2019: (106.4, 452.2), 2020: (124.5, 435.4),
            2021: (151.1, 506.4), 2022: (153.8, 536.8), 2023: (147.8, 427.2),
            2024: (143.5, 438.9),
        },
    },
    "us_jp": {
        "source": "US Census Bureau",
        "data": {
            2000: (65.3, 146.5), 2001: (57.6, 126.5), 2002: (51.4, 121.4),
            2003: (52.1, 118.0), 2004: (54.2, 129.6), 2005: (55.4, 138.0),
            2006: (59.6, 148.2), 2007: (62.7, 145.5), 2008: (66.6, 139.3),
            2009: (51.2, 96.0), 2010: (60.5, 120.3), 2011: (66.2, 128.8),
            2012: (70.0, 146.4), 2013: (65.2, 138.6), 2014: (67.0, 134.0),
            2015: (62.5, 131.1), 2016: (63.3, 132.2), 2017: (67.6, 136.5),
            2018: (75.0, 142.6), 2019: (74.4, 143.6), 2020: (64.1, 119.5),
            2021: (75.0, 134.5), 2022: (80.3, 148.3), 2023: (79.6, 143.1),
            2024: (78.0, 140.5),
        },
    },
    "us_kr": {
        "source": "US Census Bureau",
        "data": {
            2000: (27.9, 40.3), 2001: (22.2, 35.2), 2002: (22.6, 35.6),
            2003: (24.1, 37.2), 2004: (26.3, 46.2), 2005: (27.6, 43.8),
            2006: (32.5, 45.8), 2007: (34.7, 47.6), 2008: (34.8, 48.1),
            2009: (28.6, 39.2), 2010: (38.8, 48.9), 2011: (43.5, 56.6),
            2012: (42.3, 58.9), 2013: (41.7, 62.2), 2014: (44.5, 69.6),
            2015: (43.5, 71.8), 2016: (42.3, 69.9), 2017: (48.3, 71.2),
            2018: (56.3, 74.3), 2019: (56.5, 77.5), 2020: (51.2, 76.0),
            2021: (65.8, 93.2), 2022: (65.2, 115.0), 2023: (65.6, 114.7),
            2024: (64.5, 112.0),
        },
    },
    "us_ru": {
        "source": "US Census Bureau / IMF DOTS",
        "data": {
            2000: (2.1, 10.0), 2001: (2.7, 6.3), 2002: (2.4, 6.8),
            2003: (2.5, 8.6), 2004: (3.0, 11.9), 2005: (3.9, 15.3),
            2006: (4.7, 19.8), 2007: (7.4, 19.3), 2008: (9.3, 26.7),
            2009: (5.4, 18.2), 2010: (5.9, 25.7), 2011: (8.3, 34.6),
            2012: (10.7, 29.3), 2013: (11.2, 27.2), 2014: (10.8, 23.7),
            2015: (7.1, 16.4), 2016: (5.8, 14.5), 2017: (7.0, 17.4),
            2018: (6.6, 22.0), 2019: (5.8, 22.3), 2020: (4.1, 16.5),
            2021: (6.4, 29.7), 2022: (1.5, 14.5), 2023: (0.8, 4.5),
            2024: (0.5, 3.2),
        },
    },
    "cn_jp": {
        "source": "IMF DOTS / China Customs",
        "data": {
            2000: (41.7, 41.5), 2001: (45.0, 42.8), 2002: (48.5, 53.5),
            2003: (59.4, 74.2), 2004: (73.5, 94.4), 2005: (84.0, 100.5),
            2006: (91.6, 115.7), 2007: (102.1, 134.0), 2008: (116.1, 150.2),
            2009: (97.9, 130.9), 2010: (121.0, 176.7), 2011: (148.3, 194.6),
            2012: (151.6, 177.8), 2013: (150.3, 162.3), 2014: (149.4, 163.3),
            2015: (135.8, 143.3), 2016: (129.3, 145.7), 2017: (137.2, 165.8),
            2018: (147.2, 180.2), 2019: (143.2, 171.8), 2020: (142.6, 175.1),
            2021: (165.8, 205.6), 2022: (172.9, 184.5), 2023: (157.4, 162.0),
            2024: (152.0, 165.8),
        },
    },
    "cn_kr": {
        "source": "IMF DOTS / Korea Customs",
        "data": {
            2000: (11.3, 23.2), 2001: (12.5, 23.4), 2002: (15.5, 28.6),
            2003: (20.1, 43.1), 2004: (27.8, 62.2), 2005: (35.1, 76.8),
            2006: (44.5, 89.8), 2007: (56.1, 104.5), 2008: (73.9, 112.2),
            2009: (53.7, 102.4), 2010: (68.8, 138.4), 2011: (82.9, 162.7),
            2012: (87.7, 168.7), 2013: (91.2, 183.1), 2014: (100.3, 190.1),
            2015: (101.4, 174.5), 2016: (93.7, 158.6), 2017: (102.8, 178.0),
            2018: (109.2, 204.2), 2019: (111.0, 173.6), 2020: (112.5, 172.7),
            2021: (148.9, 213.5), 2022: (162.5, 193.1), 2023: (164.8, 170.5),
            2024: (158.0, 175.0),
        },
    },
    "cn_ru": {
        "source": "IMF DOTS / China Customs",
        "data": {
            2000: (5.8, 5.2), 2001: (6.0, 5.6), 2002: (6.8, 5.9),
            2003: (9.3, 7.3), 2004: (10.1, 12.1), 2005: (13.2, 15.9),
            2006: (15.8, 17.6), 2007: (28.5, 19.7), 2008: (33.0, 23.8),
            2009: (17.5, 21.3), 2010: (29.6, 25.8), 2011: (38.9, 40.3),
            2012: (44.1, 44.1), 2013: (49.6, 39.6), 2014: (53.7, 41.6),
            2015: (34.8, 33.3), 2016: (37.3, 32.2), 2017: (42.9, 41.2),
            2018: (47.9, 59.1), 2019: (49.7, 61.1), 2020: (50.6, 57.2),
            2021: (67.6, 79.3), 2022: (76.1, 114.1), 2023: (110.9, 129.1),
            2024: (115.0, 135.0),
        },
    },
    "us_eu": {
        "source": "IMF DOTS / Eurostat (EU27 approximation)",
        "data": {
            2000: (165.0, 220.0), 2001: (158.0, 212.0), 2002: (143.0, 218.0),
            2003: (151.0, 241.0), 2004: (172.0, 272.0), 2005: (186.0, 302.0),
            2006: (211.0, 325.0), 2007: (247.0, 356.0), 2008: (271.0, 367.0),
            2009: (220.0, 281.0), 2010: (239.0, 319.0), 2011: (269.0, 364.0),
            2012: (265.0, 381.0), 2013: (262.0, 387.0), 2014: (276.0, 418.0),
            2015: (272.0, 427.0), 2016: (270.0, 418.0), 2017: (283.0, 434.0),
            2018: (318.0, 487.0), 2019: (337.0, 515.0), 2020: (272.0, 439.0),
            2021: (364.0, 498.0), 2022: (376.0, 553.0), 2023: (368.0, 559.0),
            2024: (360.0, 545.0),
        },
    },
    "cn_eu": {
        "source": "IMF DOTS / Eurostat",
        "data": {
            2000: (38.2, 35.6), 2001: (41.0, 39.0), 2002: (48.2, 42.2),
            2003: (60.2, 53.0), 2004: (79.0, 64.0), 2005: (103.0, 73.0),
            2006: (130.0, 83.0), 2007: (160.0, 98.0), 2008: (178.0, 110.0),
            2009: (152.0, 98.0), 2010: (197.0, 131.0), 2011: (230.0, 157.0),
            2012: (245.0, 164.0), 2013: (255.0, 170.0), 2014: (278.0, 185.0),
            2015: (298.0, 175.0), 2016: (312.0, 173.0), 2017: (333.0, 195.0),
            2018: (359.0, 215.0), 2019: (362.0, 210.0), 2020: (383.0, 202.0),
            2021: (472.0, 236.0), 2022: (509.0, 240.0), 2023: (480.0, 225.0),
            2024: (465.0, 220.0),
        },
    },
    "eu_ru": {
        "source": "Eurostat / IMF DOTS",
        "data": {
            2000: (22.7, 63.8), 2001: (25.0, 62.0), 2002: (28.0, 58.0),
            2003: (33.0, 66.0), 2004: (42.0, 84.0), 2005: (56.0, 112.0),
            2006: (72.0, 143.0), 2007: (89.0, 146.0), 2008: (105.0, 180.0),
            2009: (65.0, 115.0), 2010: (86.0, 160.0), 2011: (108.0, 199.0),
            2012: (123.0, 213.0), 2013: (120.0, 206.0), 2014: (103.0, 183.0),
            2015: (73.0, 136.0), 2016: (72.0, 118.0), 2017: (86.0, 145.0),
            2018: (85.0, 168.0), 2019: (80.0, 145.0), 2020: (70.0, 95.0),
            2021: (89.0, 158.0), 2022: (50.0, 110.0), 2023: (35.0, 48.0),
            2024: (30.0, 42.0),
        },
    },
}

# GDP data (USD trillions) for trade-to-GDP ratios
GDP_DATA = {
    "us": {
        2000: 10.25, 2001: 10.58, 2002: 10.94, 2003: 11.46, 2004: 12.21,
        2005: 13.04, 2006: 13.81, 2007: 14.45, 2008: 14.71, 2009: 14.42,
        2010: 14.99, 2011: 15.54, 2012: 16.20, 2013: 16.78, 2014: 17.52,
        2015: 18.22, 2016: 18.71, 2017: 19.54, 2018: 20.53, 2019: 21.37,
        2020: 20.89, 2021: 23.32, 2022: 25.46, 2023: 27.36, 2024: 28.78,
    },
    "cn": {
        2000: 1.21, 2001: 1.34, 2002: 1.47, 2003: 1.66, 2004: 1.96,
        2005: 2.29, 2006: 2.75, 2007: 3.55, 2008: 4.59, 2009: 5.10,
        2010: 6.09, 2011: 7.55, 2012: 8.53, 2013: 9.57, 2014: 10.48,
        2015: 11.06, 2016: 11.23, 2017: 12.31, 2018: 13.89, 2019: 14.28,
        2020: 14.72, 2021: 17.73, 2022: 17.96, 2023: 17.79, 2024: 18.53,
    },
    "jp": {
        2000: 4.89, 2001: 4.30, 2002: 4.12, 2003: 4.39, 2004: 4.66,
        2005: 4.76, 2006: 4.53, 2007: 4.52, 2008: 5.04, 2009: 5.23,
        2010: 5.76, 2011: 6.16, 2012: 6.27, 2013: 5.16, 2014: 4.85,
        2015: 4.39, 2016: 4.92, 2017: 4.87, 2018: 5.04, 2019: 5.12,
        2020: 4.90, 2021: 5.01, 2022: 4.23, 2023: 4.21, 2024: 4.07,
    },
    "kr": {
        2000: 0.58, 2001: 0.53, 2002: 0.61, 2003: 0.68, 2004: 0.76,
        2005: 0.90, 2006: 1.01, 2007: 1.12, 2008: 1.00, 2009: 0.90,
        2010: 1.09, 2011: 1.20, 2012: 1.22, 2013: 1.31, 2014: 1.41,
        2015: 1.38, 2016: 1.42, 2017: 1.53, 2018: 1.72, 2019: 1.65,
        2020: 1.64, 2021: 1.81, 2022: 1.67, 2023: 1.71, 2024: 1.76,
    },
    "eu": {
        2000: 8.50, 2001: 8.60, 2002: 9.39, 2003: 11.06, 2004: 12.59,
        2005: 13.79, 2006: 14.67, 2007: 16.68, 2008: 18.28, 2009: 16.44,
        2010: 16.96, 2011: 18.35, 2012: 17.26, 2013: 17.95, 2014: 18.61,
        2015: 14.77, 2016: 14.90, 2017: 15.68, 2018: 18.76, 2019: 18.30,
        2020: 17.87, 2021: 20.07, 2022: 17.16, 2023: 18.34, 2024: 18.97,
    },
    "ru": {
        2000: 0.26, 2001: 0.31, 2002: 0.35, 2003: 0.43, 2004: 0.59,
        2005: 0.76, 2006: 0.99, 2007: 1.30, 2008: 1.66, 2009: 1.22,
        2010: 1.52, 2011: 2.05, 2012: 2.21, 2013: 2.29, 2014: 2.06,
        2015: 1.36, 2016: 1.28, 2017: 1.57, 2018: 1.66, 2019: 1.69,
        2020: 1.48, 2021: 1.78, 2022: 2.24, 2023: 1.86, 2024: 2.02,
    },
}


def interpolate_gdp(country, year):
    """Get GDP for a country/year."""
    return GDP_DATA.get(country, {}).get(year)


def analyze_decoupling(records, c1, c2):
    """Analyze decoupling trends from trade data."""
    if len(records) < 5:
        return {"status": "insufficient_data", "records": len(records)}

    peak = max(records, key=lambda r: r["total"])
    latest = records[-1]
    decline_from_peak = 0
    if peak["total"] > 0:
        decline_from_peak = (latest["total"] - peak["total"]) / peak["total"] * 100

    # Average YoY growth for different periods
    def avg_growth(start_yr, end_yr):
        gs = [r["yoy_change_pct"] for r in records
              if start_yr <= r["year"] <= end_yr and "yoy_change_pct" in r]
        return round(sum(gs) / len(gs), 2) if gs else None

    pre_2018 = avg_growth(2000, 2017)
    post_2018 = avg_growth(2018, 2024)

    # Trade-to-GDP trend
    gdp_key = f"trade_pct_gdp_{c1}"
    ratios = [(r["year"], r.get(gdp_key)) for r in records if r.get(gdp_key)]
    gdp_trend = None
    if len(ratios) >= 2:
        first, last = ratios[0][1], ratios[-1][1]
        if first > 0:
            gdp_trend = round((last - first) / first * 100, 1)

    # Determine status
    if "ru" in (c1, c2):
        if decline_from_peak < -50:
            status = "severe_decoupling_sanctions"
        elif decline_from_peak < -20:
            status = "significant_decoupling_sanctions"
        else:
            status = "partial_sanctions_impact"
    elif set((c1, c2)) == {"us", "cn"}:
        if decline_from_peak < -15:
            status = "significant_decoupling"
        elif post_2018 is not None and pre_2018 is not None and post_2018 < pre_2018 * 0.3:
            status = "growth_slowdown_decoupling"
        elif decline_from_peak < -5:
            status = "moderate_decoupling"
        else:
            status = "no_clear_decoupling"
    else:
        if decline_from_peak < -15:
            status = "declining"
        elif post_2018 is not None and pre_2018 is not None and post_2018 < pre_2018 * 0.3:
            status = "growth_slowdown"
        else:
            status = "stable_or_growing"

    return {
        "status": status,
        "peak_year": peak["year"],
        "peak_total_bn": peak["total"],
        "latest_year": latest["year"],
        "latest_total_bn": latest["total"],
        "decline_from_peak_pct": round(decline_from_peak, 1),
        "avg_yoy_growth_pre_2018": pre_2018,
        "avg_yoy_growth_post_2018": post_2018,
        "trade_gdp_ratio_trend_pct": gdp_trend,
    }


def build_pair_records(c1, c2, raw_data, source):
    """Convert raw (year -> (exports, imports)) to output records."""
    records = []
    for year in sorted(raw_data.keys()):
        exp_val, imp_val = raw_data[year]
        total = exp_val + imp_val
        rec = {
            "year": year,
            f"{c1}_exports_to_{c2}": round(exp_val, 2),
            f"{c1}_imports_from_{c2}": round(imp_val, 2),
            "total": round(total, 2),
        }
        # Trade as % of GDP (values in billions, GDP in trillions)
        gdp_c1 = interpolate_gdp(c1, year)
        gdp_c2 = interpolate_gdp(c2, year)
        if gdp_c1:
            rec[f"trade_pct_gdp_{c1}"] = round(total / (gdp_c1 * 1000) * 100, 3)
        if gdp_c2:
            rec[f"trade_pct_gdp_{c2}"] = round(total / (gdp_c2 * 1000) * 100, 3)
        records.append(rec)

    # YoY changes
    for i in range(1, len(records)):
        prev = records[i - 1]["total"]
        curr = records[i]["total"]
        if prev > 0:
            records[i]["yoy_change_pct"] = round((curr - prev) / prev * 100, 2)

    return records


def try_comtrade_for_pair(c1, c2):
    """Try to get Comtrade data for a non-EU pair. Returns dict {year: (exports, imports)} or None."""
    rep_code = COMTRADE_CODES.get(c1)
    par_code = COMTRADE_CODES.get(c2)
    if not rep_code or not par_code:
        return None

    print(f"  [Comtrade] {c1.upper()}({rep_code}) -> {c2.upper()}({par_code})...")

    results = {}
    # Test with recent year first
    test = comtrade_fetch(rep_code, par_code, [2022, 2023])
    if not test:
        print(f"    No Comtrade data available")
        return None

    # Fetch all years
    for i in range(0, len(YEARS), 5):
        batch = YEARS[i:i + 5]
        batch_data = comtrade_fetch(rep_code, par_code, batch)
        for yr, vals in batch_data.items():
            exp = vals.get("exports", 0)
            imp = vals.get("imports", 0)
            results[yr] = (exp / 1e9, imp / 1e9)  # Convert to billions
        time.sleep(1.2)

    if results:
        print(f"    Got {len(results)} years from Comtrade")
    return results if results else None


def main():
    print("=" * 60)
    print("Bilateral Trade Data Fetcher")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    all_data = {}
    comtrade_available = True

    for c1, c2 in TRADE_PAIRS:
        pair_key = f"{c1}_{c2}"
        print(f"\n--- {pair_key.upper()} ---")

        source = None
        raw_data = None

        # 1. Try Comtrade (skip EU pairs - would need aggregation)
        if comtrade_available and c1 != "eu" and c2 != "eu":
            comtrade_result = try_comtrade_for_pair(c1, c2)
            if comtrade_result:
                raw_data = comtrade_result
                source = "UN Comtrade"
            elif comtrade_result is None:
                # If first pair fails, likely rate limited or down
                pass

        # 2. Fall back to reference data
        if not raw_data and pair_key in KNOWN_TRADE:
            print(f"  [Reference] Using compiled data")
            known = KNOWN_TRADE[pair_key]
            raw_data = known["data"]
            source = known["source"]

        if not raw_data:
            print(f"  WARNING: No data for {pair_key}")
            all_data[pair_key] = {"source": "unavailable", "data": [], "decoupling_trend": None}
            continue

        records = build_pair_records(c1, c2, raw_data, source)
        decoupling = analyze_decoupling(records, c1, c2)

        all_data[pair_key] = {
            "source": source,
            "data": records,
            "decoupling_trend": decoupling,
        }
        print(f"  -> {len(records)} records, trend: {decoupling['status']}")

    # Metadata
    all_data["_metadata"] = {
        "description": "Bilateral trade flows between major economies for decoupling analysis",
        "generated": datetime.now().isoformat(),
        "units": "USD billions",
        "pairs": [f"{a}_{b}" for a, b in TRADE_PAIRS],
        "notes": [
            "EU approximated by aggregate EU27/EU28 figures from Eurostat/IMF DOTS",
            "trade_pct_gdp_X = bilateral trade total as % of country X GDP",
            "yoy_change_pct = year-over-year percentage change in total bilateral trade",
            "Decoupling analysis splits at 2018 (US-China trade war onset)",
            "Russia pairs show sanctions impact post-2022",
        ],
    }

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nOutput: {OUTPUT_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for c1, c2 in TRADE_PAIRS:
        pk = f"{c1}_{c2}"
        info = all_data.get(pk, {})
        n = len(info.get("data", []))
        src = info.get("source", "?")
        dt = info.get("decoupling_trend", {})
        status = dt.get("status", "?") if dt else "?"
        peak = f"peak {dt.get('peak_year','?')}" if dt and dt.get("peak_year") else ""
        decline = f"{dt.get('decline_from_peak_pct',0):+.1f}%" if dt and dt.get("decline_from_peak_pct") is not None else ""
        print(f"  {pk:10s} | {n:2d} yrs | {status:35s} | {peak} {decline} | {src}")

    print(f"\nDone: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
