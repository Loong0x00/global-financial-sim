#!/usr/bin/env python3
"""
Multi-Leader Transformer Decision Model  v4 — PCA All-Data
=============================================================
Extends v3 by replacing 10 hand-picked economic features with PCA
components from ALL available economic data series (~500+ series).

Key improvements over v3:
  - Load ALL economic data: macro/financial (6 countries), BIS (10 datasets),
    commodities (FRED+Yahoo), global indices, crypto, supplements, CFTC, trade
  - Build monthly matrix: rows=months (1971-01 to 2026-02), cols=all series
  - PCA on training set only (no data leakage) → top 30 components
  - Larger model: d_model=128, nhead=8, 4 layers (richer input warrants it)

New input vector (~85D total):
  [pca_economic_state 30D]        PCA of all economic series
  [central_banker_embedding 16D]  same as v3
  [context 5D]                    same as v3
  [us_president_embedding 8D]     same as v3
  [xi_jinping_embedding 8D]       same as v3
  [putin_embedding 8D]            same as v3
  [political_action_signals 5D]   same as v3
  [time_features 6D]              same as v3
"""

import json
import math
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 70)
print("MULTI-LEADER TRANSFORMER DECISION MODEL  v4 — PCA ALL-DATA")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not available — will skip model training")

# ─── GPU Device Setup ─────────────────────────────────────────────────────────
if TORCH_OK:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU")
        device = torch.device("cpu")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
        print(f"[GPU] CUDA {torch.version.cuda}  device={device}")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path("/home/user/global-financial-sim")
DATA        = BASE / "data"
ECON_DIR    = DATA / "economic"
PROFILES    = DATA / "leaders" / "profiles"
TIMELINES   = DATA / "leaders" / "timelines"
MODELS_DIR  = BASE / "models" / "decision_functions"
OUTPUT_DIR  = BASE / "output" / "decision_function"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Central Bank Leader Registry ─────────────────────────────────────────────
LEADER_REGISTRY = [
    ("greenspan",       "US", "greenspan",    "1987-08", "2006-01"),
    ("bernanke",        "US", "bernanke",     "2006-02", "2014-01"),
    ("yellen",          "US", "yellen",       "2014-02", "2018-01"),
    ("powell",          "US", "powell",       "2018-02", None),
    ("draghi",          "EU", "draghi",       "2011-11", "2019-10"),
    ("lagarde",         "EU", "lagarde",      "2019-11", None),
    ("nabiullina",      "RU", "nabiullina",   "2013-06", None),
    ("zhou_xiaochuan",  "CN", "zhou_xiaochuan","2002-12", "2018-03"),
    ("kuroda",          "JP", None,           "2013-03", "2023-04"),
    ("ueda",            "JP", None,           "2023-04", None),
    ("carney_boe",      "UK", None,           "2013-07", "2020-03"),
    ("bailey_boe",      "UK", None,           "2020-03", None),
]

# ─── Political Leader Registry ────────────────────────────────────────────────
POLITICAL_REGISTRY = [
    ("obama",      "obama",    "2009-01", "2017-01"),
    ("trump_t1",   "trump",    "2017-01", "2021-01"),
    ("biden",      None,       "2021-01", "2025-01"),
    ("trump_t2",   "trump",    "2025-01", None),
    ("xi_jinping", "xi_jinping", "2012-11", None),
    ("putin",      "putin",    "2000-05", None),
]

# ─── Dimensions ──────────────────────────────────────────────────────────────
N_PCA_COMPONENTS    = 30   # PCA economic features
LEADER_EMB_DIM      = 16
N_CONTEXT_FEATURES  = 5
POL_EMB_DIM         = 8
N_POL_SLOTS         = 3
N_POL_ACTION        = 5
N_TIME_FEATURES     = 6

TOTAL_DIM = (N_PCA_COMPONENTS + LEADER_EMB_DIM + N_CONTEXT_FEATURES
             + N_POL_SLOTS * POL_EMB_DIM + N_POL_ACTION + N_TIME_FEATURES)
# = 30 + 16 + 5 + 24 + 5 + 6 = 86

# ─── Common CB Params for central banker embedding ────────────────────────────
COMMON_CB_PARAMS = [
    "inflation_sensitivity_headline_cpi",
    "unemployment_sensitivity",
    "gdp_growth_sensitivity",
    "financial_stability_weight",
    "data_dependence",
    "forward_guidance_credibility",
    "independence_from_political_pressure",
    "risk_management_approach",
    "communication_transparency",
    "global_spillover_sensitivity",
    "quantitative_easing_willingness",
    "emergency_rate_hike_decisiveness",
    "dovish_bias",
    "hawkish_bias",
    "crisis_response_speed",
    "verbal_intervention_potency",
    "labor_market_dovish_bias",
    "deflation_spiral_fear",
    "exchange_rate_reform_commitment",
    "covid_pepp_response_speed",
    "flexibility_under_constraint",
    "consensus_building",
    "asset_bubble_tolerance",
    "yield_curve_control_willingness",
    "currency_stability_priority",
    "information_fidelity",
    "institutional_credibility_weight",
    "policy_reversal_willingness",
    "external_shock_responsiveness",
    "fiscal_monetary_coordination",
    "emerging_market_sensitivity",
    "macro_prudential_integration",
    "academic_rigor_in_decisions",
    "market_communication_skill",
    "patience_duration",
    "geopolitical_sensitivity",
]
assert len(COMMON_CB_PARAMS) >= LEADER_EMB_DIM

# ─── Political Leader Embedding Params ───────────────────────────────────────
TRUMP_POL_PARAMS = [
    "fed_political_pressure_intensity",
    "tariff_as_negotiation_weapon",
    "fiscal_deficit_tolerance",
    "china_confrontation_intensity",
    "stock_market_as_approval_metric",
    "deal_making_maximum_pressure_pattern",
    "supply_side_tax_cut_preference",
    "multilateral_institution_respect",
]

XI_POL_PARAMS = [
    "political_security_over_economic_growth",
    "state_vs_market_balance_preference",
    "flood_style_stimulus_aversion",
    "targeted_stimulus_preference",
    "trade_retaliation_willingness",
    "capital_controls_tightening_willingness",
    "tech_sector_crackdown_intensity",
    "dual_circulation_strategy_commitment",
]

PUTIN_POL_PARAMS = [
    "energy_as_geopolitical_weapon_propensity",
    "sanctions_resilience_preparation_premeditation",
    "military_escalation_threshold",
    "political_survival_primacy_over_economic_optimality",
    "state_enterprise_preference_over_private",
    "gdp_growth_tolerance_for_geopolitical_goals",
    "information_warfare_as_economic_tool",
    "fiscal_conservatism_base_disposition",
]

BIDEN_DEFAULT_EMBEDDING = np.array([
    0.10, 0.15, 0.75, 0.30, 0.30, 0.20, 0.35, 0.75,
], dtype=np.float32)

OBAMA_DEFAULT_EMBEDDING = np.array([
    0.05, 0.10, 0.60, 0.20, 0.25, 0.15, 0.25, 0.90,
], dtype=np.float32)

OBAMA_NATIVE_PARAMS = [
    "multilateral_institution_preference",
    "fiscal_stimulus_economic_multiplier_belief",
    "geopolitical_risk_aversion_military",
    "qe_coordination_bernanke_yellen",
    "dont_do_stupid_stuff_risk_aversion",
    "tpp_trade_policy_engagement",
    "financial_sector_regulatory_toughness",
    "income_inequality_attention",
]


# ─── Data Loading Utilities ───────────────────────────────────────────────────

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _date_to_month(date_str):
    """Convert various date formats to YYYY-MM."""
    if not date_str:
        return None
    # Already YYYY-MM
    if re.match(r'^\d{4}-\d{2}$', date_str):
        return date_str
    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{2})-\d{2}$', date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # YYYY-QN (quarterly)
    m = re.match(r'^(\d{4})-Q(\d)$', date_str)
    if m:
        yr, q = int(m.group(1)), int(m.group(2))
        month = (q - 1) * 3 + 1
        return f"{yr:04d}-{month:02d}"
    return None


def _all_months(start="1971-01", end="2026-02"):
    """Generate list of YYYY-MM from start to end."""
    months = []
    y, m = int(start[:4]), int(start[5:])
    ey, em = int(end[:4]), int(end[5:])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


# ─── Universal Economic Data Loader ──────────────────────────────────────────

class AllEconomicDataStore:
    """
    Loads ALL economic data from every source into a unified monthly matrix.
    Each series becomes a column. Rows are months from 1971-01 to 2026-02.
    """

    def __init__(self):
        self._series = {}          # series_name -> {month: value}
        self._rate_data = {}       # country -> {month: rate}  (for rate decisions)
        self._bis_rates = {}       # country -> {month: value}
        self._all_months = _all_months("1971-01", "2026-02")
        self._month_to_idx = {m: i for i, m in enumerate(self._all_months)}
        self._load_all()

    def _add_series(self, name, data_dict):
        """Add a series: name -> {month_str: float_value}"""
        monthly = {}
        for date_str, val in data_dict.items():
            if val is None:
                continue
            month = _date_to_month(date_str)
            if month and month in self._month_to_idx:
                try:
                    monthly[month] = float(val)
                except (ValueError, TypeError):
                    continue
        if len(monthly) >= 6:  # require at least 6 data points
            self._series[name] = monthly

    def _load_macro_financial(self):
        """Load macro.json and financial.json for all 6 countries."""
        for country in ["us", "eu", "ru", "cn", "jp", "uk"]:
            cc = country.upper()
            for ftype in ["macro", "financial"]:
                path = ECON_DIR / country / f"{ftype}.json"
                if not path.exists():
                    continue
                d = load_json(path)
                for ind_name, ind_data in d.get("indicators", {}).items():
                    series_data = {}
                    for item in ind_data.get("series", []):
                        if isinstance(item, dict) and "date" in item and item.get("value") is not None:
                            series_data[item["date"]] = item["value"]
                    self._add_series(f"{cc}_{ftype}_{ind_name}", series_data)

                    # Also store for rate lookup
                    if ftype == "macro" and ind_name == "interest_rate":
                        monthly = {}
                        for date_str, val in series_data.items():
                            month = _date_to_month(date_str)
                            if month:
                                try:
                                    monthly[month] = float(val)
                                except (ValueError, TypeError):
                                    pass
                        self._rate_data[cc] = monthly

                    if ftype == "macro" and ind_name == "fed_funds_rate" and cc == "US":
                        monthly = {}
                        for date_str, val in series_data.items():
                            month = _date_to_month(date_str)
                            if month:
                                try:
                                    monthly[month] = float(val)
                                except (ValueError, TypeError):
                                    pass
                        self._rate_data["US"] = monthly

    def _load_bis(self):
        """Load all BIS datasets."""
        bis_dir = ECON_DIR / "bis"
        if not bis_dir.exists():
            return

        country_map = {"US": "US", "CN": "CN", "JP": "JP", "GB": "UK",
                        "DE": "EU", "RU": "RU", "FR": "EU"}

        for fpath in bis_dir.glob("*.json"):
            if fpath.name.startswith("_"):
                continue
            d = load_json(fpath)
            dataset = d.get("dataset", fpath.stem)

            # BIS policy rates for rate lookup
            if dataset == "policy_rates":
                bis_cc_map = {"US": "US", "RU": "RU", "CN": "CN", "JP": "JP", "GB": "UK"}
                for bis_code, our_code in bis_cc_map.items():
                    if bis_code in d.get("countries", {}):
                        ser_data = d["countries"][bis_code]["series"]
                        for sk, sv in ser_data.items():
                            monthly = {}
                            for k, v in sv["data"].items():
                                month = _date_to_month(k)
                                if month:
                                    try:
                                        monthly[month] = float(v)
                                    except (ValueError, TypeError):
                                        pass
                            self._bis_rates[our_code] = monthly
                            break

            # Skip mega-datasets (20k+ sub-series each, too granular)
            if dataset in ("debt_securities", "locational_banking", "consolidated_banking"):
                continue

            for country_code, cdata in d.get("countries", {}).items():
                our_cc = country_map.get(country_code, country_code)
                for i, (ser_name, ser_val) in enumerate(cdata.get("series", {}).items()):
                    data_dict = {}
                    for k, v in ser_val.get("data", {}).items():
                        data_dict[k] = v
                    short_name = f"BIS_{dataset}_{our_cc}_{i}"
                    self._add_series(short_name, data_dict)

    def _load_commodities(self):
        """Load commodities data (FRED + Yahoo + gold/silver daily)."""
        com_dir = ECON_DIR / "commodities"
        if not com_dir.exists():
            return

        for fpath in com_dir.glob("*.json"):
            if fpath.name == "manifest.json":
                continue
            d = load_json(fpath)

            if fpath.name in ("gold_daily.json", "silver_daily.json"):
                # Daily data with close prices - aggregate to monthly
                name_prefix = fpath.stem.replace("_daily", "")
                if isinstance(d, dict) and "data" in d:
                    items = d["data"]
                elif isinstance(d, list):
                    items = d
                else:
                    continue
                monthly_agg = defaultdict(list)
                for item in items:
                    if isinstance(item, dict):
                        date = item.get("date", "")
                        val = item.get("close") or item.get("value")
                        month = _date_to_month(date)
                        if month and val is not None:
                            try:
                                monthly_agg[month].append(float(val))
                            except (ValueError, TypeError):
                                pass
                data_dict = {m: np.mean(vals) for m, vals in monthly_agg.items()}
                self._add_series(f"COM_{name_prefix}", data_dict)
                continue

            # fred_commodities.json or yahoo_futures.json: dict of series
            if isinstance(d, dict):
                for ser_name, ser_val in d.items():
                    if not isinstance(ser_val, dict) or "data" not in ser_val:
                        continue
                    items = ser_val["data"]
                    if not isinstance(items, list):
                        continue
                    # Aggregate daily to monthly
                    monthly_agg = defaultdict(list)
                    for item in items:
                        if isinstance(item, dict):
                            date = item.get("date", "")
                            val = item.get("value") or item.get("close")
                            month = _date_to_month(date)
                            if month and val is not None:
                                try:
                                    monthly_agg[month].append(float(val))
                                except (ValueError, TypeError):
                                    pass
                    data_dict = {m: np.mean(vals) for m, vals in monthly_agg.items()}
                    self._add_series(f"COM_{ser_name}", data_dict)

    def _load_indices(self):
        """Load global stock indices (daily → monthly average)."""
        path = ECON_DIR / "indices" / "global_indices.json"
        if not path.exists():
            return
        d = load_json(path)
        for idx_name, idx_data in d.items():
            if not isinstance(idx_data, dict) or "data" not in idx_data:
                continue
            monthly_agg = defaultdict(list)
            for item in idx_data["data"]:
                if isinstance(item, dict):
                    date = item.get("date", "")
                    val = item.get("close")
                    month = _date_to_month(date)
                    if month and val is not None:
                        try:
                            monthly_agg[month].append(float(val))
                        except (ValueError, TypeError):
                            pass
            data_dict = {m: np.mean(vals) for m, vals in monthly_agg.items()}
            self._add_series(f"IDX_{idx_name}", data_dict)

    def _load_crypto(self):
        """Load crypto data (daily → monthly average)."""
        path = ECON_DIR / "crypto" / "crypto.json"
        if not path.exists():
            return
        d = load_json(path)
        for coin_name, coin_data in d.items():
            if not isinstance(coin_data, dict) or "data" not in coin_data:
                continue
            monthly_agg = defaultdict(list)
            for item in coin_data["data"]:
                if isinstance(item, dict):
                    date = item.get("date", "")
                    val = item.get("close")
                    month = _date_to_month(date)
                    if month and val is not None:
                        try:
                            monthly_agg[month].append(float(val))
                        except (ValueError, TypeError):
                            pass
            data_dict = {m: np.mean(vals) for m, vals in monthly_agg.items()}
            self._add_series(f"CRYPTO_{coin_name}", data_dict)

    def _load_supplements(self):
        """Load supplementary data (uk_bank_rate, russia, china, us_additional)."""
        sup_dir = ECON_DIR / "supplementary"
        if not sup_dir.exists():
            return
        for fpath in sup_dir.glob("*.json"):
            if fpath.name == "manifest.json":
                continue
            d = load_json(fpath)
            for ser_name, ser_val in d.items():
                if not isinstance(ser_val, dict) or "data" not in ser_val:
                    continue
                items = ser_val["data"]
                if not isinstance(items, list):
                    continue
                data_dict = {}
                for item in items:
                    if isinstance(item, dict):
                        date = item.get("date", "")
                        val = item.get("value")
                        if date and val is not None:
                            data_dict[date] = val
                self._add_series(f"SUP_{ser_name}", data_dict)

    def _load_cftc(self):
        """Load CFTC COT data — net speculative position per contract."""
        path = ECON_DIR / "cftc" / "cot_data.json"
        if not path.exists():
            return
        d = load_json(path)
        for contract_code, cdata in d.get("contracts", {}).items():
            name = cdata.get("contract_name", contract_code)
            items = cdata.get("data", [])
            # Weekly data → take last value per month
            monthly = {}
            for item in items:
                if isinstance(item, dict):
                    date = item.get("date", "")
                    val = item.get("net_speculative")
                    month = _date_to_month(date)
                    if month and val is not None:
                        try:
                            monthly[month] = float(val)  # last week of month wins
                        except (ValueError, TypeError):
                            pass
            self._add_series(f"CFTC_{name.replace(' ', '_')}", monthly)

    def _load_trade(self):
        """Load bilateral trade data (annual → spread to months)."""
        path = ECON_DIR / "trade" / "bilateral_trade.json"
        if not path.exists():
            return
        d = load_json(path)
        for pair_name, pair_data in d.items():
            if not isinstance(pair_data, dict) or "data" not in pair_data:
                continue
            items = pair_data["data"]
            if not isinstance(items, list):
                continue
            for field in ["total", "trade_pct_gdp_us", "trade_pct_gdp_cn",
                          "us_exports_to_cn", "us_imports_from_cn"]:
                yearly = {}
                for item in items:
                    if isinstance(item, dict) and "year" in item and field in item:
                        yr = item["year"]
                        val = item[field]
                        if val is not None:
                            yearly[yr] = val
                if not yearly:
                    continue
                # Spread annual to monthly (constant within year)
                monthly = {}
                for yr, val in yearly.items():
                    for mo in range(1, 13):
                        monthly[f"{yr:04d}-{mo:02d}"] = val
                self._add_series(f"TRADE_{pair_name}_{field}", monthly)

    def _load_all(self):
        print("\n[0] Loading ALL economic data...")
        self._load_macro_financial()
        print(f"  Macro/Financial: {len(self._series)} series")
        n_before = len(self._series)
        self._load_bis()
        print(f"  BIS: +{len(self._series) - n_before} series")
        n_before = len(self._series)
        self._load_commodities()
        print(f"  Commodities: +{len(self._series) - n_before} series")
        n_before = len(self._series)
        self._load_indices()
        print(f"  Indices: +{len(self._series) - n_before} series")
        n_before = len(self._series)
        self._load_crypto()
        print(f"  Crypto: +{len(self._series) - n_before} series")
        n_before = len(self._series)
        self._load_supplements()
        print(f"  Supplements: +{len(self._series) - n_before} series")
        n_before = len(self._series)
        self._load_cftc()
        print(f"  CFTC: +{len(self._series) - n_before} series")
        n_before = len(self._series)
        self._load_trade()
        print(f"  Trade: +{len(self._series) - n_before} series")
        print(f"  TOTAL: {len(self._series)} economic series loaded")
        print(f"  Rate data for: {list(self._rate_data.keys())}")
        print(f"  BIS rates for: {list(self._bis_rates.keys())}")

    def get_rate(self, country, month):
        """Get interest rate for a country at a given month."""
        if country in self._rate_data:
            r = self._rate_data[country].get(month)
            if r is not None:
                return r
        if country in self._bis_rates:
            return self._bis_rates[country].get(month)
        return None

    def build_monthly_matrix(self):
        """
        Build the full monthly matrix: rows=months, cols=series.
        Returns: matrix (n_months, n_series), series_names, month_list
        """
        series_names = sorted(self._series.keys())
        n_months = len(self._all_months)
        n_series = len(series_names)

        matrix = np.full((n_months, n_series), np.nan, dtype=np.float64)
        for j, sname in enumerate(series_names):
            sdata = self._series[sname]
            for month, val in sdata.items():
                if month in self._month_to_idx:
                    matrix[self._month_to_idx[month], j] = val

        return matrix, series_names, self._all_months

    # Compatibility methods for political action signals (use raw data)
    def get(self, source, country, indicator, month):
        key = f"{country}_{source}_{indicator}"
        sdata = self._series.get(key, {})
        return sdata.get(month)

    def get_yoy(self, source, country, indicator, month):
        y, m = int(month[:4]), int(month[5:])
        prev_month = f"{y-1:04d}-{m:02d}"
        curr = self.get(source, country, indicator, month)
        prev = self.get(source, country, indicator, prev_month)
        if curr is not None and prev is not None and prev != 0:
            return (curr - prev) / abs(prev) * 100.0
        return None


# ─── PCA Pipeline ─────────────────────────────────────────────────────────────

class PCAEconomicPipeline:
    """
    Fits PCA on training months, transforms any month to N_PCA_COMPONENTS features.
    PCA is fit ONLY on training data to avoid data leakage.
    """

    def __init__(self, n_components=N_PCA_COMPONENTS):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False
        self._matrix = None
        self._series_names = None
        self._months = None
        self._month_to_idx = None

    def set_matrix(self, matrix, series_names, months):
        """Store the full matrix for lookup."""
        self._matrix = matrix
        self._series_names = series_names
        self._months = months
        self._month_to_idx = {m: i for i, m in enumerate(months)}

    def _impute_matrix(self, mat):
        """Forward fill, backward fill, then fill remaining with 0."""
        result = mat.copy()
        # Forward fill along time axis
        for j in range(result.shape[1]):
            last_valid = np.nan
            for i in range(result.shape[0]):
                if np.isnan(result[i, j]):
                    if not np.isnan(last_valid):
                        result[i, j] = last_valid
                else:
                    last_valid = result[i, j]
        # Backward fill
        for j in range(result.shape[1]):
            last_valid = np.nan
            for i in range(result.shape[0] - 1, -1, -1):
                if np.isnan(result[i, j]):
                    if not np.isnan(last_valid):
                        result[i, j] = last_valid
                else:
                    last_valid = result[i, j]
        # Fill remaining NaN with 0
        result = np.nan_to_num(result, nan=0.0)
        return result

    def fit(self, train_month_indices):
        """Fit scaler + PCA on training months only."""
        # Impute the full matrix first
        imputed = self._impute_matrix(self._matrix)

        # Extract training rows
        train_data = imputed[train_month_indices]

        # Fit scaler and PCA on training data only
        scaled = self.scaler.fit_transform(train_data)

        # Determine actual n_components (may be less than requested)
        max_comp = min(self.n_components, scaled.shape[0], scaled.shape[1])
        if max_comp < self.n_components:
            print(f"  [PCA] Reducing components from {self.n_components} to {max_comp}")
            self.pca = PCA(n_components=max_comp)

        self.pca.fit(scaled)
        self._imputed = imputed
        self._fitted = True

        explained = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"  [PCA] Fitted on {len(train_month_indices)} months × {imputed.shape[1]} series")
        print(f"  [PCA] {self.pca.n_components_} components explain {explained:.1f}% variance")
        return explained

    def transform_month(self, month):
        """Transform a single month to PCA features."""
        if not self._fitted:
            return np.zeros(self.n_components, dtype=np.float32)
        idx = self._month_to_idx.get(month)
        if idx is None:
            return np.zeros(self.n_components, dtype=np.float32)

        row = self._imputed[idx:idx+1]
        scaled = self.scaler.transform(row)
        pca_features = self.pca.transform(scaled)[0]

        # Pad if fewer components than requested
        if len(pca_features) < self.n_components:
            pca_features = np.pad(pca_features,
                                   (0, self.n_components - len(pca_features)),
                                   constant_values=0.0)
        return pca_features.astype(np.float32)

    def get_n_actual_components(self):
        if self._fitted:
            return self.pca.n_components_
        return self.n_components


# ─── Rate Decision Extraction ─────────────────────────────────────────────────

def classify_rate_change(delta_bp):
    if delta_bp < -5:
        direction = 0
    elif delta_bp > 5:
        direction = 2
    else:
        direction = 1

    abs_bp = abs(delta_bp)
    if abs_bp < 12.5:
        magnitude = 0
    elif abs_bp < 37.5:
        magnitude = 1
    elif abs_bp < 62.5:
        magnitude = 2
    elif abs_bp < 87.5:
        magnitude = 3
    else:
        magnitude = 4

    return direction, magnitude


def extract_rate_decisions(econ, leader_id, country, start_yyyymm, end_yyyymm=None):
    months = _all_months(start_yyyymm, end_yyyymm or "2026-02")
    decisions = []
    prev_rate = None

    for month in months:
        rate = econ.get_rate(country, month)
        if rate is None:
            prev_rate = None
            continue
        if prev_rate is not None:
            delta_bp = (rate - prev_rate) * 100.0
            direction, magnitude = classify_rate_change(delta_bp)
            decisions.append({
                "month": month,
                "rate": rate,
                "prev_rate": prev_rate,
                "delta_bp": round(delta_bp, 1),
                "direction": direction,
                "magnitude": magnitude,
            })
        prev_rate = rate

    return decisions


# ─── CB Leader Embedding ──────────────────────────────────────────────────────

def extract_leader_embedding(profile_name):
    if profile_name is None:
        return np.full(LEADER_EMB_DIM, 0.5, dtype=np.float32)

    path = PROFILES / f"{profile_name}.json"
    if not path.exists():
        return np.full(LEADER_EMB_DIM, 0.5, dtype=np.float32)

    try:
        profile = load_json(path)
    except Exception:
        return np.full(LEADER_EMB_DIM, 0.5, dtype=np.float32)

    lu = {}
    for v in profile.get("behavior_matrix", {}).get("vectors", []):
        if "label" in v and "value" in v and v["value"] is not None:
            lu[v["label"]] = float(v["value"])
    for group in profile.get("behavioral_parameters", []):
        for param in group.get("parameters", []):
            if "name" in param and "value" in param:
                lu[param["name"]] = float(param["value"])

    vals = []
    for label in COMMON_CB_PARAMS[:LEADER_EMB_DIM]:
        if label in lu:
            vals.append(lu[label])
        else:
            matches = [v for k, v in lu.items() if k.startswith(label[:10])]
            vals.append(float(np.mean(matches)) if matches else 0.5)

    arr = np.array(vals[:LEADER_EMB_DIM], dtype=np.float32)
    if len(arr) < LEADER_EMB_DIM:
        arr = np.pad(arr, (0, LEADER_EMB_DIM - len(arr)), constant_values=0.5)
    return arr


# ─── Political Leader Embedding ───────────────────────────────────────────────

def _extract_pol_embedding_from_profile(profile_name, param_list, default_val=0.5):
    path = PROFILES / f"{profile_name}.json"
    if not path.exists():
        return np.full(POL_EMB_DIM, default_val, dtype=np.float32)

    try:
        profile = load_json(path)
    except Exception:
        return np.full(POL_EMB_DIM, default_val, dtype=np.float32)

    lu = {}
    for v in profile.get("behavior_matrix", {}).get("vectors", []):
        if "label" in v and "value" in v and v["value"] is not None:
            lu[v["label"]] = float(v["value"])

    vals = []
    for label in param_list:
        if label in lu:
            vals.append(lu[label])
        else:
            matches = [v for k, v in lu.items() if k.startswith(label[:12])]
            vals.append(float(np.mean(matches)) if matches else default_val)

    arr = np.array(vals[:POL_EMB_DIM], dtype=np.float32)
    if len(arr) < POL_EMB_DIM:
        arr = np.pad(arr, (0, POL_EMB_DIM - len(arr)), constant_values=default_val)
    return arr


def build_political_embeddings():
    embeddings = {}
    trump_emb = _extract_pol_embedding_from_profile("trump", TRUMP_POL_PARAMS)
    embeddings["trump_t1"] = trump_emb
    embeddings["trump_t2"] = trump_emb
    embeddings["biden"] = BIDEN_DEFAULT_EMBEDDING.copy()

    obama_emb = _extract_pol_embedding_from_profile("obama", OBAMA_NATIVE_PARAMS)
    if float(np.std(obama_emb)) < 0.05:
        embeddings["obama"] = OBAMA_DEFAULT_EMBEDDING.copy()
    else:
        embeddings["obama"] = obama_emb

    embeddings["xi_jinping"] = _extract_pol_embedding_from_profile(
        "xi_jinping", XI_POL_PARAMS)
    embeddings["putin"] = _extract_pol_embedding_from_profile(
        "putin", PUTIN_POL_PARAMS)
    embeddings["zero"] = np.zeros(POL_EMB_DIM, dtype=np.float32)

    return embeddings


def _month_in_range(month, start, end):
    if month < start:
        return False
    if end is not None and month >= end:
        return False
    return True


def get_active_political_leaders(month):
    result = {"us_president": None, "xi": None, "putin": None}
    for (lid, profile_name, start, end) in POLITICAL_REGISTRY:
        if not _month_in_range(month, start, end):
            continue
        if lid in ("obama", "trump_t1", "biden", "trump_t2"):
            result["us_president"] = lid
        elif lid == "xi_jinping":
            result["xi"] = lid
        elif lid == "putin":
            result["putin"] = lid
    return result


def extract_political_embeddings_for_month(month, pol_embeddings):
    active = get_active_political_leaders(month)
    us_emb  = pol_embeddings.get(active["us_president"], pol_embeddings["zero"])
    xi_emb  = pol_embeddings.get(active["xi"],           pol_embeddings["zero"])
    put_emb = pol_embeddings.get(active["putin"],        pol_embeddings["zero"])
    return np.concatenate([us_emb, xi_emb, put_emb]).astype(np.float32)


# ─── Time Features (same as v3) ──────────────────────────────────────────────

US_ELECTION_YEARS = [1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000,
                     2004, 2008, 2012, 2016, 2020, 2024, 2028]

US_PRESIDENT_TENURES = [
    ("1969-01", "1974-08"), ("1974-08", "1977-01"), ("1977-01", "1981-01"),
    ("1981-01", "1989-01"), ("1989-01", "1993-01"), ("1993-01", "2001-01"),
    ("2001-01", "2009-01"), ("2009-01", "2017-01"), ("2017-01", "2021-01"),
    ("2021-01", "2025-01"), ("2025-01", "2029-01"),
]


def extract_time_features(month, cb_start, cb_end):
    features = np.zeros(N_TIME_FEATURES, dtype=np.float32)
    yr = int(month[:4])
    mo = int(month[5:7])
    month_abs = yr * 12 + mo

    features[0] = (yr - 1970) / 60.0

    if cb_start:
        s_yr, s_mo = int(cb_start[:4]), int(cb_start[5:7])
        start_abs = s_yr * 12 + s_mo
        if cb_end:
            e_yr, e_mo = int(cb_end[:4]), int(cb_end[5:7])
            end_abs = e_yr * 12 + e_mo
        else:
            end_abs = 2026 * 12 + 3
        total = max(end_abs - start_abs, 1)
        progress = (month_abs - start_abs) / total
        features[1] = float(np.clip(progress, 0, 1))

    for ey in US_ELECTION_YEARS:
        elec_abs = ey * 12 + 11
        if elec_abs >= month_abs:
            dist = elec_abs - month_abs
            features[2] = float(np.clip(1.0 - dist / 48.0, 0, 1))
            break

    decade_phase = (yr + mo / 12.0) / 10.0
    features[3] = math.sin(2 * math.pi * decade_phase)
    features[4] = math.cos(2 * math.pi * decade_phase)

    for ps, pe in US_PRESIDENT_TENURES:
        ps_yr, ps_mo = int(ps[:4]), int(ps[5:7])
        pe_yr, pe_mo = int(pe[:4]), int(pe[5:7])
        ps_abs = ps_yr * 12 + ps_mo
        pe_abs = pe_yr * 12 + pe_mo
        if ps_abs <= month_abs < pe_abs:
            total_p = max(pe_abs - ps_abs, 1)
            features[5] = float((month_abs - ps_abs) / total_p)
            break

    return features


# ─── Political Action Signals ────────────────────────────────────────────────

def extract_political_action_signals(econ, month):
    signals = np.zeros(N_POL_ACTION, dtype=np.float32)

    us_exports_yoy = econ.get_yoy("macro", "US", "nonfarm_payrolls_change", month)
    if us_exports_yoy is None:
        us_exports_yoy = econ.get_yoy("macro", "CN", "exports", month)
    if us_exports_yoy is not None:
        signals[0] = float(np.clip(us_exports_yoy / 20.0, -1.5, 1.5))

    m2_yoy = econ.get_yoy("macro", "US", "fed_funds_rate", month)
    for country, src, ind in [
        ("US", "macro", "m2_money_supply"),
        ("US", "financial", "m2"),
        ("EU", "macro", "m3_money_supply"),
    ]:
        v = econ.get_yoy(src, country, ind, month)
        if v is not None:
            m2_yoy = v
            break
    if m2_yoy is not None:
        signals[1] = float(np.clip(m2_yoy / 15.0, -1.5, 1.5))

    vix = econ.get("financial", "US", "vix", month)
    if vix is not None:
        signals[2] = float(np.clip((vix - 10.0) / 60.0, 0.0, 1.5))

    ru_fx_yoy = econ.get_yoy("macro", "RU", "exchange_rate_usd_rub", month)
    if ru_fx_yoy is not None:
        signals[3] = float(np.clip(ru_fx_yoy / 50.0, -1.0, 1.5))

    yield_curve = econ.get("financial", "US", "yield_curve_10y2y", month)
    if yield_curve is not None:
        signals[4] = float(np.clip(-yield_curve / 2.0, -1.0, 1.5))

    return signals


# ─── Context Features ────────────────────────────────────────────────────────

def compute_context(decision_idx, decisions, month):
    hike_streak = 0
    cut_streak  = 0
    hold_streak = 0

    for i in range(decision_idx - 1, max(-1, decision_idx - 12), -1):
        d = decisions[i]["direction"]
        if d == 2:
            hike_streak += 1
            cut_streak = 0
            hold_streak = 0
        elif d == 0:
            cut_streak += 1
            hike_streak = 0
            hold_streak = 0
        else:
            hold_streak += 1
            break

    m = int(month[5:])
    return np.array([hike_streak, cut_streak, hold_streak,
                     math.sin(2 * math.pi * m / 12),
                     math.cos(2 * math.pi * m / 12)], dtype=np.float32)


# ─── Political Action Normalizer ─────────────────────────────────────────────

class PoliticalActionNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, signal_matrix):
        self.mean = np.nanmean(signal_matrix, axis=0)
        self.std  = np.nanstd(signal_matrix, axis=0)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, signals):
        if self.mean is None:
            return signals
        out = (signals - self.mean) / self.std
        return np.clip(out, -3, 3)

    def to_dict(self):
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std":  self.std.tolist()  if self.std  is not None else None,
        }


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_multi_leader_dataset(econ, pol_embeddings):
    """Build dataset with all samples (PCA features added later)."""
    print("\n[1] Building political leader embeddings...")
    for lid, emb in pol_embeddings.items():
        if lid == "zero":
            continue
        nz = np.count_nonzero(emb)
        print(f"  {lid:15s}: {emb.round(2).tolist()}  (non-zero: {nz}/{POL_EMB_DIM})")

    print("\n[2] Extracting rate decisions for all central bank leaders...")

    all_samples  = []
    cb_embeddings = {}
    stats        = {}

    for (lid, country, profile_name, start, end) in LEADER_REGISTRY:
        emb       = extract_leader_embedding(profile_name)
        cb_embeddings[lid] = emb
        has_profile = profile_name is not None and (PROFILES / f"{profile_name}.json").exists()

        decisions = extract_rate_decisions(econ, lid, country, start, end)
        if not decisions:
            print(f"  {lid} ({country}): NO RATE DATA found")
            continue

        samples = []
        for i, dec in enumerate(decisions):
            month    = dec["month"]
            ctx      = compute_context(i, decisions, month)

            direction = dec["direction"]
            guidance  = 2 if direction == 2 else (0 if direction == 0 else 1)

            pol_emb_vec = extract_political_embeddings_for_month(month, pol_embeddings)
            pol_actions = extract_political_action_signals(econ, month)
            time_feats = extract_time_features(month, start, end)

            samples.append({
                "leader_id":   lid,
                "country":     country,
                "month":       month,
                "context":     ctx,
                "pol_emb":     pol_emb_vec,
                "pol_actions": pol_actions,
                "time_feats":  time_feats,
                "direction":   direction,
                "magnitude":   dec["magnitude"],
                "guidance":    guidance,
                "delta_bp":    dec["delta_bp"],
            })

        dirs = [s["direction"] for s in samples]
        n_cut, n_hold, n_hike = dirs.count(0), dirs.count(1), dirs.count(2)
        print(f"  {lid:22s} ({country}) [{start}→{end or 'now'}]: "
              f"{len(samples):4d} samples  cut={n_cut} hold={n_hold} hike={n_hike}  "
              f"profile={'YES' if has_profile else 'NO '}")

        stats[lid] = {
            "n_samples": len(samples),
            "n_cut": n_cut, "n_hold": n_hold, "n_hike": n_hike,
            "country": country,
            "has_profile": has_profile,
        }
        all_samples.extend(samples)

    print(f"\n  Total samples: {len(all_samples)}")
    return all_samples, cb_embeddings, stats


# ─── Feature Assembly ────────────────────────────────────────────────────────

def assemble_feature_vector(sample, cb_embeddings, pca_pipeline):
    """
    Combine all features into a single vector.
    Layout:
      [0:30]   PCA economic features     (30D)  ← v4 NEW
      [30:46]  cb_leader_embedding       (16D)
      [46:51]  context                   (5D)
      [51:75]  pol_embeddings            (24D = 3 slots × 8D)
      [75:80]  pol_actions_norm          (5D)
      [80:86]  time_features             (6D)
    """
    pca_feats     = pca_pipeline.transform_month(sample["month"])   # (30,)
    cb_emb        = cb_embeddings[sample["leader_id"]]              # (16,)
    ctx           = sample["context"].copy()                        # (5,)
    ctx[:3]      /= 12.0
    pol_emb       = sample["pol_emb"]                               # (24,)
    pol_act       = sample["pol_actions_norm"]                      # (5,)
    time_f        = sample["time_feats"]                            # (6,)

    return np.concatenate([pca_feats, cb_emb, ctx, pol_emb, pol_act, time_f]).astype(np.float32)


# ─── Model Architecture ─────────────────────────────────────────────────────

if TORCH_OK:
    class MultiLeaderDecisionTransformerV4(nn.Module):
        """
        v4: Larger model (d_model=128, 4 layers, nhead=8) with PCA all-data input.
        """

        def __init__(self, n_features=TOTAL_DIM, d_model=128,
                     nhead=8, n_layers=4, dropout=0.1):
            super().__init__()
            self.n_features = n_features
            self.d_model    = d_model

            self.input_proj = nn.Linear(1, d_model)
            self.pos_emb = nn.Embedding(n_features, d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            self.direction_head = nn.Linear(d_model, 3)
            self.magnitude_head = nn.Linear(d_model, 5)
            self.guidance_head  = nn.Linear(d_model, 3)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            B, N = x.shape
            tok = self.input_proj(x.unsqueeze(-1))
            pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            tok = tok + self.pos_emb(pos_ids)
            enc    = self.encoder(tok)
            pooled = enc.mean(dim=1)
            return (
                self.direction_head(pooled),
                self.magnitude_head(pooled),
                self.guidance_head(pooled),
            )

        def predict(self, x):
            self.eval()
            with torch.no_grad():
                d, m, g = self.forward(x)
                return (
                    torch.argmax(d, dim=-1),
                    torch.argmax(m, dim=-1),
                    torch.argmax(g, dim=-1),
                )

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(X, y_dir, y_mag, y_guid, n_epochs=300, lr=3e-3,
                batch_size=256, verbose=True):
    n_samples = len(X)
    if verbose:
        print(f"\n  Training on {device}  "
              f"(samples={n_samples}, features={X.shape[1]}, batch={batch_size})")

    model = MultiLeaderDecisionTransformerV4(n_features=X.shape[1])
    if verbose:
        print(f"  Parameters: {model.count_parameters():,}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dir_counts = np.bincount(y_dir, minlength=3).astype(float)
    dir_counts = np.where(dir_counts == 0, 1, dir_counts)
    dir_weights = torch.tensor(1.0 / dir_counts, dtype=torch.float32).to(device)
    dir_weights /= dir_weights.sum()

    criterion_dir  = nn.CrossEntropyLoss(weight=dir_weights)
    criterion_mag  = nn.CrossEntropyLoss()
    criterion_guid = nn.CrossEntropyLoss()

    Xt  = torch.tensor(X,      dtype=torch.float32).to(device)
    yd  = torch.tensor(y_dir,  dtype=torch.long).to(device)
    ym  = torch.tensor(y_mag,  dtype=torch.long).to(device)
    yg  = torch.tensor(y_guid, dtype=torch.long).to(device)

    history = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for start_i in range(0, n_samples, batch_size):
            idx = perm[start_i:start_i + batch_size]
            xb  = Xt[idx]
            ydb = yd[idx]
            ymb = ym[idx]
            ygb = yg[idx]

            optimizer.zero_grad()
            d_logits, m_logits, g_logits = model(xb)
            loss = (criterion_dir(d_logits, ydb)
                    + 0.5 * criterion_mag(m_logits, ymb)
                    + 0.3 * criterion_guid(g_logits, ygb))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                d_all, _, _ = model(Xt)
                d_pred = torch.argmax(d_all, dim=-1)
                acc = (d_pred == yd).float().mean().item()
            avg_loss = epoch_loss / max(n_batches, 1)
            if verbose:
                print(f"  epoch {epoch:3d}/{n_epochs}  "
                      f"loss={avg_loss:.4f}  dir_acc={acc:.3f}  "
                      f"lr={scheduler.get_last_lr()[0]:.5f}")
            history.append({"epoch": epoch, "loss": avg_loss, "dir_acc": acc})

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training completed in {elapsed:.1f}s  "
              f"({elapsed/n_epochs*1000:.1f}ms/epoch)")

    return model, history, elapsed


# ─── Leave-One-Leader-Out Evaluation ────────────────────────────────────────

def loo_eval_by_leader(samples, cb_embeddings, econ, pca_pipeline_full, verbose=True):
    """Leave-One-Leader-Out cross-validation with per-fold PCA refit."""
    print("\n[5] Leave-One-Leader-Out (Cross-Leader Generalization)...")

    by_leader = defaultdict(list)
    for s in samples:
        by_leader[s["leader_id"]].append(s)

    eval_leaders = [lid for lid, slist in by_leader.items() if len(slist) >= 5]
    lolo_results = {}

    # Get month->index mapping
    month_to_idx = pca_pipeline_full._month_to_idx

    for test_leader in eval_leaders:
        test_samps  = by_leader[test_leader]
        train_samps = [s for lid, slist in by_leader.items()
                       if lid != test_leader for s in slist]

        if len(train_samps) < 10:
            continue

        # Refit PCA on training months only
        train_months = set(s["month"] for s in train_samps)
        train_month_indices = sorted([month_to_idx[m] for m in train_months if m in month_to_idx])

        pca_fold = PCAEconomicPipeline(n_components=N_PCA_COMPONENTS)
        pca_fold.set_matrix(pca_pipeline_full._matrix, pca_pipeline_full._series_names,
                            pca_pipeline_full._months)
        pca_fold.fit(train_month_indices)

        # Normalize political actions on training set
        pol_norm = PoliticalActionNormalizer()
        pol_norm.fit(np.array([s["pol_actions"] for s in train_samps]))
        for s in train_samps + test_samps:
            s["pol_actions_norm"] = pol_norm.transform(s["pol_actions"])

        X_train  = np.array([assemble_feature_vector(s, cb_embeddings, pca_fold) for s in train_samps])
        y_dir_tr = np.array([s["direction"] for s in train_samps])
        y_mag_tr = np.array([s["magnitude"] for s in train_samps])
        y_guid_tr= np.array([s["guidance"]  for s in train_samps])

        X_test   = np.array([assemble_feature_vector(s, cb_embeddings, pca_fold) for s in test_samps])
        y_dir_te = np.array([s["direction"] for s in test_samps])
        y_mag_te = np.array([s["magnitude"] for s in test_samps])

        model, _, _ = train_model(X_train, y_dir_tr, y_mag_tr, y_guid_tr,
                                  n_epochs=150, verbose=False)

        model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            d_pred, m_pred, _ = model.predict(Xt)
        d_pred = d_pred.cpu().numpy()
        m_pred = m_pred.cpu().numpy()

        dir_acc      = float(np.mean(d_pred == y_dir_te))
        mag_acc      = float(np.mean(m_pred == y_mag_te))
        combined_acc = float(np.mean((d_pred == y_dir_te) & (m_pred == y_mag_te)))

        turning_pts = [i for i in range(1, len(test_samps))
                       if test_samps[i-1]["direction"] != test_samps[i]["direction"]]
        tp_acc = None
        if turning_pts:
            tp_correct = sum(1 for i in turning_pts if d_pred[i] == y_dir_te[i])
            tp_acc = tp_correct / len(turning_pts)

        lolo_results[test_leader] = {
            "dir_acc":      dir_acc,
            "mag_acc":      mag_acc,
            "combined_acc": combined_acc,
            "tp_acc":       tp_acc,
            "n_test":       len(test_samps),
            "n_turning_pts":len(turning_pts),
            "country":      by_leader[test_leader][0]["country"],
        }

        if verbose:
            tp_str = f"TP={tp_acc:.3f}({len(turning_pts)})" if tp_acc is not None else "TP=N/A"
            print(f"  LOLO {test_leader:20s}: dir={dir_acc:.3f}  mag={mag_acc:.3f}  "
                  f"comb={combined_acc:.3f}  {tp_str}  n={len(test_samps)}")

    return lolo_results


def loo_eval_within_leader(leader_id, samples, cb_embeddings, pca_pipeline_full, verbose=True):
    """Standard LOO within a single leader. PCA fitted once on all OTHER leaders' months."""
    leader_samps = [s for s in samples if s["leader_id"] == leader_id]
    other_samps  = [s for s in samples if s["leader_id"] != leader_id]
    if len(leader_samps) < 5:
        return None

    if verbose:
        print(f"\n[6] Within-leader LOO for {leader_id} (n={len(leader_samps)})...")

    month_to_idx = pca_pipeline_full._month_to_idx

    # Fit PCA ONCE on all other leaders' months (no data leakage — leader's months excluded)
    other_months = set(s["month"] for s in other_samps)
    other_month_indices = sorted([month_to_idx[m] for m in other_months if m in month_to_idx])
    pca_loo = PCAEconomicPipeline(n_components=N_PCA_COMPONENTS)
    pca_loo.set_matrix(pca_pipeline_full._matrix, pca_pipeline_full._series_names,
                       pca_pipeline_full._months)
    pca_loo.fit(other_month_indices)

    # Pre-normalize political actions once
    pol_norm = PoliticalActionNormalizer()
    pol_norm.fit(np.array([s["pol_actions"] for s in other_samps]))
    for s in leader_samps + other_samps:
        s["pol_actions_norm"] = pol_norm.transform(s["pol_actions"])

    all_pred_dir = []
    all_true_dir = []
    all_pred_mag = []
    all_true_mag = []

    for test_idx in range(len(leader_samps)):
        train_samps = other_samps + [s for i, s in enumerate(leader_samps) if i != test_idx]
        test_samp   = leader_samps[test_idx]

        X_train  = np.array([assemble_feature_vector(s, cb_embeddings, pca_loo) for s in train_samps])
        y_dir_tr = np.array([s["direction"] for s in train_samps])
        y_mag_tr = np.array([s["magnitude"] for s in train_samps])
        y_guid_tr= np.array([s["guidance"]  for s in train_samps])
        X_test   = np.array([assemble_feature_vector(test_samp, cb_embeddings, pca_loo)])

        model, _, _ = train_model(X_train, y_dir_tr, y_mag_tr, y_guid_tr,
                                  n_epochs=150, verbose=False)
        model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            d_pred, m_pred, _ = model.predict(Xt)

        all_pred_dir.append(int(d_pred[0]))
        all_true_dir.append(test_samp["direction"])
        all_pred_mag.append(int(m_pred[0]))
        all_true_mag.append(test_samp["magnitude"])

    dir_acc = float(np.mean(np.array(all_pred_dir) == np.array(all_true_dir)))
    mag_acc = float(np.mean(np.array(all_pred_mag) == np.array(all_true_mag)))

    turning_pts = [i for i in range(1, len(leader_samps))
                   if leader_samps[i-1]["direction"] != leader_samps[i]["direction"]]
    tp_acc = None
    if turning_pts:
        tp_correct = sum(1 for i in turning_pts
                         if all_pred_dir[i] == all_true_dir[i])
        tp_acc = tp_correct / len(turning_pts)

    if verbose:
        tp_str = f"{tp_acc:.3f} ({len(turning_pts)} pts)" if tp_acc is not None else "N/A"
        print(f"  {leader_id} LOO: dir_acc={dir_acc:.3f}  mag_acc={mag_acc:.3f}  "
              f"turning_pts={tp_str}")

    return {
        "dir_acc":        dir_acc,
        "mag_acc":        mag_acc,
        "tp_acc":         tp_acc,
        "n_samples":      len(leader_samps),
        "n_turning_pts":  len(turning_pts),
        "all_pred_dir":   all_pred_dir,
        "all_true_dir":   all_true_dir,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    if not TORCH_OK:
        print("[ERROR] PyTorch required for model training")
        return

    # Step 0: Load ALL economic data
    econ = AllEconomicDataStore()

    # Step 1: Build monthly matrix for PCA
    print("\n[1] Building monthly matrix for PCA...")
    matrix, series_names, all_months = econ.build_monthly_matrix()
    print(f"  Matrix shape: {matrix.shape} (months × series)")

    # Check data coverage
    non_nan_pct = np.mean(~np.isnan(matrix)) * 100
    print(f"  Non-NaN coverage: {non_nan_pct:.1f}%")

    # Step 2: Build political embeddings
    print("\n[2] Building political leader embeddings...")
    pol_embeddings = build_political_embeddings()

    test_months = ["2008-01", "2013-01", "2018-06", "2020-04", "2022-01", "2025-06"]
    print("\n  Political leaders active at key months:")
    for m in test_months:
        active = get_active_political_leaders(m)
        print(f"  {m}: US={active['us_president'] or 'none':10s} "
              f"XI={active['xi'] or 'none':12s} "
              f"PUTIN={active['putin'] or 'none'}")

    # Step 3: Build dataset (without PCA features yet)
    samples, cb_embeddings, stats = build_multi_leader_dataset(econ, pol_embeddings)

    if len(samples) < 20:
        print("[ERROR] Insufficient samples")
        return

    # Step 4: Fit PCA on ALL training months (for final model)
    print("\n[3] Fitting PCA on all sample months...")
    pca_pipeline = PCAEconomicPipeline(n_components=N_PCA_COMPONENTS)
    pca_pipeline.set_matrix(matrix, series_names, all_months)

    # Use all months that appear in samples for PCA fitting
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    all_sample_months = set(s["month"] for s in samples)
    train_month_indices = sorted([month_to_idx[m] for m in all_sample_months if m in month_to_idx])
    explained = pca_pipeline.fit(train_month_indices)

    # Step 5: Normalize political action signals
    print("\n[4] Normalizing political action signals...")
    pol_norm = PoliticalActionNormalizer()
    pol_norm.fit(np.array([s["pol_actions"] for s in samples]))
    for s in samples:
        s["pol_actions_norm"] = pol_norm.transform(s["pol_actions"])

    # Step 6: Train final model
    print("\n[5] Training final model on ALL samples...")
    X     = np.array([assemble_feature_vector(s, cb_embeddings, pca_pipeline) for s in samples])
    y_dir = np.array([s["direction"] for s in samples])
    y_mag = np.array([s["magnitude"] for s in samples])
    y_guid= np.array([s["guidance"]  for s in samples])

    actual_dim = X.shape[1]
    print(f"  Actual feature dimension: {actual_dim}")

    model, history, train_time = train_model(X, y_dir, y_mag, y_guid, n_epochs=400)

    # In-sample accuracy
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        d_pred, m_pred, _ = model.predict(Xt)
    d_pred_np = d_pred.cpu().numpy()
    m_pred_np = m_pred.cpu().numpy()

    train_dir_acc = float(np.mean(d_pred_np == y_dir))
    train_mag_acc = float(np.mean(m_pred_np == y_mag))
    print(f"  Training dir_acc={train_dir_acc:.3f}  mag_acc={train_mag_acc:.3f}  "
          f"({train_time:.1f}s on {device})")

    # Step 7: Per-leader in-sample accuracy
    print("\n[7] Per-leader in-sample accuracy...")
    by_leader = defaultdict(list)
    for i, s in enumerate(samples):
        by_leader[s["leader_id"]].append(i)

    per_leader = {}
    for lid, indices in sorted(by_leader.items()):
        y_d_sub = y_dir[indices]
        y_m_sub = y_mag[indices]
        d_p_sub = d_pred_np[indices]
        m_p_sub = m_pred_np[indices]
        per_leader[lid] = {
            "dir_acc":   float(np.mean(d_p_sub == y_d_sub)),
            "mag_acc":   float(np.mean(m_p_sub == y_m_sub)),
            "n_samples": len(indices),
            "country":   samples[indices[0]]["country"],
        }

    print(f"\n{'Leader':22s} {'Country':7s} {'N':6s} {'Dir%':8s} {'Mag%':8s}")
    print("-" * 55)
    for lid, res in sorted(per_leader.items(), key=lambda x: x[0]):
        print(f"  {lid:20s} {res['country']:7s} {res['n_samples']:6d} "
              f"{res['dir_acc']*100:.1f}%    {res['mag_acc']*100:.1f}%")

    total_dir_acc = float(np.mean([r["dir_acc"] for r in per_leader.values()]))
    total_mag_acc = float(np.mean([r["mag_acc"] for r in per_leader.values()]))
    print(f"\n  Overall (macro-avg): dir={total_dir_acc:.3f}  mag={total_mag_acc:.3f}")

    # Step 8: Powell within-leader LOO
    powell_loo = loo_eval_within_leader("powell", samples, cb_embeddings, pca_pipeline)

    # Step 9: Leave-One-Leader-Out generalization
    lolo_results = loo_eval_by_leader(samples, cb_embeddings, econ, pca_pipeline)

    # Step 10: Save model
    model_path = MODELS_DIR / "multi_leader_v4_pca.pt"
    torch.save({
        "model_state_dict":  model.state_dict(),
        "n_features":        actual_dim,
        "n_pca":             N_PCA_COMPONENTS,
        "n_leader_emb":      LEADER_EMB_DIM,
        "n_context":         N_CONTEXT_FEATURES,
        "n_pol_emb":         N_POL_SLOTS * POL_EMB_DIM,
        "n_pol_action":      N_POL_ACTION,
        "n_time":            N_TIME_FEATURES,
        "cb_embeddings":     {k: v.tolist() for k, v in cb_embeddings.items()},
        "pol_embeddings":    {k: v.tolist() for k, v in pol_embeddings.items()},
        "pol_action_norm":   pol_norm.to_dict(),
        "pca_explained_variance": explained,
        "pca_n_series":      len(series_names),
        "history":           history,
        "device_used":       str(device),
        "architecture":      {
            "d_model": 128, "nhead": 8, "n_layers": 4,
            "total_dim": actual_dim,
            "pol_slots": N_POL_SLOTS,
        },
    }, model_path)
    print(f"\n[10] Model saved: {model_path}")

    # ─── Comparison with v3/v1 ────────────────────────────────────────────────
    V1_POWELL_LOO_DIR  = 0.6146
    V1_POWELL_LOO_TP   = 0.3939
    V3_POWELL_LOO_DIR  = 0.758   # v3 result
    V3_POWELL_LOO_TP   = 0.485   # v3 result
    V3_AVG_LOLO_DIR    = 0.4879

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY  (v4 PCA All-Data vs v3/v1)")
    print("=" * 70)

    print(f"\n  Architecture:")
    print(f"    v1: d_model=32, nhead=4, 2 layers, 31D input  (10 hand-picked econ)")
    print(f"    v3: d_model=64, nhead=8, 3 layers, 66D input  (10 hand-picked econ)")
    print(f"    v4: d_model=128, nhead=8, 4 layers, {actual_dim}D input  "
          f"({N_PCA_COMPONENTS} PCA from {len(series_names)} series)  [{device}]")
    print(f"    GPU training time: {train_time:.1f}s")

    print(f"\n  Feature breakdown (v4):")
    print(f"    PCA economic state:   {N_PCA_COMPONENTS}D (from {len(series_names)} series, "
          f"{explained:.1f}% variance explained)")
    print(f"    CB leader embedding:  {LEADER_EMB_DIM}D")
    print(f"    Context (streak):     {N_CONTEXT_FEATURES}D")
    print(f"    US president:         {POL_EMB_DIM}D")
    print(f"    Xi Jinping:           {POL_EMB_DIM}D")
    print(f"    Putin:                {POL_EMB_DIM}D")
    print(f"    Political actions:    {N_POL_ACTION}D")
    print(f"    Time features:        {N_TIME_FEATURES}D")
    print(f"    Total:                {actual_dim}D")

    print(f"\n  POWELL LOO (within-leader):")
    if powell_loo:
        v4_dir = powell_loo["dir_acc"]
        v4_tp  = powell_loo.get("tp_acc")
        delta_dir_v3 = (v4_dir - V3_POWELL_LOO_DIR) * 100
        delta_dir_v1 = (v4_dir - V1_POWELL_LOO_DIR) * 100
        print(f"    Direction accuracy: {v4_dir:.3f}  "
              f"({delta_dir_v3:+.1f}pp vs v3 {V3_POWELL_LOO_DIR:.3f}, "
              f"{delta_dir_v1:+.1f}pp vs v1 {V1_POWELL_LOO_DIR:.3f})")
        if v4_tp is not None:
            delta_tp_v3 = (v4_tp - V3_POWELL_LOO_TP) * 100
            delta_tp_v1 = (v4_tp - V1_POWELL_LOO_TP) * 100
            print(f"    Turning point acc:  {v4_tp:.3f}  "
                  f"({delta_tp_v3:+.1f}pp vs v3 {V3_POWELL_LOO_TP:.3f}, "
                  f"{delta_tp_v1:+.1f}pp vs v1 {V1_POWELL_LOO_TP:.3f})")
    else:
        print("    Powell LOO failed")

    print(f"\n  IN-SAMPLE (all leaders):")
    print(f"    v4 dir acc: {total_dir_acc:.3f}  mag acc: {total_mag_acc:.3f}")

    print(f"\n  CROSS-LEADER GENERALIZATION (LOLO):")
    if lolo_results:
        avg_v4_dir = float(np.mean([r["dir_acc"] for r in lolo_results.values()]))
        tp_vals = [r["tp_acc"] for r in lolo_results.values() if r["tp_acc"] is not None]
        avg_v4_tp  = float(np.mean(tp_vals)) if tp_vals else 0.0
        delta_lolo = (avg_v4_dir - V3_AVG_LOLO_DIR) * 100
        print(f"    Average LOLO dir:  {avg_v4_dir:.3f}  "
              f"({delta_lolo:+.1f}pp vs v3 {V3_AVG_LOLO_DIR:.3f})")
        print(f"    Average LOLO TP:   {avg_v4_tp:.3f}")
        print()
        for lid, r in sorted(lolo_results.items()):
            tp_str = f"{r['tp_acc']:.3f}" if r["tp_acc"] is not None else " N/A"
            print(f"    {lid:22s}: dir={r['dir_acc']:.3f}  tp={tp_str}")

    print(f"\n  Total elapsed: {time.time() - t_start:.1f}s")
    print("=" * 70)

    # ─── Save evaluation ─────────────────────────────────────────────────────
    eval_result = {
        "model":            "multi_leader_v4_pca",
        "generated_at":     datetime.now().isoformat(),
        "training_samples": len(samples),
        "device":           str(device),
        "architecture": {
            "d_model":    128,
            "nhead":      8,
            "n_layers":   4,
            "total_dim":  actual_dim,
            "n_pca":      N_PCA_COMPONENTS,
            "n_cb_emb":   LEADER_EMB_DIM,
            "n_context":  N_CONTEXT_FEATURES,
            "n_pol_emb":  N_POL_SLOTS * POL_EMB_DIM,
            "n_pol_act":  N_POL_ACTION,
            "n_time":     N_TIME_FEATURES,
        },
        "pca_info": {
            "n_input_series": len(series_names),
            "n_components":   N_PCA_COMPONENTS,
            "explained_variance_pct": explained,
        },
        "v3_comparison": {
            "v3_powell_loo_dir":  V3_POWELL_LOO_DIR,
            "v3_powell_loo_tp":   V3_POWELL_LOO_TP,
            "v3_avg_lolo_dir":    V3_AVG_LOLO_DIR,
        },
        "dataset_stats": {
            lid: {
                "n_samples":  s["n_samples"],
                "n_cut":      s["n_cut"],
                "n_hold":     s["n_hold"],
                "n_hike":     s["n_hike"],
                "country":    s["country"],
                "has_profile":s["has_profile"],
            }
            for lid, s in stats.items()
        },
        "per_leader_in_sample_accuracy": per_leader,
        "overall_in_sample_accuracy": {
            "direction": total_dir_acc,
            "magnitude": total_mag_acc,
        },
        "powell_loo":    powell_loo,
        "lolo_results":  lolo_results,
        "gpu_train_time_seconds": train_time,
        "elapsed_seconds": time.time() - t_start,
    }

    eval_path = OUTPUT_DIR / "multi_leader_v4_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2, default=str)
    print(f"\n[11] Evaluation saved: {eval_path}")


if __name__ == "__main__":
    import os
    os.chdir(BASE)
    main()
