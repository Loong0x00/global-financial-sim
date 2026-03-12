#!/usr/bin/env python3
"""
Multi-Leader Transformer Decision Model  v3 — Time Dimension
=============================================================
Extends v2 by adding temporal features so the model understands
WHEN a decision happens (tenure progress, era, election proximity).

Key insight: Central bank decisions don't happen in isolation.
  - Powell's rate decisions depend on Trump/Biden policy stance
  - Xi's economic policies affect global conditions
  - Putin's energy weapon changes the inflation picture

New input vector (~52D total):
  [economic_state 10D]            same as v1
  [central_banker_embedding 16D]  same as v1
  [context 5D]                    same as v1
  [us_president_embedding 8D]     Trump T1/Biden/Trump T2 (zero pre-2009)
  [xi_jinping_embedding 8D]       active 2012-11 onwards (zero before)
  [putin_embedding 8D]            active 2000-05 onwards
  [political_action_signals 5D]   trade_delta, fiscal_delta, vix, credit, sanction_proxy

Architecture upgrade (GPU allows bigger model):
  d_model=64, nhead=8, 3 layers (was d_model=32, nhead=4, 2 layers)

Political leaders covered:
  US Presidents: Trump T1 (2017-01 to 2021-01), Biden (2021-01 to 2025-01),
                 Obama (2009-01 to 2017-01), Trump T2 (2025-01 to present)
  China:         Xi Jinping (2012-11 to present)
  Russia:        Putin (2000-05 to present)
"""

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

print("=" * 70)
print("MULTI-LEADER TRANSFORMER DECISION MODEL  v3 — TIME DIMENSION")
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
PROFILES    = DATA / "leaders" / "profiles"
TIMELINES   = DATA / "leaders" / "timelines"
MODELS_DIR  = BASE / "models" / "decision_functions"
OUTPUT_DIR  = BASE / "output" / "decision_function"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Central Bank Leader Registry ─────────────────────────────────────────────
# (leader_id, country, profile_filename, start_yyyymm, end_yyyymm_or_None)
LEADER_REGISTRY = [
    # US Fed Chairs
    ("greenspan",       "US", "greenspan",    "1987-08", "2006-01"),
    ("bernanke",        "US", "bernanke",     "2006-02", "2014-01"),
    ("yellen",          "US", "yellen",       "2014-02", "2018-01"),
    ("powell",          "US", "powell",       "2018-02", None),
    # ECB Presidents
    ("draghi",          "EU", "draghi",       "2011-11", "2019-10"),
    ("lagarde",         "EU", "lagarde",      "2019-11", None),
    # Russia CBR
    ("nabiullina",      "RU", "nabiullina",   "2013-06", None),
    # China PBOC
    ("zhou_xiaochuan",  "CN", "zhou_xiaochuan","2002-12", "2018-03"),
    # Japan BOJ
    ("kuroda",          "JP", None,           "2013-03", "2023-04"),
    ("ueda",            "JP", None,           "2023-04", None),
    # UK BOE
    ("carney_boe",      "UK", None,           "2013-07", "2020-03"),
    ("bailey_boe",      "UK", None,           "2020-03", None),
]

# ─── Political Leader Registry ────────────────────────────────────────────────
# (leader_id, profile_filename, start_yyyymm, end_yyyymm_or_None)
# Slot assignment:
#   SLOT_US_PRES  — the active US president (Trump/Obama/Biden)
#   SLOT_XI       — Xi Jinping (fixed slot, zero before 2012-11)
#   SLOT_PUTIN    — Putin (fixed slot, zero before 2000-05)
POLITICAL_REGISTRY = [
    # US Presidents (mutually exclusive, fill SLOT_US_PRES)
    ("obama",      "obama",    "2009-01", "2017-01"),
    ("trump_t1",   "trump",    "2017-01", "2021-01"),
    ("biden",      None,       "2021-01", "2025-01"),   # no profile → default
    ("trump_t2",   "trump",    "2025-01", None),
    # Xi Jinping — always SLOT_XI when active
    ("xi_jinping", "xi_jinping", "2012-11", None),
    # Putin — always SLOT_PUTIN when active
    ("putin",      "putin",    "2000-05", None),
]

# ─── Economic Features ────────────────────────────────────────────────────────
COUNTRY_FEATURES = {
    "US": [
        ("macro",    "cpi_yoy",              "level"),
        ("macro",    "core_cpi_yoy",         "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "fed_funds_rate",       "level"),
        ("financial","yield_curve_10y2y",    "level"),
        ("financial","vix",                  "level"),
        ("financial","sp500",                "yoy"),
        ("macro",    "nonfarm_payrolls_change", "level"),
        ("macro",    "pce_yoy",             "level"),
    ],
    "EU": [
        ("macro",    "hicp_all_items",       "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "eur_usd",              "yoy"),
        ("macro",    "industrial_production","yoy"),
        ("financial","germany_10y_bond",     "level"),
        ("macro",    "m3_money_supply",      "yoy"),
    ],
    "RU": [
        ("macro",    "cpi",                  "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp",                  "level"),
        ("macro",    "exchange_rate_usd_rub","yoy"),
        ("macro",    "industrial_production","yoy"),
        ("macro",    "m2",                   "yoy"),
    ],
    "CN": [
        ("macro",    "cpi_yoy",              "level"),
        ("macro",    "discount_rate",        "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "exports",              "yoy"),
        ("financial","usd_cny",              "yoy"),
        ("financial","short_rate",           "level"),
        ("macro",    "m2",                   "yoy"),
    ],
    "JP": [
        ("macro",    "cpi",                  "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "exchange_rate_usd_jpy","yoy"),
        ("financial","nikkei225",            "yoy"),
        ("macro",    "industrial_production","yoy"),
    ],
    "UK": [
        ("macro",    "cpi",                  "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "gbp_usd",              "yoy"),
        ("macro",    "industrial_production","yoy"),
        ("macro",    "m2_money_supply",      "yoy"),
    ],
}
N_ECON_FEATURES     = 10   # max econ features per country (pad/truncate)
LEADER_EMB_DIM      = 16   # central banker embedding dim
N_CONTEXT_FEATURES  = 5    # hike_streak, cut_streak, hold_streak, month_sin, month_cos
POL_EMB_DIM         = 8    # political leader embedding dim
N_POL_SLOTS         = 3    # US president + Xi + Putin
N_POL_ACTION        = 5    # trade_delta, fiscal_delta, vix, credit_spread, sanction_proxy
N_TIME_FEATURES     = 6    # absolute_year, tenure_progress, months_to_election,
                           # decade_sin, decade_cos, us_president_tenure_progress

TOTAL_DIM = (N_ECON_FEATURES + LEADER_EMB_DIM + N_CONTEXT_FEATURES
             + N_POL_SLOTS * POL_EMB_DIM + N_POL_ACTION + N_TIME_FEATURES)
# = 10 + 16 + 5 + 24 + 5 = 60

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
# Key behavioral dimensions that affect central bank environment
# For each political leader we extract these 8 dimensions.

TRUMP_POL_PARAMS = [
    "fed_political_pressure_intensity",    # how hard they push on CB
    "tariff_as_negotiation_weapon",        # trade aggressiveness
    "fiscal_deficit_tolerance",            # fiscal stimulus tendency
    "china_confrontation_intensity",       # trade war driver
    "stock_market_as_approval_metric",     # care about markets
    "deal_making_maximum_pressure_pattern",# unpredictability
    "supply_side_tax_cut_preference",      # fiscal stance
    "multilateral_institution_respect",    # global coordination
]

XI_POL_PARAMS = [
    "political_security_over_economic_growth",  # priority
    "state_vs_market_balance_preference",       # market orientation (inverted: high=more state)
    "flood_style_stimulus_aversion",            # fiscal restraint
    "targeted_stimulus_preference",             # nuanced policy
    "trade_retaliation_willingness",            # reaction to trade war
    "capital_controls_tightening_willingness",  # capital account management
    "tech_sector_crackdown_intensity",          # regulatory risk
    "dual_circulation_strategy_commitment",     # decoupling tendency
]

PUTIN_POL_PARAMS = [
    "energy_as_geopolitical_weapon_propensity",   # use energy as leverage
    "sanctions_resilience_preparation_premeditation",  # sanctions defiance
    "military_escalation_threshold",              # geopolitical confrontation
    "political_survival_primacy_over_economic_optimality",  # regime priority
    "state_enterprise_preference_over_private",   # state ownership
    "gdp_growth_tolerance_for_geopolitical_goals",# growth sacrifice for geo
    "information_warfare_as_economic_tool",       # information warfare
    "fiscal_conservatism_base_disposition",       # fiscal stance
]

# Default Biden embedding (institutional, moderate, low Fed pressure)
# Biden doesn't have a profile — use reasonable default based on known characteristics
BIDEN_DEFAULT_EMBEDDING = np.array([
    0.10,  # fed_political_pressure_intensity — very low (respected Fed independence)
    0.15,  # tariff_as_negotiation_weapon — low (kept some Trump tariffs but not aggressive)
    0.75,  # fiscal_deficit_tolerance — high (IRA, infrastructure bill)
    0.30,  # china_confrontation_intensity — moderate (tech restrictions, not trade war)
    0.30,  # stock_market_as_approval_metric — low-moderate
    0.20,  # deal_making_maximum_pressure_pattern — low (institutional style)
    0.35,  # supply_side_tax_cut_preference — low (preferred demand-side)
    0.75,  # multilateral_institution_respect — high (rejoined Paris, WHO, etc.)
], dtype=np.float32)

# Obama default embedding (post-GFC fiscal restraint, institutional, low Fed pressure)
OBAMA_DEFAULT_EMBEDDING = np.array([
    0.05,  # fed_political_pressure_intensity — very low
    0.10,  # tariff_as_negotiation_weapon — very low
    0.60,  # fiscal_deficit_tolerance — moderate-high (stimulus 2009, then austerity)
    0.20,  # china_confrontation_intensity — low
    0.25,  # stock_market_as_approval_metric — low
    0.15,  # deal_making_maximum_pressure_pattern — low
    0.25,  # supply_side_tax_cut_preference — low (ACA, Dodd-Frank)
    0.90,  # multilateral_institution_respect — very high (TPP, Paris etc.)
], dtype=np.float32)

# Obama profile-native parameters (actual keys from obama.json)
OBAMA_NATIVE_PARAMS = [
    "multilateral_institution_preference",       # 0.86 — multilateral vs unilateral
    "fiscal_stimulus_economic_multiplier_belief",# 0.79 — fiscal stimulus stance
    "geopolitical_risk_aversion_military",       # 0.78 — military risk aversion
    "qe_coordination_bernanke_yellen",           # 0.72 — Fed coordination (low pressure)
    "dont_do_stupid_stuff_risk_aversion",        # 0.84 — caution / unpredictability (low)
    "tpp_trade_policy_engagement",               # 0.79 — trade openness
    "financial_sector_regulatory_toughness",     # 0.60 — financial regulation
    "income_inequality_attention",               # 0.76 — fiscal distribution priority
]


# ─── Data Loading Utilities ───────────────────────────────────────────────────

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


class EconomicDataStore:
    """Loads all economic data and provides fast lookup by (country, indicator, month)."""

    def __init__(self):
        self._data = {}        # (country, source, indicator) -> {month: value}
        self._bis_rates = {}   # country -> {month: value}
        self._load_all()

    def _load_all(self):
        for country in ["US", "EU", "RU", "CN", "JP", "UK"]:
            cc = country.lower()
            macro_path = DATA / "economic" / cc / "macro.json"
            if macro_path.exists():
                md = load_json(macro_path)
                for ind_name, ind_data in md.get("indicators", {}).items():
                    self._store("macro", country, ind_name, ind_data.get("series", []))

            fin_path = DATA / "economic" / cc / "financial.json"
            if fin_path.exists():
                fd = load_json(fin_path)
                for ind_name, ind_data in fd.get("indicators", {}).items():
                    self._store("financial", country, ind_name, ind_data.get("series", []))

        # BIS policy rates
        bis_path = DATA / "economic" / "bis" / "policy_rates.json"
        if bis_path.exists():
            bis = load_json(bis_path)
            country_map = {"US": "US", "EU": None, "RU": "RU",
                           "CN": "CN", "JP": "JP", "UK": "GB"}
            for our_country, bis_code in country_map.items():
                if bis_code and bis_code in bis.get("countries", {}):
                    ser_data = bis["countries"][bis_code]["series"]
                    for sk, sv in ser_data.items():
                        monthly = {k: v for k, v in sv["data"].items() if len(k) == 7}
                        self._bis_rates[our_country] = monthly
                        break

    def _store(self, source, country, ind_name, series):
        if not series:
            return
        key = (country, source, ind_name)
        d = {}
        for item in series:
            if isinstance(item, dict) and "date" in item and item.get("value") is not None:
                d[item["date"]] = float(item["value"])
        self._data[key] = d

    def get_rate(self, country, month):
        if country == "EU":
            d = self._data.get(("EU", "macro", "interest_rate"), {})
            return d.get(month)
        elif country in self._bis_rates:
            return self._bis_rates[country].get(month)
        return None

    def get(self, source, country, indicator, month):
        d = self._data.get((country, source, indicator), {})
        return d.get(month)

    def get_yoy(self, source, country, indicator, month):
        y, m = int(month[:4]), int(month[5:])
        prev_month = f"{y-1:04d}-{m:02d}"
        curr = self.get(source, country, indicator, month)
        prev = self.get(source, country, indicator, prev_month)
        if curr is not None and prev is not None and prev != 0:
            return (curr - prev) / abs(prev) * 100.0
        return None

    def get_prev_month(self, month):
        y, m = int(month[:4]), int(month[5:])
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        return f"{y:04d}-{m:02d}"

    def all_months_range(self, start_yyyymm, end_yyyymm=None):
        months = []
        y, m = int(start_yyyymm[:4]), int(start_yyyymm[5:])
        if end_yyyymm is None:
            end_yyyymm = "2026-02"
        ey, em = int(end_yyyymm[:4]), int(end_yyyymm[5:])
        while (y, m) <= (ey, em):
            months.append(f"{y:04d}-{m:02d}")
            m += 1
            if m > 12:
                m = 1
                y += 1
        return months


# ─── Rate Decision Extraction ─────────────────────────────────────────────────

def classify_rate_change(delta_bp):
    if delta_bp < -5:
        direction = 0   # cut
    elif delta_bp > 5:
        direction = 2   # hike
    else:
        direction = 1   # hold

    abs_bp = abs(delta_bp)
    if abs_bp < 12.5:
        magnitude = 0
    elif abs_bp < 37.5:
        magnitude = 1   # 25bp
    elif abs_bp < 62.5:
        magnitude = 2   # 50bp
    elif abs_bp < 87.5:
        magnitude = 3   # 75bp
    else:
        magnitude = 4   # 100bp+

    return direction, magnitude


def extract_rate_decisions(econ, leader_id, country, start_yyyymm, end_yyyymm=None):
    months = econ.all_months_range(start_yyyymm, end_yyyymm)
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


# ─── Economic State Vector ────────────────────────────────────────────────────

def extract_econ_state(econ, country, month):
    features_spec = COUNTRY_FEATURES.get(country, COUNTRY_FEATURES["US"])
    vals = []
    for (src, ind, transform) in features_spec:
        if transform == "level":
            v = econ.get(src, country, ind, month)
        elif transform == "yoy":
            v = econ.get_yoy(src, country, ind, month)
        else:
            v = None
        vals.append(v)

    arr = np.zeros(N_ECON_FEATURES, dtype=np.float32)
    for i, v in enumerate(vals[:N_ECON_FEATURES]):
        arr[i] = float(v) if v is not None else np.nan
    return arr


# ─── Context Features ─────────────────────────────────────────────────────────

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
    """Extract POL_EMB_DIM values from a political leader profile."""
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
            # Partial match fallback
            matches = [v for k, v in lu.items() if k.startswith(label[:12])]
            vals.append(float(np.mean(matches)) if matches else default_val)

    arr = np.array(vals[:POL_EMB_DIM], dtype=np.float32)
    if len(arr) < POL_EMB_DIM:
        arr = np.pad(arr, (0, POL_EMB_DIM - len(arr)), constant_values=default_val)
    return arr


def build_political_embeddings():
    """
    Build fixed political leader embeddings.
    Returns dict: {leader_id: np.array(POL_EMB_DIM)}
    """
    embeddings = {}

    # Trump (same profile used for T1 and T2)
    trump_emb = _extract_pol_embedding_from_profile("trump", TRUMP_POL_PARAMS)
    embeddings["trump_t1"] = trump_emb
    embeddings["trump_t2"] = trump_emb

    # Biden — no profile, use hardcoded default
    embeddings["biden"] = BIDEN_DEFAULT_EMBEDDING.copy()

    # Obama — use profile-native parameters (actual keys from obama.json)
    obama_emb = _extract_pol_embedding_from_profile("obama", OBAMA_NATIVE_PARAMS)
    # If profile gives mostly defaults (i.e. params not in profile), use our curated default
    if float(np.std(obama_emb)) < 0.05:   # near-uniform → profile didn't have these params
        embeddings["obama"] = OBAMA_DEFAULT_EMBEDDING.copy()
    else:
        embeddings["obama"] = obama_emb

    # Xi Jinping
    embeddings["xi_jinping"] = _extract_pol_embedding_from_profile(
        "xi_jinping", XI_POL_PARAMS)

    # Putin
    embeddings["putin"] = _extract_pol_embedding_from_profile(
        "putin", PUTIN_POL_PARAMS)

    # Zero embedding for "no political leader active"
    embeddings["zero"] = np.zeros(POL_EMB_DIM, dtype=np.float32)

    return embeddings


def _month_in_range(month, start, end):
    """Check if month is in [start, end). end=None means open."""
    if month < start:
        return False
    if end is not None and month >= end:
        return False
    return True


def get_active_political_leaders(month):
    """
    For a given month, return the active political leaders for each slot.
    Returns: {
        'us_president': leader_id or None,
        'xi': 'xi_jinping' or None,
        'putin': 'putin' or None,
    }
    """
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
    """
    Return concatenated political embeddings for a given month.
    Shape: (N_POL_SLOTS * POL_EMB_DIM,) = (24,)

    Slot order:
      0: US president  (8D)
      1: Xi Jinping    (8D)
      2: Putin         (8D)
    """
    active = get_active_political_leaders(month)

    us_emb  = pol_embeddings.get(active["us_president"], pol_embeddings["zero"])
    xi_emb  = pol_embeddings.get(active["xi"],           pol_embeddings["zero"])
    put_emb = pol_embeddings.get(active["putin"],        pol_embeddings["zero"])

    return np.concatenate([us_emb, xi_emb, put_emb]).astype(np.float32)


# ─── Time Features (v3 NEW) ──────────────────────────────────────────────────

# US election years for distance-to-election calculation
US_ELECTION_YEARS = [1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000,
                     2004, 2008, 2012, 2016, 2020, 2024, 2028]

# US president tenure mapping (start_month, end_month)
US_PRESIDENT_TENURES = [
    ("1969-01", "1974-08"),  # Nixon
    ("1974-08", "1977-01"),  # Ford
    ("1977-01", "1981-01"),  # Carter
    ("1981-01", "1989-01"),  # Reagan
    ("1989-01", "1993-01"),  # H.W. Bush
    ("1993-01", "2001-01"),  # Clinton
    ("2001-01", "2009-01"),  # W. Bush
    ("2009-01", "2017-01"),  # Obama
    ("2017-01", "2021-01"),  # Trump T1
    ("2021-01", "2025-01"),  # Biden
    ("2025-01", "2029-01"),  # Trump T2
]


def extract_time_features(month, cb_start, cb_end):
    """
    6D temporal feature vector:
      [0] absolute_year_norm: (year - 1970) / 60, in [0, ~1]
      [1] cb_tenure_progress: how far into this CB chair's term [0, 1]
      [2] months_to_us_election: proximity to next US election, [0, 1] (0=election month)
      [3] decade_sin: sin(2π * year/10) — cyclical decade position
      [4] decade_cos: cos(2π * year/10) — cyclical decade position
      [5] us_president_tenure_progress: how far into US president's term [0, 1]
    """
    import math
    features = np.zeros(N_TIME_FEATURES, dtype=np.float32)

    yr = int(month[:4])
    mo = int(month[5:7])
    month_abs = yr * 12 + mo

    # [0] Absolute year normalized
    features[0] = (yr - 1970) / 60.0

    # [1] CB tenure progress
    if cb_start:
        s_yr, s_mo = int(cb_start[:4]), int(cb_start[5:7])
        start_abs = s_yr * 12 + s_mo
        if cb_end:
            e_yr, e_mo = int(cb_end[:4]), int(cb_end[5:7])
            end_abs = e_yr * 12 + e_mo
        else:
            end_abs = 2026 * 12 + 3  # current
        total = max(end_abs - start_abs, 1)
        progress = (month_abs - start_abs) / total
        features[1] = float(np.clip(progress, 0, 1))

    # [2] Months to next US election (November of election year)
    for ey in US_ELECTION_YEARS:
        elec_abs = ey * 12 + 11  # November
        if elec_abs >= month_abs:
            dist = elec_abs - month_abs
            features[2] = float(np.clip(1.0 - dist / 48.0, 0, 1))  # 0=far, 1=election month
            break

    # [3,4] Decade cyclical encoding
    decade_phase = (yr + mo / 12.0) / 10.0
    features[3] = math.sin(2 * math.pi * decade_phase)
    features[4] = math.cos(2 * math.pi * decade_phase)

    # [5] US president tenure progress
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


# ─── Political Action Signals ─────────────────────────────────────────────────

def extract_political_action_signals(econ, month):
    """
    5D vector of economic signals that reflect political decisions:
      [0] US trade balance delta (YoY change — trade aggressiveness proxy)
      [1] US fiscal deficit proxy (M2 growth — fiscal stimulus proxy)
      [2] VIX (policy uncertainty / market stress)
      [3] US-RU trade proxy (sanctions intensity — RU exchange rate YoY)
      [4] Global credit conditions (US yield spread or risk proxy)

    All values are normalized loosely to [-1, 1] or [0, 1] range.
    Missing data → 0 (neutral).
    """
    signals = np.zeros(N_POL_ACTION, dtype=np.float32)

    # [0] Trade aggressiveness proxy: YoY change in US exports (negative = trade war)
    us_exports_yoy = econ.get_yoy("macro", "US", "nonfarm_payrolls_change", month)
    if us_exports_yoy is None:
        # fallback: global trade proxy via CN exports
        us_exports_yoy = econ.get_yoy("macro", "CN", "exports", month)
    if us_exports_yoy is not None:
        signals[0] = float(np.clip(us_exports_yoy / 20.0, -1.5, 1.5))

    # [1] Fiscal stance proxy: US M2 YoY growth (high = fiscal stimulus)
    # Use EU M3 as fallback for non-US contexts
    m2_yoy = econ.get_yoy("macro", "US", "fed_funds_rate", month)  # proxy via rate context
    # Better: use actual M2 if available
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

    # [2] VIX — policy uncertainty
    vix = econ.get("financial", "US", "vix", month)
    if vix is not None:
        # Normalize: VIX 10→low risk=0, VIX 80→extreme=1
        signals[2] = float(np.clip((vix - 10.0) / 60.0, 0.0, 1.5))

    # [3] Sanctions intensity proxy: RU-USD exchange rate YoY
    # Large positive = RUB depreciation = sanctions/conflict impact
    ru_fx_yoy = econ.get_yoy("macro", "RU", "exchange_rate_usd_rub", month)
    if ru_fx_yoy is not None:
        signals[3] = float(np.clip(ru_fx_yoy / 50.0, -1.0, 1.5))

    # [4] Global credit conditions: US yield curve (inverted = tightening = credit stress)
    yield_curve = econ.get("financial", "US", "yield_curve_10y2y", month)
    if yield_curve is not None:
        # Normal: +2 = loose credit; inverted: -1 = tight
        signals[4] = float(np.clip(-yield_curve / 2.0, -1.0, 1.5))

    return signals


# ─── Normalizers ─────────────────────────────────────────────────────────────

class PerCountryNormalizer:
    """Robust z-score normalization per country."""

    def __init__(self):
        self.medians = {}
        self.iqrs    = {}

    def fit(self, country_samples):
        for country, samples in country_samples.items():
            X = np.array([s for s in samples if not np.any(np.isnan(s))])
            if len(X) < 5:
                self.medians[country] = np.zeros(N_ECON_FEATURES)
                self.iqrs[country]    = np.ones(N_ECON_FEATURES)
                continue
            self.medians[country] = np.nanmedian(X, axis=0)
            q75 = np.nanpercentile(X, 75, axis=0)
            q25 = np.nanpercentile(X, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr < 1e-8] = 1.0
            self.iqrs[country] = iqr

    def transform(self, country, vec):
        out = vec.copy()
        med = self.medians.get(country, np.zeros(N_ECON_FEATURES))
        iqr = self.iqrs.get(country, np.ones(N_ECON_FEATURES))
        for i in range(len(out)):
            if np.isnan(out[i]):
                out[i] = 0.0
            else:
                out[i] = (out[i] - med[i]) / iqr[i]
        return np.clip(out, -3, 3)

    def to_dict(self):
        return {
            "medians": {k: v.tolist() for k, v in self.medians.items()},
            "iqrs":    {k: v.tolist() for k, v in self.iqrs.items()},
        }


class PoliticalActionNormalizer:
    """Normalize the political action signals across all months."""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, signal_matrix):
        """signal_matrix: np.array shape (N, N_POL_ACTION)"""
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
    """
    Build the full dataset across all leaders, now with political context.

    Returns:
      samples: list of dicts with all features
      cb_embeddings: {leader_id: np.array(LEADER_EMB_DIM)}
    """
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
            econ_raw = extract_econ_state(econ, country, month)
            ctx      = compute_context(i, decisions, month)

            direction = dec["direction"]
            guidance  = 2 if direction == 2 else (0 if direction == 0 else 1)

            # Skip months with too many NaN
            if np.mean(np.isnan(econ_raw)) > 0.6:
                continue

            # Political embeddings for this month
            pol_emb_vec = extract_political_embeddings_for_month(month, pol_embeddings)

            # Political action signals (raw, normalized later)
            pol_actions = extract_political_action_signals(econ, month)

            # Time features (v3 NEW)
            time_feats = extract_time_features(month, start, end)

            samples.append({
                "leader_id":   lid,
                "country":     country,
                "month":       month,
                "econ_raw":    econ_raw,
                "context":     ctx,
                "pol_emb":     pol_emb_vec,    # (N_POL_SLOTS * POL_EMB_DIM,)
                "pol_actions": pol_actions,    # (N_POL_ACTION,) — raw, normalize later
                "time_feats":  time_feats,     # (N_TIME_FEATURES,) — v3 NEW
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


# ─── Normalization ────────────────────────────────────────────────────────────

def fit_and_normalize(samples):
    """Fit per-country normalizer and political action normalizer, apply to all samples."""
    print("\n[3] Fitting normalizers...")

    country_raw = defaultdict(list)
    for s in samples:
        country_raw[s["country"]].append(s["econ_raw"])

    norm = PerCountryNormalizer()
    norm.fit(country_raw)

    for s in samples:
        s["econ_norm"] = norm.transform(s["country"], s["econ_raw"])

    # Fit political action normalizer
    pol_action_matrix = np.array([s["pol_actions"] for s in samples])
    pol_norm = PoliticalActionNormalizer()
    pol_norm.fit(pol_action_matrix)

    for s in samples:
        s["pol_actions_norm"] = pol_norm.transform(s["pol_actions"])

    print(f"  Normalized {len(samples)} samples across {len(country_raw)} countries")
    return norm, pol_norm


# ─── Feature Assembly ─────────────────────────────────────────────────────────

def assemble_feature_vector(sample, cb_embeddings):
    """
    Combine all features into a single vector.
    Shape: (TOTAL_DIM,) = (66,)

    Layout:
      [0:10]   econ_norm           (10D)
      [10:26]  cb_leader_embedding (16D)
      [26:31]  context             (5D)
      [31:55]  pol_embeddings      (24D = 3 slots × 8D)
      [55:60]  pol_actions_norm    (5D)
      [60:66]  time_features       (6D)  ← v3 NEW
    """
    econ_norm     = sample["econ_norm"]                          # (10,)
    cb_emb        = cb_embeddings[sample["leader_id"]]           # (16,)
    ctx           = sample["context"].copy()                     # (5,)
    ctx[:3]      /= 12.0   # normalize streaks
    pol_emb       = sample["pol_emb"]                            # (24,)
    pol_act       = sample["pol_actions_norm"]                   # (5,)
    time_f        = sample["time_feats"]                         # (6,)  v3 NEW

    return np.concatenate([econ_norm, cb_emb, ctx, pol_emb, pol_act, time_f]).astype(np.float32)


# ─── Model Architecture ───────────────────────────────────────────────────────

if TORCH_OK:
    class MultiLeaderDecisionTransformerV2(nn.Module):
        """
        v2: Larger model (d_model=64, 3 layers, nhead=8) using GPU.
        Same feature-as-token approach, now with political context features.
        """

        def __init__(self, n_features=TOTAL_DIM, d_model=64,
                     nhead=8, n_layers=3, dropout=0.1):
            super().__init__()
            self.n_features = n_features
            self.d_model    = d_model

            # Project each scalar feature to d_model
            self.input_proj = nn.Linear(1, d_model)

            # Learnable position embeddings per feature slot
            self.pos_emb = nn.Embedding(n_features, d_model)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            # Output heads
            self.direction_head = nn.Linear(d_model, 3)   # cut/hold/hike
            self.magnitude_head = nn.Linear(d_model, 5)   # 0/25/50/75/100bp
            self.guidance_head  = nn.Linear(d_model, 3)   # dovish/neutral/hawkish

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            B, N = x.shape
            tok = self.input_proj(x.unsqueeze(-1))        # (B, N, d_model)
            pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            tok = tok + self.pos_emb(pos_ids)
            enc    = self.encoder(tok)                    # (B, N, d_model)
            pooled = enc.mean(dim=1)                      # (B, d_model)
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


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(X, y_dir, y_mag, y_guid, n_epochs=300, lr=3e-3,
                batch_size=256, verbose=True):
    """Train the v2 model on GPU with mini-batches."""
    n_samples = len(X)
    if verbose:
        print(f"\n  Training on {device}  "
              f"(samples={n_samples}, features={X.shape[1]}, batch={batch_size})")

    model = MultiLeaderDecisionTransformerV2(n_features=X.shape[1])
    if verbose:
        print(f"  Parameters: {model.count_parameters():,}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Class weights for direction
    dir_counts = np.bincount(y_dir, minlength=3).astype(float)
    dir_counts = np.where(dir_counts == 0, 1, dir_counts)
    dir_weights = torch.tensor(1.0 / dir_counts, dtype=torch.float32).to(device)
    dir_weights /= dir_weights.sum()

    criterion_dir  = nn.CrossEntropyLoss(weight=dir_weights)
    criterion_mag  = nn.CrossEntropyLoss()
    criterion_guid = nn.CrossEntropyLoss()

    # Move full dataset to GPU once
    Xt  = torch.tensor(X,      dtype=torch.float32).to(device)
    yd  = torch.tensor(y_dir,  dtype=torch.long).to(device)
    ym  = torch.tensor(y_mag,  dtype=torch.long).to(device)
    yg  = torch.tensor(y_guid, dtype=torch.long).to(device)

    history = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()

        # Shuffle indices for mini-batch
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


# ─── Leave-One-Leader-Out Evaluation ──────────────────────────────────────────

def loo_eval_by_leader(samples, cb_embeddings, verbose=True):
    """Leave-One-Leader-Out cross-validation."""
    print("\n[5] Leave-One-Leader-Out (Cross-Leader Generalization)...")

    by_leader = defaultdict(list)
    for s in samples:
        by_leader[s["leader_id"]].append(s)

    eval_leaders = [lid for lid, slist in by_leader.items() if len(slist) >= 5]
    lolo_results = {}

    for test_leader in eval_leaders:
        test_samps  = by_leader[test_leader]
        train_samps = [s for lid, slist in by_leader.items()
                       if lid != test_leader for s in slist]

        if len(train_samps) < 10:
            continue

        X_train  = np.array([assemble_feature_vector(s, cb_embeddings) for s in train_samps])
        y_dir_tr = np.array([s["direction"] for s in train_samps])
        y_mag_tr = np.array([s["magnitude"] for s in train_samps])
        y_guid_tr= np.array([s["guidance"]  for s in train_samps])

        X_test   = np.array([assemble_feature_vector(s, cb_embeddings) for s in test_samps])
        y_dir_te = np.array([s["direction"] for s in test_samps])
        y_mag_te = np.array([s["magnitude"] for s in test_samps])

        model, _, _ = train_model(X_train, y_dir_tr, y_mag_tr, y_guid_tr,
                                  n_epochs=200, verbose=False)

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


def loo_eval_within_leader(leader_id, samples, cb_embeddings, verbose=True):
    """Standard LOO within a single leader."""
    leader_samps = [s for s in samples if s["leader_id"] == leader_id]
    if len(leader_samps) < 5:
        return None

    if verbose:
        print(f"\n[6] Within-leader LOO for {leader_id} (n={len(leader_samps)})...")

    all_pred_dir = []
    all_true_dir = []
    all_pred_mag = []
    all_true_mag = []

    for test_idx in range(len(leader_samps)):
        train_samps = [s for i, s in enumerate(leader_samps) if i != test_idx]
        test_samp   = leader_samps[test_idx]

        X_train  = np.array([assemble_feature_vector(s, cb_embeddings) for s in train_samps])
        y_dir_tr = np.array([s["direction"] for s in train_samps])
        y_mag_tr = np.array([s["magnitude"] for s in train_samps])
        y_guid_tr= np.array([s["guidance"]  for s in train_samps])
        X_test   = np.array([assemble_feature_vector(test_samp, cb_embeddings)])

        model, _, _ = train_model(X_train, y_dir_tr, y_mag_tr, y_guid_tr,
                                  n_epochs=200, verbose=False)
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


# ─── Final Training on All Data ───────────────────────────────────────────────

def train_final_model(samples, cb_embeddings):
    """Train final model on all samples."""
    print("\n[4] Training final model on ALL samples...")

    X     = np.array([assemble_feature_vector(s, cb_embeddings) for s in samples])
    y_dir = np.array([s["direction"] for s in samples])
    y_mag = np.array([s["magnitude"] for s in samples])
    y_guid= np.array([s["guidance"]  for s in samples])

    model, history, train_time = train_model(X, y_dir, y_mag, y_guid, n_epochs=400)

    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        d_pred, m_pred, _ = model.predict(Xt)
    d_pred = d_pred.cpu().numpy()
    m_pred = m_pred.cpu().numpy()

    train_dir_acc = float(np.mean(d_pred == y_dir))
    train_mag_acc = float(np.mean(m_pred == y_mag))
    print(f"  Training dir_acc={train_dir_acc:.3f}  mag_acc={train_mag_acc:.3f}  "
          f"({train_time:.1f}s on {device})")

    return model, history, X, y_dir, y_mag, y_guid, train_time


def per_leader_accuracy(model, samples, cb_embeddings):
    """Per-leader in-sample accuracy."""
    by_leader = defaultdict(list)
    for s in samples:
        by_leader[s["leader_id"]].append(s)

    results = {}
    for lid, slist in sorted(by_leader.items()):
        X = np.array([assemble_feature_vector(s, cb_embeddings) for s in slist])
        y_dir = np.array([s["direction"] for s in slist])
        y_mag = np.array([s["magnitude"] for s in slist])

        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            d_pred, m_pred, _ = model.predict(Xt)
        d_pred = d_pred.cpu().numpy()
        m_pred = m_pred.cpu().numpy()

        results[lid] = {
            "dir_acc":   float(np.mean(d_pred == y_dir)),
            "mag_acc":   float(np.mean(m_pred == y_mag)),
            "n_samples": len(slist),
            "country":   slist[0]["country"],
        }
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    if not TORCH_OK:
        print("[ERROR] PyTorch required for model training")
        return

    # Step 0: Load economic data
    print("\n[0] Loading economic data...")
    econ = EconomicDataStore()
    print(f"  Loaded {len(econ._data)} indicator series")
    print(f"  BIS policy rates for: {list(econ._bis_rates.keys())}")

    # Step 1: Build political leader embeddings (FIXED, not trained)
    print("\n[1] Building political leader embeddings (FIXED, not trained)...")
    pol_embeddings = build_political_embeddings()

    # Verify active political leaders at key months
    test_months = ["2008-01", "2013-01", "2018-06", "2020-04", "2022-01", "2025-06"]
    print("\n  Political leaders active at key months:")
    for m in test_months:
        active = get_active_political_leaders(m)
        print(f"  {m}: US={active['us_president'] or 'none':10s} "
              f"XI={active['xi'] or 'none':12s} "
              f"PUTIN={active['putin'] or 'none'}")

    # Step 2: Build dataset
    samples, cb_embeddings, stats = build_multi_leader_dataset(econ, pol_embeddings)

    if len(samples) < 20:
        print("[ERROR] Insufficient samples")
        return

    # Step 3: Normalize
    norm, pol_norm = fit_and_normalize(samples)

    # Step 4: Train final model (GPU)
    model, history, X, y_dir, y_mag, y_guid, train_time = train_final_model(
        samples, cb_embeddings
    )

    # Step 5: Per-leader in-sample accuracy
    print("\n[7] Per-leader in-sample accuracy...")
    per_leader = per_leader_accuracy(model, samples, cb_embeddings)

    print(f"\n{'Leader':22s} {'Country':7s} {'N':6s} {'Dir%':8s} {'Mag%':8s}")
    print("-" * 55)
    for lid, res in sorted(per_leader.items(), key=lambda x: x[0]):
        print(f"  {lid:20s} {res['country']:7s} {res['n_samples']:6d} "
              f"{res['dir_acc']*100:.1f}%    {res['mag_acc']*100:.1f}%")

    total_dir_acc = float(np.mean([r["dir_acc"] for r in per_leader.values()]))
    total_mag_acc = float(np.mean([r["mag_acc"] for r in per_leader.values()]))
    print(f"\n  Overall (macro-avg): dir={total_dir_acc:.3f}  mag={total_mag_acc:.3f}")
    print(f"  Total samples: {len(samples)}")

    # Step 6: Powell within-leader LOO
    powell_loo = loo_eval_within_leader("powell", samples, cb_embeddings)

    # Step 7: Leave-One-Leader-Out generalization
    lolo_results = loo_eval_by_leader(samples, cb_embeddings)

    # Step 8: Save model
    model_path = MODELS_DIR / "multi_leader_v3_time.pt"
    torch.save({
        "model_state_dict":  model.state_dict(),
        "n_features":        TOTAL_DIM,
        "n_econ":            N_ECON_FEATURES,
        "n_leader_emb":      LEADER_EMB_DIM,
        "n_context":         N_CONTEXT_FEATURES,
        "n_pol_emb":         N_POL_SLOTS * POL_EMB_DIM,
        "n_pol_action":      N_POL_ACTION,
        "cb_embeddings":     {k: v.tolist() for k, v in cb_embeddings.items()},
        "pol_embeddings":    {k: v.tolist() for k, v in pol_embeddings.items()},
        "normalizer":        norm.to_dict(),
        "pol_action_norm":   pol_norm.to_dict(),
        "history":           history,
        "device_used":       str(device),
        "architecture":      {
            "d_model": 64, "nhead": 8, "n_layers": 3,
            "total_dim": TOTAL_DIM,
            "pol_slots": N_POL_SLOTS,
        },
    }, model_path)
    print(f"\n[8] Model saved: {model_path}")

    # ─── Comparison with v1 ────────────────────────────────────────────────────
    V1_POWELL_LOO_DIR  = 0.6146
    V1_POWELL_LOO_TP   = 0.3939
    V1_OVERALL_DIR_ACC = 0.6646
    V1_AVG_LOLO_DIR    = 0.4879   # computed from v1 eval file

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY  (v3 Time Dimension vs v2/v1)")
    print("=" * 70)

    print(f"\n  Architecture:")
    print(f"    v1: d_model=32, nhead=4, 2 layers, 31D input")
    print(f"    v2: d_model=64, nhead=8, 3 layers, 60D input")
    print(f"    v3: d_model=64, nhead=8, 3 layers, {TOTAL_DIM}D input  [{device}]")
    print(f"    GPU training time: {train_time:.1f}s")

    print(f"\n  Feature breakdown (v3):")
    print(f"    Economic state:       {N_ECON_FEATURES}D")
    print(f"    CB leader embedding:  {LEADER_EMB_DIM}D")
    print(f"    Context (streak):     {N_CONTEXT_FEATURES}D")
    print(f"    US president:         {POL_EMB_DIM}D — Obama/Trump T1/Biden/Trump T2")
    print(f"    Xi Jinping:           {POL_EMB_DIM}D — active 2012-11+")
    print(f"    Putin:                {POL_EMB_DIM}D — active 2000-05+")
    print(f"    Political actions:    {N_POL_ACTION}D — trade/fiscal/VIX/sanctions/credit")
    print(f"    Time features:        {N_TIME_FEATURES}D — v3 NEW (year/tenure/election/decade)")
    print(f"    Total:                {TOTAL_DIM}D")

    print(f"\n  POWELL LOO (within-leader):")
    if powell_loo:
        v2_dir = powell_loo["dir_acc"]
        v2_tp  = powell_loo.get("tp_acc")
        delta_dir = (v2_dir - V1_POWELL_LOO_DIR) * 100
        print(f"    Direction accuracy: {v2_dir:.3f}  "
              f"({delta_dir:+.1f}pp vs v1 {V1_POWELL_LOO_DIR:.3f})")
        if v2_tp is not None:
            delta_tp = (v2_tp - V1_POWELL_LOO_TP) * 100
            print(f"    Turning point acc:  {v2_tp:.3f}  "
                  f"({delta_tp:+.1f}pp vs v1 {V1_POWELL_LOO_TP:.3f})")
    else:
        print("    Powell LOO failed")

    print(f"\n  IN-SAMPLE (all leaders):")
    delta_overall = (total_dir_acc - V1_OVERALL_DIR_ACC) * 100
    print(f"    v2 dir acc: {total_dir_acc:.3f}  "
          f"({delta_overall:+.1f}pp vs v1 {V1_OVERALL_DIR_ACC:.3f})")

    print(f"\n  CROSS-LEADER GENERALIZATION (LOLO):")
    if lolo_results:
        avg_v2_dir = float(np.mean([r["dir_acc"] for r in lolo_results.values()]))
        avg_v2_tp  = float(np.nanmean([r["tp_acc"] for r in lolo_results.values()
                                       if r["tp_acc"] is not None]))
        delta_lolo = (avg_v2_dir - V1_AVG_LOLO_DIR) * 100
        print(f"    Average LOLO dir:  {avg_v2_dir:.3f}  "
              f"({delta_lolo:+.1f}pp vs v1 {V1_AVG_LOLO_DIR:.3f})")
        print(f"    Average LOLO TP:   {avg_v2_tp:.3f}")
        print()
        for lid, r in sorted(lolo_results.items()):
            # v1 LOLO reference
            tp_str = f"{r['tp_acc']:.3f}" if r["tp_acc"] is not None else " N/A"
            print(f"    {lid:22s}: dir={r['dir_acc']:.3f}  tp={tp_str}")

    print(f"\n  Total elapsed: {time.time() - t_start:.1f}s")
    print("=" * 70)

    # ─── Save evaluation ───────────────────────────────────────────────────────
    eval_result = {
        "model":            "multi_leader_v2_political",
        "generated_at":     datetime.now().isoformat(),
        "training_samples": len(samples),
        "device":           str(device),
        "architecture": {
            "d_model":    64,
            "nhead":      8,
            "n_layers":   3,
            "total_dim":  TOTAL_DIM,
            "n_econ":     N_ECON_FEATURES,
            "n_cb_emb":   LEADER_EMB_DIM,
            "n_context":  N_CONTEXT_FEATURES,
            "n_pol_emb":  N_POL_SLOTS * POL_EMB_DIM,
            "n_pol_act":  N_POL_ACTION,
        },
        "political_leaders_added": {
            "us_presidents": ["obama (2009-01→2017-01)",
                              "trump_t1 (2017-01→2021-01)",
                              "biden (2021-01→2025-01)",
                              "trump_t2 (2025-01→present)"],
            "xi_jinping":   "2012-11 to present",
            "putin":        "2000-05 to present",
            "embedding_dim": POL_EMB_DIM,
            "embedding_type": "FIXED (not trained), from behavioral profiles",
            "biden_source":  "hardcoded default (no profile)",
        },
        "v1_comparison": {
            "v1_powell_loo_dir_acc":  V1_POWELL_LOO_DIR,
            "v1_powell_loo_tp_acc":   V1_POWELL_LOO_TP,
            "v1_overall_dir_acc":     V1_OVERALL_DIR_ACC,
            "v1_avg_lolo_dir":        V1_AVG_LOLO_DIR,
            "v1_samples":             1227,
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

    eval_path = OUTPUT_DIR / "multi_leader_v3_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2, default=str)
    print(f"\n[9] Evaluation saved: {eval_path}")


if __name__ == "__main__":
    import os
    os.chdir(BASE)
    main()
